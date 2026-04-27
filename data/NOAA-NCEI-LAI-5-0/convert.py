import zipfile
from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr

import ilamb3_data as ild

"""
This data was downloaded from https://drive.google.com/drive/folders/1jiFVlLarFb0m6lI3CvqKO9pgdBbTcQX4?usp=sharing
"""

VAR = "lai"

# unzip the data we downloaded from the Google Drive
zip_path = Path(__file__).parent / "_raw" / "LAI_regridded-20260423T194123Z-3-001.zip"
extract_dir = zip_path.with_suffix("")


def unzip(z: Path, extract_dir: Path) -> Path:
    if zipfile.is_zipfile(z) and not extract_dir.is_dir():
        with zipfile.ZipFile(z, "r") as zf:
            zf.extractall(extract_dir)
    return extract_dir


unzip(zip_path, extract_dir)

# determine when the data were downloaded
files = sorted(extract_dir.rglob("*.nc"))
download_stamp = ild.gen_utc_timestamp(Path(files[0]).stat().st_mtime)

# open the data as one xarray dataset
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_mfdataset(files, combine="by_coords", decode_times=time_coder)

# fix up dims/coords/attrs
ds = ild.set_time_attrs(
    ds, bounds_frequency="M", ref_date=cf.DatetimeGregorian(1981, 6, 1)
)
ds = ild.set_lat_attrs(ds)
ds = ild.set_lon_attrs(
    ds, to_180=True
)  # data were in 0-360, which might be fine but unsure, so I changed to -180-180
ds = ild.set_coord_bounds(ds, "lat")
ds = ild.set_coord_bounds(ds, "lon")

# fix up variable attributes
var_info = ild.get_cmip6_variable_info(VAR, VAR)
ds = ild.set_var_attrs(
    ds,
    VAR,
    units=var_info["variable_units"],
    standard_name=var_info["cf_standard_name"],
    long_name=var_info["variable_long_name"],
    target_dtype=np.dtype("float32"),
    convert=False,
    cell_methods="area: mean where land time: mean",
)
for attr in ["cell_measures", "comment"]:
    ds[VAR].attrs.pop(attr, None)

# set some global attributes
ds = ild.set_ods26_global_attrs(
    ds,
    comment=ds.attrs["comment"],
    contact="Jessica Matthews (jessica.matthews@noaa.gov)",
    creation_date=ild.gen_utc_timestamp(),
    dataset_contributor=ds.attrs["dataset_contriubtor"],
    doi=ds.attrs["doi"],
    frequency=ds.attrs["frequency"],
    grid="data re-gridded to 0.5x0.5 degree latxlon grid from the native 0.05x0.05 deg latxlon grid using bilinear interpolation",
    grid_label="gr",
    has_aux_unc="FALSE",
    history=f"""
2026-01-08T00:12:57Z ; CMOR rewrote data to be consistent with CMIP6, CF-1.11; ODS-2.5 and CF standards.;\nCreated via CMOR
{download_stamp}: downloaded from Google Drive;
{ild.gen_utc_timestamp()}: converted to obs4MIPs format""",
    institution=ds.attrs["institution"],
    institution_id=ds.attrs["institution_id"],
    license="Users must cite this dataset when used as a source. No other constraints on data access or use.",
    nominal_resolution="0.5 degree",
    processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/NOAA-NCEI-LAI-5-0/convert.py",
    product=ds.attrs["product"],
    realm=ds.attrs["realm"],
    references="Vermote, Eric; NOAA CDR Program. (2019): NOAA Climate Data Record (CDR) of AVHRR and VIIRS Leaf Area Index (LAI) and Fraction of Absorbed Photosynthetically Active Radiation (FAPAR), Version 5. NOAA National Centers for Environmental Information. doi:10.7289/V5TT4P69. Accessed 2019-09-11.",
    region=ds.attrs["region"],
    source="NOAA-NCEI-LAI-5-0 (2019): AVHRR-VIIRS Leaf Area Index",
    source_data_retrieval_date=download_stamp,
    source_data_url=ds.attrs["source_data_url"],
    source_id="NOAA-NCEI-LAI-5-0",
    source_label="NOAA-NCEI-LAI",
    source_type=ds.attrs["source_type"],
    source_version_number=ds.attrs["source_version_number"],
    table_id=ds.attrs["table_id"],
    title="AVHRR-VIIRS Leaf Area Index",
    tracking_id=ds.attrs["tracking_id"],
    variable_id=VAR,
    variant_label="NOAA-NCEI",
    variant_info="CMORized product prepared by NOAA-NCEI",
)

# create output filename and save the data
out_path = ild.create_obs4mips26_filename(ds, ds.attrs)
ds.to_netcdf(out_path)
