import time
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray as rxr

import ilamb3_data as ild

VAR = "landCoverCat"

RAW_PATH = Path("_raw")
if not RAW_PATH.is_dir():
    RAW_PATH.mkdir()

# --------------------------------------------------------------------------------------
# 1) Download
# --------------------------------------------------------------------------------------

# Download CAVM 2.0 data from Mendeley Data HTML and unzip
remote_source = "https://data.mendeley.com/public-files/datasets/c4xj5rv6kv/files/5223c414-234a-498c-ae08-3100cb38510f/file_downloaded"
local_source = RAW_PATH / "Raster CAVM GIS data.zip"
if not local_source.is_file():
    local_source = ild.download_from_html(remote_source, local_source)
download_stamp = ild.gen_utc_timestamp(local_source.stat().st_mtime)

# --------------------------------------------------------------------------------------
# 2) Read Data
# --------------------------------------------------------------------------------------

# Read TIF and legend CSV
ds = rxr.open_rasterio(
    local_source / "raster_cavm_v1.tif", band_as_variable=True, masked=True
)

# --------------------------------------------------------------------------------------
# 3) Manual cleaning/formatting
# --------------------------------------------------------------------------------------


# Helper function for later
def _clean_flag_description(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(",", "")
    s = s.replace("/", "-")
    s = s.replace(" ", "_")
    return s


# Reproject to lat/lon and rename dimensions/variables
ds = ds.rio.reproject("EPSG:4326")
ds = ds.rename({"band_1": VAR, "x": "lon", "y": "lat"})

# Read CAVM legend and use it to set flag values/meanings and descriptions
legend = local_source / "Raster CAVM legend.csv"
df = pd.read_csv(legend, skiprows=[0, 1], na_filter=False)
# Fill in some missing short descriptions manually
desc_fill = {
    "FW": "Fresh water",
    "SW": "Salt water",
    "GL": "Glacier",
    "NA": "Not arctic",
}
mask = df["Short Description"].str.strip() == ""
df.loc[mask, "Short Description"] = df.loc[mask, "Vegetation Unit"].map(desc_fill)

# --------------------------------------------------------------------------------------
# 4) Automatic cleaning/formatting using ILAMB utilities
# --------------------------------------------------------------------------------------

# Set lat/lon/var attributes
ds = ild.set_lat_attrs(ds)
ds = ild.set_lon_attrs(ds)
ds = ild.set_coord_bounds(ds, "lat")
ds = ild.set_coord_bounds(ds, "lon")

# Decisions about naming derived from https://raw.githubusercontent.com/PCMDI/mip-cmor-tables/refs/heads/main/MIP_variables.json
# Search for "landCoverFrac" to see how I modeled this
ds = ild.set_var_attrs(
    ds,
    VAR,
    units="",
    standard_name="cover_category",
    long_name="Vegetation or Land-Cover Category",
    flag_values=np.unique(ds[VAR].values[~np.isnan(ds[VAR].values)]),
    flag_meanings=df["Vegetation Unit"].to_list(),
    extra_attrs={
        "flag_descriptions": " ".join(
            _clean_flag_description(s) for s in df["Short Description"]
        )
    },
    target_dtype=ds[VAR].dtype,
    compression={"zlib": True, "complevel": 4, "shuffle": True},
    overwrite=True,
)

# Final cleanup
ds = ds.drop_vars("spatial_ref")

# --------------------------------------------------------------------------------------
# 5) Set Obs4MIPs-compliant global attributes and write NetCDF
# --------------------------------------------------------------------------------------

# Set global attributes and write NetCDF
generate_stamp = time.strftime("%Y%m%d")
tracking_id = ild.gen_trackingid()
ds = ild.set_ods26_global_attrs(
    ds,
    activity_id="ILAMB",
    contact="Martha Raynolds (mkraynolds@alaska.edu)",
    creation_date=generate_stamp,
    dataset_contributor="Morgan Steckler",
    doi="https://doi.org/10.17632/c4xj5rv6kv.2",
    frequency="fx",
    grid="1x1 km Lambert Azimuthal Equal Area reprojected to latitude x longitude",
    grid_label="gr",
    has_aux_unc="FALSE",
    history=f"""
{generate_stamp}: "CMORized" data from Raster CAVM 2.0 (downloaded from Mendeley Data)\n
{generate_stamp}: Reprojected to lat/lon and added metadata using ILAMB utilities
""",
    institution="University of Alaska, Fairbanks",
    institution_id="UAF",
    license="https://creativecommons.org/licenses/by-nc/3.0/deed.en",
    nominal_resolution="1 km",
    processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/CAVM-2-0/convert.py",
    product="derived",
    realm="land",
    references="Raynolds, M.K., et al. (2019). A raster version of the Circumpolar Arctic Vegetation Map (CAVM). Remote Sensing of Environment, 232, 111297. https://doi.org/10.1016/j.rse.2019.111297",
    region="panarctic",
    source="Unsupervised classifications of seventeen geographic/floristic sub-sections of the Arctic, using AVHRR and MODIS data (reflectance and NDVI) and elevation data",
    source_id="CAVM-2-0",
    source_data_retrieval_date=download_stamp,
    source_data_url=str(remote_source),
    source_label="CAVM",
    source_type="satellite_retrieval",
    source_version_number="2.0",
    title="Raster Circumpolar Arctic Vegetation Map",
    tracking_id=tracking_id,
    variable_id=VAR,
    variant_label="ILAMB",
    variant_info="CMORized product prepared by ILAMB",
    version=f"v{generate_stamp}",
)
out_path = ild.create_output_filename(ds.attrs)
ds.to_netcdf(out_path)
