from pathlib import Path

import cf_xarray  # noqa
import cftime
import numpy as np
import xarray as xr

from ilamb3_data import download_from_html, gen_utc_timestamp, set_time_attrs

# Download source
remote_source = "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/NETCDF/heat_content/heat_content_anomaly_0-2000_yearly.nc"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / Path(remote_source).name
if not local_source.is_file():
    download_from_html(remote_source, str(local_source))
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()

# Load the dataset for adjustments
ds = xr.load_dataset(local_source, decode_times=False)

# Recode time to be CF-compliant
years = (ds["time"].astype(int) - 6) // 12 + 1955
ds["time"] = [cftime.DatetimeNoLeap(y, 6, 1) for y in years]
ds["time_bnds"] = (
    ("time", "nbounds"),
    np.asarray(
        [
            [cftime.DatetimeNoLeap(y, 1, 1) for y in years],
            [cftime.DatetimeNoLeap(y + 1, 1, 1) for y in years],
        ]
    ).T,
)

# Drop things we don't need
ds = ds.drop_vars(["crs", "h18_hc", "basin_mask", "climatology_bounds", "depth_bnds"])
vs = [v for v in ds if v.startswith("yearl") and "_se_" not in v]
ds["ohc"] = xr.concat(
    [ds[v].assign_coords(region=v.split("_")[-1]) for v in vs], dim="region"
)
ds = ds.drop_vars(vs)
vs = [v.replace("_h22_", "_h22_se_") for v in vs]
ds["ohc_uncert"] = xr.concat(
    [ds[v].assign_coords(region=v.split("_")[-1]) for v in vs], dim="region"
)
ds = ds.drop_vars(vs)
ds = ds.sel(region="WO", drop=True)
ds["ohc"].attrs = {
    "standard_name": "ocean_heat_content_anomaly",
    "long_time": "ocean_heat_content_anomaly_from_2005_to_2000m",
    "units": "ZJ",
    "ancillary_variables": "ohc_uncert",
}
ds["ohc_uncert"].attrs = {
    "standard_name": "ocean_heat_content_anomaly standard_error",
    "long_time": "ocean_heat_content_anomaly_uncertainty_from_2005_to_2000m",
    "units": "ZJ",
}

# Fix up the dimensions
ds = set_time_attrs(ds)
ds["time"].attrs["bounds"] = "time_bnds"
ds["time_bnds"].attrs["long_name"] = "time_bounds"
time_range = f"{ds["time"].min().dt.year:d}{ds["time"].min().dt.month:02d}"
time_range += f"-{ds["time"].max().dt.year:d}{ds["time"].max().dt.month:02d}"

# Populate attributes
ds.attrs = {
    "activity_id": "obs4MIPs",
    "contact": "",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_data": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "2.5",
    "doi": "",
    "frequency": "yr",
    "grid": "",
    "grid_label": "gm",
    "history": f"""
{download_stamp}: downloaded {remote_source};
{generate_stamp}: converted to obs4MIP format""",
    "institution": "",
    "institution_id": "NOAA",
    "license": "",
    "nominal_resolution": "",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/NOAA/convert.py",
    "product": "observations",
    "realm": "ocean",
    "references": "S. Levitus, J. I. Antonov, T. P. Boyer, O. K. Baranova, H. E. Garcia, R. A. Locarnini, A. V. Mishonov, J. R. Reagan, D. Seidov, E. S. Yarosh, M. M. Zweng, World ocean heat content and thermosteric sea level change (0-2000 m), Geophysical Research Letters. 1955-2010. 10.1029/2012GL051106",
    "region": "global_ocean",
    "source": "",
    "source_id": "OHC",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": remote_source,
    "source_type": "",
    "source_version_number": "1",
    "title": "Ocean Heat Content Anomaly from 2005 to 2000m",
    "variable_id": "ohc",
    "variant_label": "BE",
}

# Write out files
ds.to_netcdf(
    "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
        **ds.attrs, time_mark=time_range
    ),
)
