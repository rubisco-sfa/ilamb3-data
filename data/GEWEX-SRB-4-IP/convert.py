from datetime import datetime

import cftime as cf
import earthaccess
import numpy as np
import xarray as xr

from ilamb3_data import (
    create_output_filename,
    gen_trackingid,
    gen_utc_timestamp,
    get_cmip6_variable_info,
    set_coord_bounds,
    set_lat_attrs,
    set_lon_attrs,
    set_ods26_global_attrs,
    set_time_attrs,
    set_var_attrs,
    standardize_dim_order,
)

from ilamb3_data.global_attrs_model import DataGlobalAttrs
from ilamb3_data.cf_attr import CFAttr
import pathlib
import yaml

FILENAME=pathlib.Path(__file__).name
PARENTNAME=pathlib.Path(__file__).parent.name
ILAMB_DATA_URL="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/"

# Download CERES EBAF TOA Edition4.2.1 data from Earthdata
earthaccess.login(strategy="environment")  # You must create an account at https://urs.earthdata.nasa.gov/
granules_sw = earthaccess.search_data(
    short_name='GEWEXSRB_Rel4-IP_Shortwave_monthly_utc',
    downloadable=True,
)

granules_lw = earthaccess.search_data(
    short_name='GEWEXSRB_Rel4-IP_Longwave_monthly_utc',
    downloadable=True,
)
granules = granules_sw + granules_lw

files = earthaccess.download(granules, "_raw")
files.sort()

if len(files) > 1:
    file_range = f"{files[0].name} .. {files[-1].name}"
else:
    file_range = f"{files[0].name}"

# Set timestamps and tracking id
download_stamp = gen_utc_timestamp(files[0].stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

# Open and rename vars
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_mfdataset(
    files, 
    decode_times=time_coder, 
    mask_and_scale=True,
    combine="by_coords",
    join="inner",
)

renaming_dict = {
    "all_sw_dn_sfc": "rsds",
    "all_sw_up_sfc": "rsus",
    "all_lw_dn_sfc": "rlds",
    "all_lw_up_sfc": "rlus",
    "all_sw_diff_sfc": "rsdsdiff",
#    "par": "par",
}
ds = ds.rename(renaming_dict)

# Subset to only variables we care about
vars = list(renaming_dict.values())
ds = ds[vars]

# Get variable attribute info via ESGF CMIP variable information
for var in vars:
    var_info = get_cmip6_variable_info(var, variable_id=var)
    ds = set_var_attrs(
        ds,
        var,
        ds[var].attrs["units"],
        var_info[CFAttr.cf_standard_name.value],
        var_info[CFAttr.variable_long_name.value],
        target_dtype=np.float32,
    )

    # Remove some straggling var attrs
    for attr in ["CF_name", "comment"]:
        ds[var].attrs.pop(attr, None)

# Clean up attrs
ds = set_time_attrs(ds, bounds_frequency="M", ref_date=cf.DatetimeGregorian(1900, 1, 1))
ds = set_lat_attrs(ds)
ds = set_lon_attrs(ds)
ds = set_coord_bounds(ds, "lat")
ds = set_coord_bounds(ds, "lon")
ds = standardize_dim_order(ds)

# read-in yaml

with open("global_attrs.yaml", "r") as f:
    global_attrs = yaml.safe_load(f)

# Set global attributes and export
for var in vars:
    # Create one ds per variable
    to_drop = [
        v
        for v in ds.data_vars
        if (var not in v) and ("time" not in v) and (not v.endswith("_bnds"))
    ]
    var_ds = ds.drop_vars(to_drop)

    global_attrs["dataset"]["tracking_id"] = tracking_id
    global_attrs["dataset"]["variable_id"] = var
    global_attrs["processing"]["creation_date"] = creation_stamp
    global_attrs["processing"]["version"] = f"v{today_stamp}"
    global_attrs["processing"]["history"] = (
        f"{download_stamp}: downloaded {file_range} from Earthdata;\n"
        f"{creation_stamp}: formatted attrs according to obs4MIPs conventions"
    )
    global_attrs["processing"]["source_data_retrieval_date"] = today_stamp
    global_attrs["processing"]["title"] = (
        f"{global_attrs['dataset']['source']} {global_attrs['dataset']['source_version_number']} "
        f"{var} {global_attrs['dataset']['frequency']} data"
    )
    global_attrs["processing"]["processing_code_location"] = f"{ILAMB_DATA_URL}/{PARENTNAME}/{FILENAME}"

    # validate
    validated_global_attrs = DataGlobalAttrs.model_validate(global_attrs)

    # Set global attributes
    out_ds = set_ods26_global_attrs(var_ds, **(validated_global_attrs.flatten()))

    # Prep for export
    out_path = create_output_filename(out_ds.attrs)

    out_ds.to_netcdf(out_path, format="NETCDF4")
