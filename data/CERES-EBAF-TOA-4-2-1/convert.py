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

# Download CERES EBAF TOA Edition4.2.1 data from Earthdata
earthaccess.login()  # You must create an account at https://urs.earthdata.nasa.gov/
granules = earthaccess.search_data(
    short_name="CERES_EBAF",
    granule_name="*200003-202509.nc",
)
files = earthaccess.download(granules, "_raw")

# Set timestamps and tracking id
download_stamp = gen_utc_timestamp(files[0].stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

# Open and rename vars
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_dataset(files[0], decode_times=time_coder, mask_and_scale=True)
renaming_dict = {
    "sfc_sw_down_all_mon": "rsds",
    "sfc_sw_up_all_mon": "rsus",
    "sfc_lw_down_all_mon": "rlds",
    "sfc_lw_up_all_mon": "rlus",
    "sfc_net_sw_all_mon": "rss",
    "sfc_net_lw_all_mon": "rls",
    # "sfc_net_tot_all_mon": "",  # surface_net_downward_radiative_flux
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
        var_info["cf_standard_name"],
        var_info["variable_long_name"],
        target_dtype=np.float32,
    )

    # Remove some straggling var attrs
    for attr in ["CF_name", "comment"]:
        ds[var].attrs.pop(attr, None)

# Clean up attrs
ds = set_time_attrs(ds, bounds_frequency="M", ref_date=cf.DatetimeGregorian(2000, 3, 1))
ds = set_lat_attrs(ds)
ds = set_lon_attrs(ds)
ds = set_coord_bounds(ds, "lat")
ds = set_coord_bounds(ds, "lon")
ds = standardize_dim_order(ds)

# Set global attributes and export
for var in vars:
    # Create one ds per variable
    to_drop = [
        v
        for v in ds.data_vars
        if (var not in v) and ("time" not in v) and (not v.endswith("_bnds"))
    ]
    var_ds = ds.drop_vars(to_drop)

    # Set global attributes
    out_ds = set_ods26_global_attrs(
        var_ds,
        contact="Seiji Kato (seiji.kato@nasa.gov)",
        creation_date=creation_stamp,
        dataset_contributor="Morgan Steckler",
        doi="https://doi.org/10.5067/TERRA-AQUA-NOAA20/CERES/EBAF_L3B004.2.1",
        frequency="mon",
        grid="1x1 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="FALSE",
        history=f"""
{download_stamp}: downloaded {files[0].name} from Earthdata;
{creation_stamp}: formatted attrs according to obs4MIPs conventions
""",
        institution="NASA-LaRC (Langley Research Center) Hampton, Va",
        institution_id="NASA-LaRC",
        license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution - 4.0 International (CC BY - 4.0) License (https://creativecommons.org/licenses/).",
        nominal_resolution="1 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/CERES-4-2-1/convert.py",
        product="derived",
        realm="atmos",
        references="Kato, S., F. G. Rose, D. A. Rutan, T. E. Thorsen, N. G. Loeb, D. R. Doelling, X. Huang, W. L. Smith, W. Su, and S.-H. Ham, 2018: Surface irradiances of Edition 4.0 Clouds and the Earth's Radiant Energy System (CERES) Energy Balanced and Filled (EBAF) data product, J. Climate, 31, 4501-4527, doi: 10.1175/JCLI-D-17-0523.1.",
        region="global",
        source="Data are collected on Terra, Aqua, Suomi National Polar-Orbiting Partnership (SNPP), and NOAA-20 satellites, then an objective constrainment algorithm is applied to adjust SW and LW fluxes within their ranges of uncertainty",
        source_data_retrieval_date=today_stamp,
        source_data_url="https://asdc.larc.nasa.gov/data/CERES/EBAF/Edition4.2.1/",
        source_id="CERES-EBAF-4-2-1",
        source_label="CERES_EBAF",
        source_type="satellite_blended",
        source_version_number="4.2.1",
        title=f"CERES EBAF Surface Edition 4.2.1 {var} monthly mean data",
        tracking_id=tracking_id,
        variable_id=var,
        variant_label="ILAMB",
        variant_info="CMORized product prepared by ILAMB",
        version=f"v{today_stamp}",
    )

    # Prep for export
    out_path = create_output_filename(out_ds.attrs)
    out_ds.to_netcdf(out_path, format="NETCDF4")
