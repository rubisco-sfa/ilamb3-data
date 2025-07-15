from datetime import datetime
from pathlib import Path

import cf_xarray  # noqa
import cftime  # noqa
import numpy as np
import xarray as xr

from ilamb3_data import (
    create_output_filename,
    download_from_html,
    gen_trackingid,
    gen_utc_timestamp,
    set_ods_global_attrs,
    set_time_attrs,
    set_var_attrs,
)

# Download source
remote_source = "https://www.ncei.noaa.gov/data/oceans/woa/DATA_ANALYSIS/3M_HEAT_CONTENT/NETCDF/heat_content/heat_content_anomaly_0-2000_yearly.nc"  # Redirected from: https://www.ncei.noaa.gov/access/global-ocean-heat-content/heat_global.html
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
source = local_source / Path(remote_source).name
if not source.is_file():
    download_from_html(remote_source, str(source))

# Set timestamps and tracking id
download_stamp = gen_utc_timestamp(source.stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

# Load the dataset for adjustments
ds = xr.open_dataset(source, decode_times=False, engine="netcdf4")
ds = ds.isel(depth=0)  # drop redundant depth
ds = set_time_attrs(ds, bounds_frequency="Y")

# Fix units naming
ds.h18_hc.attrs["units"] = "1e18 J"

# Compute OHC values over cell area
radius = ds.crs.semi_major_axis
lat_bnds = np.deg2rad(ds.lat_bnds.values)  # (lat,2)
lon_bnds = np.deg2rad(ds.lon_bnds.values)  # (lon,2)
delta_phi = np.sin(lat_bnds[:, 1]) - np.sin(lat_bnds[:, 0])
delta_lon = lon_bnds[:, 1] - lon_bnds[:, 0]
area = (radius**2) * delta_phi[:, None] * delta_lon[None, :]  # (lat,lon)
cell_area = xr.DataArray(
    area, dims=("lat", "lon"), coords={"lat": ds.lat, "lon": ds.lon}, name="cell_area"
)

# Set 10e18 J to just J
hc = ds["h18_hc"] * 1e18
ohc = hc / cell_area
ohc.name = "ohc"
ohc.attrs = {"cell_methods": "area: mean time: mean"}
ds["ohc"] = ohc

# Set variable attributes
ds = set_var_attrs(
    ds,
    var="ohc",  # I think a lat/lon global mean should be ohc_global maybe?
    cmip6_units="J m-2",
    cmip6_standard_name="integral_wrt_depth_of_sea_water_potential_temperature_expressed_as_heat_content",
    cmip6_long_name="ocean heat content anomaly (0-2000 m) relative to the WOA09 1955-2006 climatology",
    target_dtype=np.float32,
    convert=False,
)

# Clean up attrs
for var in ds.variables:
    ds[var].encoding.pop("missing_value", None)
ds["ohc"].attrs.pop("bounds", None)

# Set global attributes and export
for var in ["ohc"]:
    # Create one ds per variable
    out_ds = ds.drop_vars([v for v in ds if (var not in v and "time" not in v)])

    # Set global attributes
    out_ds = set_ods_global_attrs(
        out_ds,
        activity_id="obs4MIPs",
        aux_variable_id="N/A",
        comment="Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet",
        contact="ncei.info@noaa.gov",
        conventions="CF-1.12 ODS-2.5",
        creation_date=creation_stamp,
        dataset_contributor="Morgan Steckler",
        data_specs_version="2.5",
        doi="N/A",
        external_variables="N/A",
        frequency="yr",
        grid="mean of area, depth, and time to 1x1 degree latxlon grid",
        grid_label="gn",
        has_auxdata="False",
        history=f"""
{download_stamp}: downloaded {remote_source};
{creation_stamp}: converted to obs4MIPs format""",
        institution="National Oceanic and Atmospheric Administration, National Centers for Environmental Information, Ocean Climate Laboratory, Asheville, NC, USA",
        institution_id="NOAA-NCEI-OCL",
        license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
        nominal_resolution="1x1 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/NOAA/convert.py",
        product="observations",
        realm="ocean",
        references="S. Levitus, J. I. Antonov, T. P. Boyer, O. K. Baranova, H. E. Garcia, R. A. Locarnini, A. V. Mishonov, J. R. Reagan, D. Seidov, E. S. Yarosh, M. M. Zweng, World ocean heat content and thermosteric sea level change (0-2000 m), Geophysical Research Letters. 1955-2010. 10.1029/2012GL051106",
        region="global_ocean",
        source="WOA23 yearly mean ocean heat content anomaly from in-situ profile data",
        source_id="WOA-23",
        source_data_retrieval_date=download_stamp,
        source_data_url=remote_source,
        source_label="WOA09",
        source_type="gridded_insitu",
        source_version_number="1.0",
        title="WOA23 yearly mean ocean heat content (0-2000m) anomaly (WOA09 1955-2006 anomaly baseline) from in-situ profile data",
        tracking_id=tracking_id,
        variable_id="ohc",
        variant_label="REF",
        variant_info="CMORized product prepared by ILAMB and CMIP IPO",
        version=f"v{today_stamp}",
    )

    # Prep for export
    out_path = create_output_filename(out_ds.attrs)
    out_ds.to_netcdf(out_path, format="NETCDF4")
