import glob
import os
import zipfile
from datetime import datetime
from pathlib import Path

import numpy as np
import requests
import xarray as xr

from ilamb3_data import (
    create_output_filename,
    download_from_zenodo,
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

# Set download parameters
dataset_title = "Global Fire Emissions Database (GFED5) Burned Area"
params = {"q": f'title:"{dataset_title}"'}
remote_source = "https://zenodo.org/api/records"

# Query Zenodo for the dataset
response = requests.get(remote_source, params=params)
if response.status_code != 200:
    print("Error during search:", response.status_code)
    exit(1)

# Parse the JSON response
data = response.json()

# Select the first record; there should only be one
records = data["hits"]["hits"]
i = 0
for record in records:
    i += 1
    title = record["metadata"].get("title")
    print(f"\n{i}. Dataset title: {title}")
record = data["hits"]["hits"][0]

# Download
download_from_zenodo(record, download_dir="_raw")

# Unzip downloaded data file (BA.zip)
path_to_zip_file = Path("_raw/BA.zip")
full_path = os.path.abspath(path_to_zip_file)
full_path_without_zip, _ = os.path.splitext(full_path)
with zipfile.ZipFile(path_to_zip_file, "r") as zip_ref:
    zip_ref.extractall(full_path_without_zip)

# Set timestamps and tracking id
download_stamp = gen_utc_timestamp(path_to_zip_file.stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

######################################################################
# Open netcdfs
######################################################################

# Get a list of all netCDF files in the unzipped folder
data_dir = "_raw/BA"
all_files = glob.glob(os.path.join(data_dir, "*.nc"))

# Get separate lists for coarse data and fine data
coarse_files = []  # For 1997-2000 (1 degree)
fine_files = []  # For 2001-2020 (0.25 degree)
for f in all_files:
    basename = os.path.basename(f)  # e.g., "BA200012.nc"
    # Extract the year from the filename; characters at positions 2:6.
    year = int(basename[2:6])
    if year < 2001:
        coarse_files.append(f)
    else:
        fine_files.append(f)

# Load the coarse and fine datasets separately
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds_coarse = xr.open_mfdataset(
    coarse_files, combine="by_coords", decode_times=time_coder
)
ds_fine = xr.open_mfdataset(fine_files, combine="by_coords", decode_times=time_coder)

# Load burnable area (and mask) datasets
da_coarse_mask = xr.open_dataset(
    "_raw/BurnableArea_preMOD.nc", decode_times=time_coder
)["BurableArea"]  # note the mispelling of Burnable in the original file
da_fine_mask = xr.open_dataset("_raw/BurnableArea.nc", decode_times=time_coder)[
    "BurableArea"
]

######################################################################
# Process netcdfs
######################################################################

# Calculate burned fraction of burnable area as a percent
percent_burned_coarse = (ds_coarse["Total"] / da_coarse_mask) * 100
ds_coarse = ds_coarse.assign({"burntFractionAll": percent_burned_coarse})
percent_burned_fine = (ds_fine["Total"] / da_fine_mask) * 100
ds_fine = ds_fine.assign({"burntFractionAll": percent_burned_fine})

# Mask the datasets
percent_burned_coarse_masked = ds_coarse.where(da_coarse_mask > 0)
percent_burned_fine_masked = ds_fine.where(da_fine_mask > 0)

# Interpolate coarse 1 degree data to 0.25 degrees
res = 0.25
newlon = np.arange(-180, 180, res)
newlat = np.arange(-90, 90, res)
percent_burned_coarse_masked_interp = percent_burned_coarse_masked.interp(
    method="nearest", lat=newlat, lon=newlon
)

# Combine coarse-interpolated and fine data into one dataset at 0.25 degree resolution
ds = xr.concat(
    [percent_burned_fine_masked, percent_burned_coarse_masked_interp], dim="time"
)
ds = ds.sortby("time")
ds = ds["burntFractionAll"].to_dataset()

######################################################################
# Set CF compliant netcdf attributes
######################################################################

# Clean up attrs
ds = set_time_attrs(ds, bounds_frequency="M")
ds = set_lat_attrs(ds)
ds = set_lon_attrs(ds)
ds = set_coord_bounds(ds, "lat")
ds = set_coord_bounds(ds, "lon")
ds = standardize_dim_order(ds)

# Get variable attribute info via ESGF CMIP variable information
var_info = get_cmip6_variable_info("burntFractionAll")
ds = set_var_attrs(
    ds,
    var="burntFractionAll",
    cmip6_units=var_info["variable_units"],
    cmip6_standard_name=var_info["cf_standard_name"],
    cmip6_long_name=var_info["variable_long_name"],
    target_dtype=np.int8,
    convert=False,
)

# Set global attributes and export
out_ds = set_ods26_global_attrs(
    ds,
    comment="Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet",
    contact="Thomas Nagler (thomas.nagler@enveo.at)",
    creation_date=creation_stamp,
    dataset_contributor="Morgan Steckler",
    doi="10.5281/zenodo.7668424",
    frequency="mon",
    grid="0.25x0.25 degree latitude x longitude",
    grid_label="gr",
    has_aux_unc="FALSE",
    history=f"""
{download_stamp}: downloaded {remote_source}; 
{creation_stamp}: calculate burntFractionAll as percent of burnable area;
{creation_stamp}: interpolate coarse data to 0.25 degree using nearest neighbor;
{creation_stamp}: formatted attrs according to obs4MIPs conventions""",
    institution="National Aeronautics and Space Administration, Netherlands Organisation for Scientific Research",
    institution_id="NASA-NOSR",
    license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY- 4.0) License (https://creativecommons.org/licenses/).",  # OG license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
    nominal_resolution="25 km",
    processing_code_location="https://github.com/ilamb/ilamb3-data/tree/main/data/GFED-5-0/convert.py",
    product="observations",
    realm="land",
    references="Chen, Y., Hall, J., van Wees, D., Andela, N., Hantson, S., Giglio, L., van der Werf, G. R., Morton, D. C., and Randerson, J. T.: Multi-decadal trends and variability in burned area from the fifth version of the Global Fire Emissions Database (GFED5), Earth Syst. Sci. Data, 15, 5227-5259, https://doi.org/10.5194/essd-15-5227-2023, 2023.",
    region="global_land",
    source="Monthly 24-year record of global burned area created by combining MODIS, Landsat, and Sentinel-2 satellite observations",
    source_id="GFED-5-0",
    source_data_retrieval_date=download_stamp,
    source_data_url="https://zenodo.org/record/7668424",
    source_label="GFED",
    source_type="satellite_retrieval",
    source_version_number="5.0",
    title="Global Fire Emissions Database (GFED5) Burned Area",
    tracking_id=tracking_id,
    variable_id="burntFractionAll",
    variant_label="REF",
    variant_info="CMORized product prepared by ILAMB and CMIP IPO",
    version=f"v{today_stamp}",
)

# Prep for export
out_path = create_output_filename(ds.attrs)
ds.to_netcdf(out_path, format="NETCDF4")
