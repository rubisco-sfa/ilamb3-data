import glob
import os
import warnings
import zipfile
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import requests
import xarray as xr
from matplotlib.colors import LogNorm

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

warnings.filterwarnings(
    "ignore",
    message="invalid value encountered in divide",
    category=RuntimeWarning,
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
print(f"\nSelected dataset: {record['metadata'].get('title')}")
download_from_zenodo(record, download_dir="_raw")

# Unzip downloaded ZIP files
raw_dir = Path("_raw")
for zip_path in raw_dir.glob("*.zip"):
    full_path = zip_path.resolve()
    out_dir = full_path.with_suffix("")  # strip .zip
    with zipfile.ZipFile(full_path, "r") as zip_ref:
        zip_ref.extractall(out_dir)

# Set timestamps and tracking id
latest_zip = max(raw_dir.glob("*.zip"), key=lambda p: p.stat().st_mtime)
download_stamp = gen_utc_timestamp(latest_zip.stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

######################################################################
# Open netcdfs
######################################################################

# Get a list of all netCDF files in the unzipped folder
nc_dir = raw_dir / "BA"
all_files = glob.glob(os.path.join(nc_dir, "*.nc"))

# Get separate lists for coarse data and fine data
files_97_00 = []  # For 1997-2000 (1 degree)
files_01_20 = []  # For 2001-2020 (0.25 degree)
for f in all_files:
    basename = os.path.basename(f)  # e.g., "BA200012.nc"
    # Extract the year from the filename; characters at positions 2:6
    year = int(basename[2:6])
    if year < 2001:
        files_97_00.append(f)
    else:
        files_01_20.append(f)
files_97_00 = sorted(files_97_00)
files_01_20 = sorted(files_01_20)

# Load the coarse and fine datasets separately
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds_97_00 = xr.open_mfdataset(files_97_00, decode_times=time_coder)
ds_01_20 = xr.open_mfdataset(files_01_20, decode_times=time_coder)

# Load burnable area (and mask) datasets
# Note: sum of 0.25deg burnable area == sum of 1deg burnable area
ba_97_00 = xr.open_dataset("_raw/BurnableArea_preMOD.nc", decode_times=time_coder)[
    "BurableArea"
]  # note the mispelling of Burnable...
ba_01_20 = xr.open_dataset("_raw/BurnableArea.nc", decode_times=time_coder)[
    "BurableArea"
]

# Drop 0 values (non-burnable areas)
ba_97_00 = ba_97_00.where(ba_97_00 > 0)
ba_01_20 = ba_01_20.where(ba_01_20 > 0)

######################################################################
# Process netcdfs
######################################################################

# Coarse data (1997-2000)
# -----------------------

# Set target grid for interpolation (0.25 degree)
newlat = ba_01_20["lat"]
newlon = ba_01_20["lon"]

# Map coarse burned area and coarse burnable area onto .25deg grid
# Nearest is fine because the 1deg and 0.25deg grids are aligned
ds_97_00 = ds_97_00["Total"].interp(lat=newlat, lon=newlon, method="nearest")
ba_97_00 = ba_97_00.interp(lat=newlat, lon=newlon, method="nearest")

# Downscale burned area (km2 -> km2 on 0.25deg) & convert to percent of burnable area
# 1deg data at 0.25deg cells * burnable area fraction of 1deg cell
tot_ds_97_00 = ds_97_00 * xr.where(ba_97_00 > 0, ba_01_20 / ba_97_00, np.nan)
pct_ds_97_00 = xr.where(ba_01_20 > 0, 100.0 * tot_ds_97_00 / ba_01_20, np.nan)
ds_97_00 = pct_ds_97_00.to_dataset(name="burntFractionAll")

# Fine data (2001-2020)
# ---------------------

pct_ds_01_20 = (ds_01_20["Total"] / ba_01_20) * 100
ds_01_20 = ds_01_20.assign({"burntFractionAll": pct_ds_01_20})

# For some reason, one single gridcell of bFA is > 100%; so mask it
ds_01_20 = ds_01_20.where(ds_01_20["burntFractionAll"] <= 100)

# Combine coarse-interpolated and fine data into one dataset at 0.25 degree resolution
da = xr.concat([ds_01_20["burntFractionAll"], ds_97_00["burntFractionAll"]], dim="time")
da = da.sortby("time")
ds = da.to_dataset()

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
    target_dtype=np.float32,
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
out_path = create_output_filename(out_ds.attrs)
out_ds.to_netcdf(out_path, format="NETCDF4")

######################################################################
# Plotting verification
######################################################################

# Temporal mean (I think I can assume consistent monthly time steps, so no integral)
mean_da = out_ds["burntFractionAll"].mean(dim="time") * 12  # annual mean
mean_da = mean_da.where(mean_da > 0)
plt.figure(figsize=(10, 5))
plt.pcolormesh(
    mean_da["lon"],
    mean_da["lat"],
    mean_da,
    norm=LogNorm(vmin=0.1, vmax=100),
    cmap="jet",
)
cbar = plt.colorbar()
cbar.set_label("Burned fraction (%)")
plt.tight_layout()

out_path = raw_dir.parent / "mean_burntFractionAll.png"
plt.savefig(out_path, dpi=300)
plt.close()
