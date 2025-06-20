from pathlib import Path
from datetime import datetime

import cf_xarray  # noqa
import cftime as cf
import numpy as np
import xarray as xr
import os

from ilamb3_data import (
    add_time_bounds_monthly,
    download_file,
    fix_lat,
    fix_lon,
    set_time_attrs,
    gen_utc_timestamp,
    get_cmip6_variable_info,
    set_lat_attrs,
    set_lon_attrs,
    set_ods_coords,
    set_ods_global_attributes,
    set_ods_var_attrs,
    set_ods_coords,
    set_ods_calendar,
    add_time_bounds
)

today = datetime.now().strftime("%Y%m%d")

RENAME = {"GPP": "gpp", "H": "hfss", "LE": "hfls"}


# Download source
remote_source = "https://avdc.gsfc.nasa.gov/pub/data/project/WECANN/WECANN_v1.0.nc"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / Path(remote_source).name
if not local_source.is_file():
    download_from_html(remote_source, str(local_source))
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()
generate_trackingid = gen_trackingid()

# Build up a new dataset
ds = xr.open_dataset(local_source)
coords = {}

# Time is encoded as a 2D character array [["2022/01",...]]
coords["time"] = [
    cf.DatetimeGregorian(int("".join(t[:4])), int("".join(t[-2:])), 15)
    for t in ds["Time"].astype(str).values.T
]

# Latitude and Longitude are 2D but have constant values in other dimension
coords["lat"] = ds["Latitude"][0, :].values
coords["lon"] = ds["Longitude"][:, 0].values

# Load the dataarrays, rename to CMOR variables, change dimension order
data = {RENAME[v]: ds[v].transpose("t", "lat", "lon").rename(t="time") for v in RENAME}

# Carbon 'units' are not CF compliant
for var, da in data.items():
    data[var].attrs["units"] = da.attrs["Units"].replace("gC", "g")
    data[var].attrs.pop("Units")

# Populate information from CMIP6 variable information
for var, da in data.items():
    info = get_cmip6_variable_info(var)
    data[var].attrs["long_name"] = info["variable_long_name"]

# Encode the dataset
out = xr.Dataset(data_vars=data, coords=coords)

# Fix up the dimensions
out["lat"] = fix_lat(out)
out["lon"] = fix_lon(out)
out = out.sortby(["time", "lat", "lon"])
out = out.cf.add_bounds(["lat","lon"])
out = add_time_bounds(out)
out = set_time_attrs(out)
out["time"].encoding.update({
        "units": "days since 2007-01-16 00:00:00",
        "calendar": "standard"})
for v in ["time", "time_bounds"]:
        for attr in ["units", "calendar", "_FillValue"]:
            if attr in out[v].attrs:
                del dsv[v].attrs[attr]
#out['time_bnds'] = out.time_bnds.astype('float64')
time_range = f"{out['time'].min().dt.year:d}{out['time'].min().dt.month:02d}"
time_range += f"-{out['time'].max().dt.year:d}{out['time'].max().dt.month:02d}"
# Populate attributes
attrs = {
    "activity_id": "obs4MIPs",
    "contact": "HAlemohammad@clarku.edu",
    "Conventions": "CF-1.12 ODS-2.5",
    "comment":"Not yet obs4MIPs compliant: 'version' attribute is temporary; Cell measure variable areacella referred to by variable is not present in dataset or external variables",
    "creation_date": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "ODS2.5",
    "doi": "N/A",
    "frequency": "mon",
    "grid": "1x1 degree",
    "grid_label": "gn",
    "has_auxdata": "False",
    "history": """
%s: downloaded %s;
%s: converted to obs4MIP format"""
    % (
        download_stamp,
        remote_source,
        generate_stamp,
    ),
    "institution": "Columbia University, NY, USA",
    "institution_id": "ColumbiaU",
    "license": "Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
    "nominal_resolution": "100 km",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/WECANN/convert.py",
    "product": "derived",
    "realm": "land",
    "references": "Alemohammad, S. H., Fang, B., Konings, A. G., Aires, F., Green, J. K., Kolassa, J., Miralles, D., Prigent, C., and Gentine, P.: Water, Energy, and Carbon with Artificial Neural Networks (WECANN): a statistically based estimate of global surface turbulent fluxes and gross primary productivity using solar-induced fluorescence, Biogeosciences, 14, 4101–4124, https://doi.org/10.5194/bg-14-4101-2017, 2017.",
    "region": "global_land",
    "source": "WECANN 1.0 (2018): Water, Energy, and Carbon with Artificial Neural Networks",
    "source_id": "WECANN-1-0",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": remote_source,
    "source_label": "WECANN",
    "source_type": "satellite_retrieval",
    "source_version_number": "1",
    "variant_label":"REF",
    "variant_info":"CMORized product prepared by ILAMB and CMIP IPO",
    "version":today,
    "title": "Water, Energy, and Carbon with Artificial Neural Networks (WECANN)",
    "tracking_id": generate_trackingid,
}

# Write out files
for var, da in out.items():
    dsv = da.to_dataset()
    
    
    dsv = out.drop_vars(set(out) - set([var]))
    #dsv = add_time_bounds_monthly(dsv)
    dsv.attrs = attrs | {"variable_id": var}
    dsv = set_ods_global_attributes(dsv, **dsv.attrs)
    dsv = set_ods_var_attrs(dsv, var)
    dsv[var] = dsv[var].astype(np.float32)
    dsv = set_ods_coords(dsv)

    out_path = (
    "/home/users/dhegedus/CMIPplacement/beta/{activity_id}/{institution_id}/{source_id}/{frequency}/{variable_id}/   {grid_label}/"
        + today
        + "/{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc"
        ).format(**dsv.attrs, time_mark=time_range)

    # Ensure the output directory exists
    os.makedirs(os.path.dirname(out_path), exist_ok=True)


    dsv.to_netcdf(out_path,
        encoding={'lat': {'zlib': False, '_FillValue': None},
                  'lon': {'zlib': False, '_FillValue': None},
                  'lat_bnds': {'zlib': False, '_FillValue': None, 'chunksizes': (180, 2)},
                  'lon_bnds': {'zlib': False, '_FillValue': None, 'chunksizes': (360, 2)},
                  'time_bnds':{'_FillValue':None},
                  'time':{"units": "days since 2007-01-16 00:00:00",
                        "calendar": "standard",
                        "dtype": "float64",
                        '_FillValue':None},
                  var: {'zlib': True,
                        '_FillValue': np.float32(1.0E20),
                        'chunksizes': (1, 180, 360)}},
    )
