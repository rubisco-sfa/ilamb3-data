from pathlib import Path

import cf_xarray  # noqa
import cftime as cf
import xarray as xr
import numpy as np

from ilamb3_data import (
    add_time_bounds_monthly,
    download_file,
    fix_lat,
    fix_lon,
    fix_time,
    gen_utc_timestamp,
    get_cmip6_variable_info,
    set_ods_global_attributes,
    gen_trackingid,
    set_ods_var_attrs,
    set_ods_coords,
    set_ods_calendar,
)

RENAME = {"GPP": "gpp", "H": "hfss", "LE": "hfls"}


# Download source
remote_source = "https://avdc.gsfc.nasa.gov/pub/data/project/WECANN/WECANN_v1.0.nc"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / Path(remote_source).name
if not local_source.is_file():
    download_file(remote_source, str(local_source))
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()
generate_trackingid = gen_trackingid()

# Build up a new dataset
ds = xr.open_dataset(local_source)
coords = {}

# Time is encoded as a 2D character array [["2022/01",...]]
coords["time"] = [
    cf.DatetimeNoLeap(int("".join(t[:4])), int("".join(t[-2:])), 15)
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
out = set_ods_calendar(out)
out["time"] = fix_time(out)
out["lat"] = fix_lat(out)
out["lon"] = fix_lon(out)
out = out.sortby(["time", "lat", "lon"])
out = out.cf.add_bounds(["lat", "lon"])
out = add_time_bounds_monthly(out)
time_range = f"{out['time'].min().dt.year:d}{out['time'].min().dt.month:02d}"
time_range += f"-{out['time'].max().dt.year:d}{out['time'].max().dt.month:02d}"
# Populate attributes
attrs = {
    "activity_id": "obs4MIPs",
    "contact": "HAlemohammad@clarku.edu",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_date": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "ODS2.5",
    "doi": "N/A",
    #"external_variables": None,
    "frequency": "mon",
    "grid": "1x1 degree",
    "grid_label": "gn",
    "history": """
%s: downloaded %s;
%s: converted to obs4MIP format"""
    % (
        download_stamp,
        remote_source,
        generate_stamp,
    ),
    "institution": "Columbia University",
    "institution_id": "ColumbiaU",
    "license": "Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
    "nominal_resolution": "1x1 degree",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/WECANN/convert.py",
    "product": "derived",
    "realm": "land",
    "references": "Alemohammad, S. H., Fang, B., Konings, A. G., Aires, F., Green, J. K., Kolassa, J., Miralles, D., Prigent, C., and Gentine, P.: Water, Energy, and Carbon with Artificial Neural Networks (WECANN): a statistically based estimate of global surface turbulent fluxes and gross primary productivity using solar-induced fluorescence, Biogeosciences, 14, 4101–4124, https://doi.org/10.5194/bg-14-4101-2017, 2017.",
    "region": "global_land",
    "source": "WECANN 1.0 (2018): Water, Energy, and Carbon with Artificial Neural Networks",
    "source_id": "WECANN-1-0",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": remote_source,
    "source_label":"WECANN",
    "source_type": "satellite_retrieval",
    "source_version_number": "1",
    "variant_label":"ILAMB_REF",
    "variant_info":"CMORized product prepared by ILAMB and CMIP IPO",
    "title": "Water, Energy, and Carbon with Artificial Neural Networks (WECANN)",
    "tracking_id": generate_trackingid,
    "variant_label": "ILAMB-REF",
}

# Write out files
for var, da in out.items():
    dsv = da.to_dataset()
    dsv = out.drop_vars(set(out) - set([var]))
    dsv.attrs = attrs | {"variable_id": var}
    dsv = set_ods_global_attributes(dsv, **dsv.attrs)
    dsv = set_ods_var_attrs(dsv, var)
    dsv[var] = dsv[var].astype(np.float32)
    dsv = set_ods_coords(dsv)
    dsv.to_netcdf(
        "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
            **dsv.attrs, time_mark=time_range
        ),
        encoding={'lat': {'zlib': False, '_FillValue': None},
                  'lon': {'zlib': False, '_FillValue': None},
                  'lat_bnds': {'zlib': False, '_FillValue': None, 'chunksizes': (180, 2)},
                  'lon_bnds': {'zlib': False, '_FillValue': None, 'chunksizes': (360, 2)},
                  var: {'zlib': True,
                        '_FillValue': np.float32(1.0E20),
                        'chunksizes': (1, 180, 360)}},
    )
