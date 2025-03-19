import os
import time

import cftime as cf
import xarray as xr

from ilamb3_data import (
    download_file,
    fix_lat,
    fix_lon,
    fix_time,
    get_cmip6_variable_info,
)

RENAME = {"GPP": "gpp", "H": "hfss", "LE": "hfls"}

# Download source
remote_source = "https://avdc.gsfc.nasa.gov/pub/data/project/WECANN/WECANN_v1.0.nc"
os.makedirs("_raw", exist_ok=True)
local_source = os.path.join("_raw", os.path.basename(remote_source))
if not os.path.isfile(local_source):
    download_file(remote_source, local_source)
download_stamp = time.strftime(
    "%Y-%m-%d", time.localtime(os.path.getmtime(local_source))
)
generate_stamp = time.strftime("%Y-%m-%d")

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
    info.pop("units")
    data[var].attrs.update(info)

# Encode the dataset
out = xr.Dataset(data_vars=data, coords=coords)

# Fix up the dimensions
out["time"] = fix_time(out)
out["lat"] = fix_lat(out)
out["lon"] = fix_lon(out)
out = out.sortby(["time", "lat", "lon"])
time_mark = f"{out["time"].min().dt.year:d}{out["time"].min().dt.month:02d}"
time_mark += f"-{out["time"].max().dt.year:d}{out["time"].max().dt.month:02d}"
attrs = {
    "title": "Water, Energy, and Carbon with Artificial Neural Networks (WECANN)",
    "version": "1",
    "institution": "Columbia University",
    "source": "Solar Induced Fluorescence (SIF), Air Temperature, Precipitation, Net Radiation, Soil Moisture, and Snow Water Equivalent",
    "history": """
%s: downloaded %s;
%s: converted to ILAMB-ready netCDF"""
    % (
        download_stamp,
        remote_source,
        generate_stamp,
    ),
    "references": """
@ARTICLE{Alemohammad2017,
  author = {Alemohammad, S. H. and Fang, B. and Konings, A. G. and Aires, F. and Green, J. K. and Kolassa, J. and Miralles, D. and Prigent, C. and Gentine, P.},
  title= {Water, Energy, and Carbon with Artificial Neural Networks (WECANN): a statistically based estimate of global surface turbulent fluxes and gross primary productivity using solar-induced fluorescence},
  journal = {Biogeosciences},
  volume = {14},
  year = {2017},
  number = {18},
  page = {4101--4124},
  doi = {https://doi.org/10.5194/bg-14-4101-2017}
}""",
    "activity_id": "obs4MIPs",
    "frequency": "mon",
    "grid_label": "gn",
    "grid": "1x1 deg",
    "institution_id": "Columbia",
    "nominal_resolution": "1x1 deg",
    "product": "observations",
    "realm": "land",
    "source_id": "WECANN",
    "source_type": "observations",
    "source_version_number": "v1",
    "variant_label": "r1i1p1f1",
}


# Write out files
for var, da in out.items():
    dsv = da.to_dataset()
    dsv.attrs = attrs | {
        "variable_id": var,
        "cf_standard_name": dsv[var].attrs.pop("cf_standard_name"),
    }
    dsv.to_netcdf(
        "{variable_id}_{frequency}_{source_id}_{institution_id}_{grid_label}_{time_mark}.nc".format(
            **dsv.attrs, time_mark=time_mark
        ),
        encoding={var: {"zlib": True}},
    )
