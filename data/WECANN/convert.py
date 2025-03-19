from pathlib import Path

import cftime as cf
import xarray as xr

from ilamb3_data import (
    download_file,
    fix_lat,
    fix_lon,
    fix_time,
    gen_utc_timestamp,
    get_cmip6_variable_info,
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
out["time"] = fix_time(out)
out["lat"] = fix_lat(out)
out["lon"] = fix_lon(out)
out = out.sortby(["time", "lat", "lon"])
time_range = f"{out["time"].min().dt.year:d}{out["time"].min().dt.month:02d}"
time_range += f"-{out["time"].max().dt.year:d}{out["time"].max().dt.month:02d}"

# Populate attributes
attrs = {
    "activity_id": "obs4MIPs",
    "contact": "S. Hamed Alemohammad (sha2128@columbia.edu) and Pierre Gentine (pg2328@columbia.edu)",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_data": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "2.5",
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
    "institution_id": "Columbia",
    "license": "No Commercial Use",
    "nominal_resolution": "1x1 degree",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/WECANN/convert.py",
    "product": "observations",
    "realm": "land",
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
    "region": "global_land",
    "source": "Solar Induced Fluorescence (SIF), Air Temperature, Precipitation, Net Radiation, Soil Moisture, and Snow Water Equivalent",
    "source_id": "WECANN",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": remote_source,
    "source_type": "statistical-estimates",
    "source_version_number": "1",
    "title": "Water, Energy, and Carbon with Artificial Neural Networks (WECANN)",
    "variant_label": "ILAMB",
}


# Write out files
for var, da in out.items():
    dsv = da.to_dataset()
    dsv.attrs = attrs | {"variable_id": var}
    dsv.to_netcdf(
        "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
            **dsv.attrs, time_mark=time_range
        ),
        encoding={var: {"zlib": True}},
    )
