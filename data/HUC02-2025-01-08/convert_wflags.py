from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ilamb3_data import (
    download_from_arcgis_rest,
    set_coord_bounds,
    set_lat_attrs,
    set_lon_attrs,
    set_var_attrs,
)

# Download USGS national Watershed Boundary Dataset HUC02 feature layer
remote_source = (
    "https://txgeo.usgs.gov/arcgis/rest/services/NationalBaseData/WBD/MapServer/0"
)
local_source = Path("_raw/WBD_HUC02.geojson")

local_source = download_from_arcgis_rest(
    remote_source,
    local_source,
    where=[f"HUC2='{region:02d}'" for region in range(1, 23)],
)

# Read the GeoJSON as a GeoDataFrame
gdf = gpd.read_file(local_source).to_crs("EPSG:4326")
gdf = gdf.reset_index(drop=True)

# Define a global grid with 0.25 degree resolution and bounds (-180, -90, 180, 90)
resolution = 0.25
west, south, east, north = (-180.0, -90.0, 180.0, 90.0)
width = round((east - west) / resolution)
height = round((north - south) / resolution)
grid_transform = from_bounds(west, south, east, north, width, height)

# Create lat/lon coordinates at the center of each grid cell
lon = west + (np.arange(width) + 0.5) * resolution
lat = north - (np.arange(height) + 0.5) * resolution

# Rasterize the polygons
ids = rasterize(
    zip(gdf.geometry, gdf.index, strict=True),
    out_shape=(height, width),
    transform=grid_transform,
    fill=-1,
    dtype=np.int16,
    all_touched=True,
)

# Store the raster as a categorical coordinate with CF flag metadata
ds = xr.Dataset(
    coords={
        "lat": lat,
        "lon": lon,
        "region": (("lat", "lon"), ids),
    },
)

# Set CF-compliant attributes
ds = set_coord_bounds(ds, "lat")
ds = set_coord_bounds(ds, "lon")
ds = set_lat_attrs(ds)
ds = set_lon_attrs(ds)
ds = set_var_attrs(
    ds,
    "region",
    units="",
    standard_name="huc2_regions",
    long_name="USGS Watershed Boundary Dataset Hydrologic Unit Code (HUC) Level 2 Regions",
    flag_values=np.arange(len(gdf), dtype=np.int8),
    flag_meanings=gdf["HUC2"].to_list(),
    extra_attrs={
        "flag_descriptions": " ".join(name.replace(" ", "_") for name in gdf["NAME"])
    },
    target_dtype=np.dtype("int8"),
    overwrite=True,
)

output = Path("HUC2_flags.nc")
ds.to_netcdf(output)
