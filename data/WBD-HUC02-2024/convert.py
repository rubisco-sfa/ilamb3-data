from datetime import datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import xarray as xr
from rasterio.features import rasterize
from rasterio.transform import from_bounds

from ilamb3_data import (
    create_output_filename,
    download_from_arcgis_rest,
    gen_trackingid,
    set_coord_bounds,
    set_lat_attrs,
    set_lon_attrs,
    set_regions_global_attrs,
    set_var_attrs,
)

########################################################################################
# Pre-define some information for the different HUC2 aggregation schemes

VAR_IDS = ["huc02sel18", "huc02agg04", "huc02agg01"]

AGGREGATE_GROUPS = {
    "huc02agg04": [
        {
            "value": 0,
            "meaning": "01",
            "description": "East_Contiguous_US_HUCs",
            "hucs": ["01", "02", "03", "05", "06"],
        },
        {
            "value": 1,
            "meaning": "02",
            "description": "North_Contiguous_US_HUCs",
            "hucs": ["04", "07", "09", "10"],
        },
        {
            "value": 2,
            "meaning": "03",
            "description": "South_Contiguous_US_HUCs",
            "hucs": ["08", "11", "12", "13"],
        },
        {
            "value": 3,
            "meaning": "04",
            "description": "West_Contiguous_US_HUCs",
            "hucs": ["14", "15", "16", "17", "18"],
        },
    ],
    "huc02agg01": [
        {
            "value": 0,
            "meaning": "01",
            "description": "Contiguous_US_HUCs",
            "hucs": [f"{huc:02d}" for huc in range(1, 19)],
        },
    ],
}


########################################################################################
# Create function that aggregates source HUC2 raster values into larger regions


def aggregate_regions(
    source_ids: xr.DataArray,
    huc_to_source_value: dict[str, int],
    groups: list[dict],
) -> xr.DataArray:
    """Remap source HUC2 raster values to a smaller set of aggregate values."""
    aggregate_ids = xr.full_like(source_ids, fill_value=-1, dtype=np.int8)
    for group in groups:
        missing_hucs = set(group["hucs"]) - huc_to_source_value.keys()
        if missing_hucs:
            raise ValueError(
                f"Aggregate {group['meaning']} contains unavailable HUC2 codes: "
                f"{sorted(missing_hucs)}"
            )
        source_values = [huc_to_source_value[huc] for huc in group["hucs"]]
        aggregate_ids = xr.where(
            source_ids.isin(source_values),
            np.int8(group["value"]),
            aggregate_ids,
        )
    return aggregate_ids


########################################################################################
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

# Get dates, etc.
today = datetime.today().strftime("%Y-%m-%d")
download_date = datetime.fromtimestamp(local_source.stat().st_mtime).strftime(
    "%Y-%m-%d"
)

########################################################################################
# Read and subselect the data

gdf = gpd.read_file(local_source).to_crs("EPSG:4326")
gdf = gdf.reset_index(drop=True)

# Drop non-contiguous US regions
gdf = gdf[
    ~gdf["NAME"].str.contains("Alaska|Hawaii|Caribbean|South Pacific")
].reset_index(drop=True)
gdf["source_value"] = np.arange(len(gdf), dtype=np.int8)
huc_to_source_value = dict(zip(gdf["HUC2"], gdf["source_value"], strict=True))

########################################################################################
# Create a global grid with 0.25 degree resolution and bounds (-180, -90, 180, 90)

# Define the resolution and bounds of the global grid
resolution = 0.25
west, south, east, north = (-180.0, -90.0, 180.0, 90.0)
width = round((east - west) / resolution)
height = round((north - south) / resolution)
grid_transform = from_bounds(west, south, east, north, width, height)

# Create lat/lon coordinates at the center of each grid cell
lon = west + (np.arange(width) + 0.5) * resolution
lat = north - (np.arange(height) + 0.5) * resolution

########################################################################################
# Turn the vector polygons into a raster grid using the grid defined above

# Rasterize the polygons
ids = rasterize(
    zip(gdf.geometry, gdf["source_value"], strict=True),
    out_shape=(height, width),
    transform=grid_transform,
    fill=-1,
    dtype=np.int8,
)

# Build the xarray DataArray from the rasterized HUC2 IDs
source_ids = xr.DataArray(
    ids,
    dims=("lat", "lon"),
    coords={"lat": lat, "lon": lon},
)

# Crop to valid data with one grid cell of padding.
valid = source_ids != -1

lat_indices = np.flatnonzero(valid.any(dim="lon").values)
lon_indices = np.flatnonzero(valid.any(dim="lat").values)

lat_start = max(0, lat_indices[0] - 1)
lat_stop = min(source_ids.sizes["lat"], lat_indices[-1] + 2)
lon_start = max(0, lon_indices[0] - 1)
lon_stop = min(source_ids.sizes["lon"], lon_indices[-1] + 2)

source_ids = source_ids.isel(
    lat=slice(lat_start, lat_stop),
    lon=slice(lon_start, lon_stop),
)

########################################################################################
# Aggregate the rasterized HUC2 regions into larger units & set attributes

# define variable attributes
attrs = {
    "huc02sel18": {
        "standard_name": "huc02_conus_selection",
        "long_name": "USGS Watershed Boundary Dataset Hydrologic Unit Code Level 2 Regions with 18 selected Contiguous USA Units",
    },
    "huc02agg04": {
        "standard_name": "huc02_conus_quadrant_aggregate",
        "long_name": "USGS Watershed Boundary Dataset Hydrologic Unit Code Level 2 Regions with 4 aggregated Contiguous USA Units",
    },
    "huc02agg01": {
        "standard_name": "huc02_conus_aggregate",
        "long_name": "USGS Watershed Boundary Dataset Hydrologic Unit Code Level 2 Regions with 1 aggregated Contiguous USA Unit",
    },
}

# loop through each variable, aggregate the regions, and set attributes
for VAR_ID in VAR_IDS:
    if VAR_ID == "huc02sel18":
        groups = [
            {
                "value": int(row.source_value),  # type: ignore
                "meaning": row.HUC2,
                "description": row.NAME.replace(" ", "_"),  # type: ignore
                "hucs": [row.HUC2],
            }
            for row in gdf.itertuples()
        ]
        region_ids = source_ids.copy()
        processing_description = (
            "retained the 18 contiguous US HUC2 regions as separate units"
        )
    else:
        groups = AGGREGATE_GROUPS[VAR_ID]
        region_ids = aggregate_regions(source_ids, huc_to_source_value, groups)
        processing_description = (
            "merged the 18 contiguous US HUC2 regions into aggregate unit(s): "
            + ", ".join(
                f"{group['description']}: [{', '.join(group['hucs'])}]"
                for group in groups
            )
        )

    # Store the selected aggregation as a categorical data variable.
    ds = region_ids.to_dataset(name=VAR_ID)
    ds = set_coord_bounds(ds, "lat")
    ds = set_coord_bounds(ds, "lon")
    ds = set_lat_attrs(ds)
    ds = set_lon_attrs(ds)

    ds = set_var_attrs(
        ds,
        VAR_ID,
        units="",
        standard_name=attrs[VAR_ID]["standard_name"],
        long_name=attrs[VAR_ID]["long_name"],
        flag_values=np.asarray([group["value"] for group in groups], dtype=np.int8),
        flag_meanings=[group["meaning"] for group in groups],
        extra_attrs={
            "flag_descriptions": " ".join(group["description"] for group in groups),
        },
        target_dtype=np.dtype("int8"),
        nodata_value=-1,
        compression={"zlib": True, "complevel": 4, "shuffle": True},
        overwrite=True,
    )

    # Set global attributes
    out_ds = set_regions_global_attrs(
        ds,
        activity_id="ILAMB",
        comment=(f"Categorical map containing {len(groups)} region unit(s)."),
        contact="Morgan Steckler (stecklermr@ornl.gov)",
        conventions="CF-1.12",
        creation_date=today,
        dataset_contributor="Morgan Steckler",
        frequency="fx",
        grid="0.25x0.25 degree latitude x longitude",
        grid_label="gn",
        history=(
            f"Downloaded ESRI version 20240105 on {download_date}; selected "
            f"the 18 contiguous US HUC2 regions, rasterized them to a "
            f"0.25x0.25 degree latitude x longitude grid, and "
            f"{processing_description} on {today}"
        ),
        institution="United States Geological Survey, Reston, Virginia, USA",
        institution_id="USGS",
        license="Public Domain",
        nominal_resolution="0.25 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/WBD-HUC02-2024/convert.py",
        references="Jones, K.A., Niknami, L.S., Buto, S.G., and Decker, D., 2022, Federal standards and procedures for the national Watershed Boundary Dataset (WBD) (5 ed.): U.S. Geological Survey Techniques and Methods 11-A3, 54 p., https://doi.org/10.3133/tm11A3.",
        region="conus",
        source="Watershed Boundary Dataset (WBD) Hydrologic Unit Code 2 (HUC02) version 2024",
        source_data_retrieval_date=download_date,
        source_data_url=remote_source,
        source_id="WBD-HUC02-2024",
        source_version_number="2024",
        title=attrs[VAR_ID]["long_name"],
        tracking_id=gen_trackingid(),
        variable_id=VAR_ID,
        variant_label="ILAMB",
        version=f"v{datetime.today().strftime('%Y%m%d')}",
    )

    # Prep for export
    out_path = create_output_filename(out_ds.attrs)
    out_ds.to_netcdf(out_path, format="NETCDF4")
