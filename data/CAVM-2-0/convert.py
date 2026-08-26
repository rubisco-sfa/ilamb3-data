import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rioxarray as rxr
from matplotlib.colors import BoundaryNorm, ListedColormap
from rasterio.enums import Resampling

import ilamb3_data as ild

VAR = "landCoverCat"

RAW_PATH = Path("_raw")
if not RAW_PATH.is_dir():
    RAW_PATH.mkdir()


# --------------------------------------------------------------------------------------
# 1) Download
# --------------------------------------------------------------------------------------


# Download CAVM 2.0 data from Mendeley Data HTML and unzip
remote_source = "https://data.mendeley.com/public-files/datasets/c4xj5rv6kv/files/5223c414-234a-498c-ae08-3100cb38510f/file_downloaded"
local_source = RAW_PATH / "Raster CAVM GIS data.zip"
local_source = ild.download.from_html(remote_source, local_source)
download_stamp = ild.output.utc_timestamp(local_source.stat().st_mtime)

# --------------------------------------------------------------------------------------
# 2) Read Data
# --------------------------------------------------------------------------------------


# Read TIF and legend CSV
ds = rxr.open_rasterio(local_source / "raster_cavm_v1.tif", band_as_variable=True)

# --------------------------------------------------------------------------------------
# 3) Manual cleaning/formatting
# --------------------------------------------------------------------------------------


# Helper function for later
def _clean_flag_description(s: str) -> str:
    s = s.strip().lower()
    s = s.replace(",", "")
    s = s.replace("/", "-")
    s = s.replace(" ", "_")
    return s


# Reproject to lat/lon and rename dimensions/variables
ds = ds.rio.reproject("EPSG:4326", resampling=Resampling.nearest)
ds = ds.rename({"band_1": VAR, "x": "lon", "y": "lat"})

# Clip CAVM extent to non-null bounding box
valid = ds[VAR] != 127
ds = ds.sel(
    lat=ds.lat[valid.any(dim="lon")],
    lon=ds.lon[valid.any(dim="lat")],
)

# Read CAVM legend and use it to set flag values/meanings and descriptions
legend = local_source / "Raster CAVM legend.csv"
df = pd.read_csv(
    legend, skiprows=[0, 1], na_filter=False, dtype={"Raster code": np.int8}
)
df = df.sort_values("Raster code").reset_index(drop=True)
# Fill in some missing short descriptions manually
desc_fill = {
    "FW": "Fresh water",
    "SW": "Salt water",
    "GL": "Glacier",
    "NA": "Non-Arctic",
}
mask = df["Short Description"].str.strip() == ""
df.loc[mask, "Short Description"] = df.loc[mask, "Vegetation Unit"].map(desc_fill)

# Palette for CAVM visualization
palette = {
    1: "#d7d7b3",
    2: "#a8a802",
    3: "#a68282",
    4: "#8282a0",
    5: "#cdcd66",
    21: "#ffebaf",
    22: "#ffd37f",
    23: "#e6e600",
    24: "#ffff00",
    31: "#dfb0b0",
    32: "#db949e",
    33: "#97e602",
    34: "#38a802",
    41: "#9eedbd",
    42: "#73ffdf",
    43: "#04e6a9",
    91: "#0070ff",
    92: "#e0f2ff",
    93: "#ffffff",
    99: "#cccccc",
}

# Merge palette into df and filter to codes present in the raster so that
# flag_values, flag_meanings, flag_descriptions, and flag_colors all align.
df["color"] = df["Raster code"].map(palette)
raster_codes = set(
    np.unique(ds[VAR].values[(~np.isnan(ds[VAR].values)) & (ds[VAR].values != 127)])
)
df = df[df["Raster code"].isin(raster_codes)].copy()


# --------------------------------------------------------------------------------------
# 4) Automatic cleaning/formatting using ILAMB utilities
# --------------------------------------------------------------------------------------

# Set lat/lon/var attributes
ds = ild.lat.standardize(ds)
ds = ild.lon.standardize(ds)
ds = ild.bounds.build_from_centers(ds, "lat")
ds = ild.bounds.build_from_centers(ds, "lon")

# Decisions about naming derived from https://raw.githubusercontent.com/PCMDI/mip-cmor-tables/refs/heads/main/MIP_variables.json
# Search for "landCoverFrac" to see how I modeled this
ds = ild.variable.standardize(
    ds,
    VAR,
    units="",
    standard_name="cover_category",
    long_name="Vegetation or Land-Cover Category",
    flag_values=np.asarray(df["Raster code"], dtype=np.int8),
    flag_meanings=df["Vegetation Unit"].to_list(),
    extra_attrs={
        "flag_descriptions": " ".join(
            _clean_flag_description(s) for s in df["Short Description"]
        ),
        "flag_colors": " ".join(df["color"]),
    },
    target_dtype=np.dtype("int8"),
    compression={"zlib": True, "complevel": 4, "shuffle": True},
    overwrite=True,
)

# Final cleanup
ds = ds.drop_vars("spatial_ref")

# --------------------------------------------------------------------------------------
# 5) Set Obs4MIPs-compliant global attributes and write NetCDF
# --------------------------------------------------------------------------------------

# Set global attributes and write NetCDF
generate_stamp = time.strftime("%Y%m%d")
tracking_id = ild.output.new_tracking_id()
ds = ild.global_attrs.set_ods26(
    ds,
    activity_id="ILAMB",
    contact="Martha Raynolds (mkraynolds@alaska.edu)",
    creation_date=generate_stamp,
    dataset_contributor="Morgan Steckler",
    doi="https://doi.org/10.17632/c4xj5rv6kv.2",
    frequency="fx",
    grid="1x1 km Lambert Azimuthal Equal Area reprojected to latitude x longitude",
    grid_label="gr",
    has_aux_unc="FALSE",
    history=f"""
{generate_stamp}: "CMORized" data from Raster CAVM 2.0 (downloaded from Mendeley Data)\n
{generate_stamp}: Reprojected to lat/lon and added metadata using ILAMB utilities
""",
    institution="University of Alaska, Fairbanks, USA",
    institution_id="UAF",
    license="https://creativecommons.org/licenses/by-nc/3.0/deed.en",
    nominal_resolution="1 km",
    processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/CAVM-2-0/convert.py",
    product="derived",
    realm="land",
    references="Raynolds, M.K., et al. (2019). A raster version of the Circumpolar Arctic Vegetation Map (CAVM). Remote Sensing of Environment, 232, 111297. https://doi.org/10.1016/j.rse.2019.111297",
    region="panarctic",
    source="Unsupervised classifications of seventeen geographic/floristic sub-sections of the Arctic, using AVHRR and MODIS data (reflectance and NDVI) and elevation data",
    source_id="CAVM-2-0",
    source_data_retrieval_date=download_stamp,
    source_data_url=str(remote_source),
    source_label="CAVM",
    source_type="satellite_retrieval",
    source_version_number="2.0",
    title="Raster Circumpolar Arctic Vegetation Map",
    tracking_id=tracking_id,
    variable_id=VAR,
    variant_label="ILAMB",
    variant_info="CMORized product prepared by ILAMB",
    version=f"v{generate_stamp}",
)
out_path = ild.output.filename_from_attrs(ds.attrs)
ds.to_netcdf(out_path)

# --------------------------------------------------------------------------------------
# Plotting verification
# --------------------------------------------------------------------------------------

codes = ds[VAR].attrs["flag_values"]
colors = ds[VAR].attrs["flag_colors"].split()
labels = ds[VAR].attrs["flag_meanings"].split()

idx_map = {c: i for i, c in enumerate(codes)}
mapped = np.vectorize(lambda x: idx_map.get(x, np.nan), otypes=[float])(ds[VAR].values)

fig, ax = plt.subplots(figsize=(10, 5))
ax.pcolormesh(
    ds["lon"],
    ds["lat"],
    mapped,
    cmap=ListedColormap(colors),
    norm=BoundaryNorm(np.arange(-0.5, len(codes)), len(codes)),
)
cbar = plt.colorbar(ax.collections[0], ax=ax, ticks=range(len(codes)))
cbar.ax.set_yticklabels(labels)
plt.tight_layout()
plt.savefig(f"{VAR}.png", dpi=300)
plt.close()
