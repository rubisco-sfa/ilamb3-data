import re
from datetime import datetime
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import xarray as xr

from ilamb3_data import (
    create_output_filename,
    download_from_figshare,
    gen_trackingid,
    set_coord_bounds,
    set_lat_attrs,
    set_lon_attrs,
    set_regions_global_attrs,
    set_var_attrs,
)

FIGSHARE_ARTICLE_ID = "6396959"
VAR_IDS = ["koppen", "koppenagg5"]
AGGREGATE_GROUPS = {
    "koppenagg5": [
        {
            "value": 0,
            "meaning": "tropical",
            "description": "Group A: Tropical Climates",
            "classes": ["af", "am", "aw"],
        },
        {
            "value": 1,
            "meaning": "arid",
            "description": "Group B: Desert and semi-arid climates",
            "classes": ["bwh", "bwk", "bsh", "bsk"],
        },
        {
            "value": 2,
            "meaning": "temperate",
            "description": "Group C: Temperate climates",
            "classes": ["csa", "csb", "csc", "cwa", "cwb", "cwc", "cfa", "cfb", "cfc"],
        },
        {
            "value": 3,
            "meaning": "cold",
            "description": "Group D: Continental climates",
            "classes": [
                "dsa",
                "dsb",
                "dsc",
                "dsd",
                "dwa",
                "dwb",
                "dwc",
                "dwd",
                "dfa",
                "dfb",
                "dfc",
                "dfd",
            ],
        },
        {
            "value": 4,
            "meaning": "polar",
            "description": "Group E: Polar and alpine climates",
            "classes": ["et", "ef"],
        },
    ],
}


def parse_legend(legend_filename: Path) -> tuple[list[int], list[str], list[str]]:
    values = []
    labels = []
    names = []
    lines = open(legend_filename, encoding="ISO 8859-1").readlines()
    for line in lines:
        m = re.match(r"\s*(\d+):\s+(\w*)\s+(.*)\s\[.*", line)
        if m:
            values.append(int(m.group(1)))
            lbl = m.group(2)
            labels.append(lbl.lower())
            names.append(f"{lbl}: {m.group(3).strip()}")
    return values, labels, names


########################################################################################
# Create function that aggregates source HUC2 raster values into larger regions


def aggregate_regions(
    source_ids: xr.DataArray,
    class_to_source_value: dict[str, int],
    groups: list[dict],
) -> xr.DataArray:
    """Remap source HUC2 raster values to a smaller set of aggregate values."""
    aggregate_ids = xr.full_like(source_ids, fill_value=-1, dtype=np.int8)
    for group in groups:
        missing_classes = set(group["classes"]) - class_to_source_value.keys()
        if missing_classes:
            raise ValueError(
                f"Aggregate {group['meaning']} contains unavailable class codes: "
                f"{sorted(missing_classes)}"
            )
        source_values = [class_to_source_value[huc] for huc in group["classes"]]
        aggregate_ids = xr.where(
            source_ids.isin(source_values),
            np.int8(group["value"]),
            aggregate_ids,
        )
    return aggregate_ids


# "https://www.gloh2o.org/koppen/"

#


########################################################################################
# Download and unzip figshare data

raw_files = download_from_figshare(FIGSHARE_ARTICLE_ID)
for raw in raw_files:
    if raw.suffix != ".zip":
        continue
    with ZipFile(raw) as zf:
        zf.extractall(raw.parent, members=["Beck_KG_V1_present_0p5.tif", "legend.txt"])

remote_source = "https://doi.org/10.6084/m9.figshare.6396959"
local_source = Path("_raw/Beck_KG_V1_present_0p5.tif")
legend_source = Path("_raw/legend.txt")

# Get dates, etc.
today = datetime.today().strftime("%Y-%m-%d")
download_date = datetime.fromtimestamp(local_source.stat().st_mtime).strftime(
    "%Y-%m-%d"
)


ds = (
    xr.open_dataset(local_source)
    .sel({"band": 1})
    .rename({"x": "lon", "y": "lat", "band_data": "ids"})
    .drop_vars(["band", "spatial_ref"])
)
source_ids = ds["ids"].astype(np.int8)
source_ids = xr.where(source_ids > 0, source_ids, -1)
region_values, region_labels, region_names = parse_legend(legend_source)
class_to_source_value = dict(zip(region_labels, region_values, strict=True))

########################################################################################
# Aggregate the Koppen regions into larger units & set attributes

# define variable attributes
attrs = {
    "koppen": {
        "standard_name": "koppen_classes",
        "long_name": "Koppen Climate Classes",
    },
    "koppenagg5": {
        "standard_name": "koppen_classes_level1_aggregate",
        "long_name": "Koppen Climate Classes Aggregated by Level 1 Label",
    },
}

# loop through each variable, aggregate the regions, and set attributes
for VAR_ID in VAR_IDS:
    if VAR_ID == "koppen":
        groups = [
            {
                "value": value,
                "meaning": label,
                "description": name.replace(" ", "_"),
                "classes": [value],
            }
            for value, label, name in zip(region_values, region_labels, region_names)
        ]
        region_ids = source_ids.copy()
        processing_description = "retained the 30 Koppen classes"
    else:
        groups = AGGREGATE_GROUPS[VAR_ID]
        region_ids = aggregate_regions(source_ids, class_to_source_value, groups)
        processing_description = (
            "merged the 30 Koppen classes into aggregate unit(s): "
            + ", ".join(
                f"{group['description']}: [{', '.join(group['classes'])}]"
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
        contact="Nathan Collier (nathaniel.collier@gmail.com)",
        conventions="CF-1.12",
        creation_date=today,
        dataset_contributor="Nathan Collier",
        frequency="fx",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        history=(
            f"Downloaded Figshare files on {download_date} and {processing_description} on {today}"
        ),
        institution="GloH2O: Environmental Services",
        institution_id="GloH2O",
        license="CC BY 4.0",
        nominal_resolution="0.5 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/GloH2O/convert.py",
        references=" Beck, H.E., T.R. McVicar, N. Vergopolan, A. Berg, N.J. Lutsko, A. Dufour, Z. Zeng, X. Jiang, A.I.J.M. van Dijk, D.G. Miralles. High-resolution (1 km) Köppen-Geiger maps for 1901–2099 based on constrained CMIP6 projections, Scientific Data 10, 724, doi:10.1038/s41597-023–02549‑6 (2023).",
        region="glb",
        source="High-resolution, observation-based climatologies",
        source_data_retrieval_date=download_date,
        source_data_url=remote_source,
        source_id="Koppen-v1",
        source_version_number="v1",
        title=attrs[VAR_ID]["long_name"],
        tracking_id=gen_trackingid(),
        variable_id=VAR_ID,
        variant_label="ILAMB",
        version=f"v{datetime.today().strftime('%Y%m%d')}",
    )

    # Prep for export
    out_path = create_output_filename(out_ds.attrs)
    out_ds.to_netcdf(out_path, format="NETCDF4")
