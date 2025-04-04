from pathlib import Path

import cf_xarray  # noqa
import xarray as xr

from ilamb3_data import (
    add_time_bounds_monthly,
    download_from_html,
    gen_utc_timestamp,
    set_lat_attrs,
    set_lon_attrs,
    set_time_attrs,
)

# Download source
remote_sources = [
    f"https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/LORA/v1-0/LORA_v1.0_{year}.nc"
    for year in range(1980, 2013)
]
local_source = Path("_raw")
local_sources = []
for remote_source in remote_sources:
    local_source.mkdir(parents=True, exist_ok=True)
    source = local_source / Path(remote_source).name
    if not source.is_file():
        download_from_html(remote_source, str(source))
    local_sources.append(source)
download_stamp = gen_utc_timestamp(local_sources[0].stat().st_mtime)
generate_stamp = gen_utc_timestamp()

# Load the dataset for adjustments
out = xr.open_mfdataset(local_sources).load()
out["mrro"].attrs["ancillary_variables"] = "mrro_sd"
out["mrro_sd"].attrs["long_name"] = "runoff_flux standard_deviation"
out["mrro_sd"].attrs["stadard_name"] = "runoff_flux standard_deviation"

# Fix up the dimensions
out = set_time_attrs(out)
out = set_lat_attrs(out)
out = set_lon_attrs(out)
out = out.sortby(["time", "lat", "lon"])
out = out.cf.add_bounds(["lat", "lon"])
out = add_time_bounds_monthly(out)
time_range = f"{out['time'].min().dt.year:d}{out['time'].min().dt.month:02d}"
time_range += f"-{out['time'].max().dt.year:d}{out['time'].max().dt.month:02d}"

# Populate attributes
attrs = {
    "activity_id": "obs4MIPs",
    "contact": "Sanaa Hobeichi (s.hobeichi@student.unsw.edu.au)",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_data": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "2.5",
    "doi": "10.25914/5b612e993d8ea",
    "frequency": "mon",
    "grid": "0.5x0.5 degree",
    "grid_label": "gn",
    "history": """
%s: downloaded %s;
%s: converted to obs4MIP format"""
    % (
        download_stamp,
        remote_sources,
        generate_stamp,
    ),
    "institution": "University of New South Wales",
    "institution_id": "UNSW",
    "license": "CC BY-NC-SA 4.0",
    "nominal_resolution": "0.5x0.5 degree",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/LORA/convert.py",
    "product": "observations",
    "realm": "land",
    "references": "Hobeichi, Sanaa, 2018: Linear Optimal Runoff Aggregate v1.0. NCI National Research Data Collection, doi:10.25914/5b612e993d8ea",
    "region": "global_land",
    "source": "total runoff and seven streamflow outputs from tiers 1 and 2 of the eartH2Observe project",
    "source_id": "LORA",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": "https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/LORA/v1-0/catalog.html",
    "source_type": "statistical-estimates",
    "source_version_number": "1",
    "title": "Linear Optimal Runoff Aggregate v1.0",
    "variant_label": "BE",
}

# Write out files
out.attrs = attrs | {"variable_id": "mrro"}
out.to_netcdf(
    "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
        **out.attrs, time_mark=time_range
    ),
    encoding={"mrro": {"zlib": True}},
)
