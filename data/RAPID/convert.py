from pathlib import Path

import cftime as cf
import xarray as xr

from ilamb3_data import add_time_bounds_monthly, fix_time, gen_utc_timestamp

# Download source, cannot be downloaded automatically
remote_source = "https://rapid.ac.uk/data/data-download"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / "moc_transports.nc"
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()

# Load the dataset for adjustments
ds = xr.open_dataset(local_source).load()
ds = ds.assign_coords({"YYYYMM": ds["time"].dt.year * 100 + ds["time"].dt.month})
ds = ds.groupby("YYYYMM").mean().rename({"YYYYMM": "time"})
ds["time"] = [
    cf.DatetimeNoLeap(int(ym / 100), (ym - int(ym / 100) * 100), 15)
    for ym in ds["time"]
]
ds = add_time_bounds_monthly(ds)
ds = ds.assign_coords({"time_bounds": ds["time_bounds"]})
ds["time"] = fix_time(ds)
ds["time"].attrs["bounds"] = "time_bounds"
ds = ds.drop_vars([v for v in ds.data_vars if v != "moc_mar_hc10"]).rename_vars(
    {"moc_mar_hc10": "amoc"}
)
ds["amoc"].attrs = {
    "standard_name": "overturning transport",
    "long_name": "Atlantic meridonal overturning transport at 26 degrees latitude",
    "units": "Sv",
}

# Fix up the dimensions
time_range = f"{ds["time"].min().dt.year:d}{ds["time"].min().dt.month:02d}"
time_range += f"-{ds["time"].max().dt.year:d}{ds["time"].max().dt.month:02d}"

# Populate attributes
ds.attrs = {
    "activity_id": "obs4MIPs",
    "contact": "Ben Moat (ben.moat@noc.ac.uk)",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_data": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "2.5",
    "doi": "10.5285/223b34a32dc5c945e0637086abc0f274",
    "frequency": "mon",
    "grid": "N/A",
    "grid_label": "NA",
    "history": f"""
{download_stamp}: downloaded {remote_source};
{generate_stamp}: converted to obs4MIP format""",
    "institution": "National Oceanography Centre,UK",
    "institution_id": "NOC",
    "license": "Freely available",
    "nominal_resolution": "N/A",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/RAPID/convert.py",
    "product": "derived",
    "realm": "ocean",
    "region": "north_atlantic_ocean",
    "references": "Moat B.I.; Smeed D.A.; Rayner D.; Johns W.E.; Smith, R.; Volkov, D.; Elipot S.; Petit T.; Kajtar J.; Baringer M. O.; and Collins, J. (2024). Atlantic meridional overturning circulation observed by the RAPID-MOCHA-WBTS (RAPID-Meridional Overturning Circulation and Heatflux Array-Western Boundary Time Series) array at 26N from 2004 to 2023 (v2023.1), British Oceanographic Data Centre - Natural Environment Research Council, UK. doi: 10.5285/223b34a3-2dc5-c945-e063-7086abc0f274",
    "source": "The RAPID-Meridional Overturning Circulation and Heatflux Array-Western Boundary Time Series at 26N that started in April 2004.",
    "source_id": "RAPID",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": remote_source,
    "source_type": "",
    "source_version_number": "2023.1",
    "title": "RAPID MOC timeseries",
    "variable_id": "amoc",
    "variant_label": "BE",
}

# Write out files
ds.to_netcdf(
    "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
        **ds.attrs, time_mark=time_range
    ),
)
