from pathlib import Path

import cf_xarray  # noqa
import xarray as xr
import numpy as np
import os

from ilamb3_data import (
    add_time_bounds_monthly,
	DH_ODS_compliance
    download_file,
    fix_lat,
    fix_lon,
    #fix_time,
    gen_utc_timestamp,
    set_ods_global_attributes,
    gen_trackingid,
    set_time_attrs,
    set_ods_var_attrs,
    set_ods_coords,
    set_ods_calendar,
    add_time_bounds,
    download_from_html,
    gen_utc_timestamp,
    set_lat_attrs,
    set_lon_attrs,
    set_time_attrs)
from datetime import datetime

today = datetime.now().strftime("%Y%m%d")


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
generate_trackingid = gen_trackingid()

# Load the dataset for adjustments
out = xr.open_mfdataset(local_sources).load()
out['mrrostdev'] = out['mrro_sd']
out = out.drop_vars(['mrro_sd'])
out["mrro"].attrs["ancillary_variables"] = "mrrostdev"
out["mrrostdev"].attrs["long_name"] = "runoff_flux standard_deviation"
out["mrrostdev"].attrs["stadard_name"] = "runoff_flux standard_deviation"

# Fix up the dimensions
#out["time"] = fix_time(out)
#out = set_time_attrs(out)
out = set_lat_attrs(out)
out = set_lon_attrs(out)
out = out.sortby(["time", "lat", "lon"])
out = out.cf.add_bounds(["lat", "lon"])
out = out.cf.add_bounds(["time"])
out = set_time_attrs(out)
#out = add_time_bounds_monthly(out)
time_range = f"{out["time"].min().dt.year:d}{out["time"].min().dt.month:02d}"
time_range += f"-{out["time"].max().dt.year:d}{out["time"].max().dt.month:02d}"

# Populate attributes
attrs = {
    "activity_id": "obs4MIPs",
    "contact": "Sanaa Hobeichi (s.hobeichi@student.unsw.edu.au)",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_date": generate_stamp,
    "comment":"Not yet obs4MIPs compliant: 'version' attribute is temporary; Cell measure variable areacella referred to by variable is not present in dataset or external variables",
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
    "institution": "ARC Centre of Excellence for Climate System Science, NSW, Australia,",
    "institution_id": "ARCCSS",
    "license": "Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
    "has_auxdata":'True',
    "aux_variable_id":'mrrosd',
    "nominal_resolution": "50 km",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/LORA/convert.py",
    "product": "observations",
    "realm": "land",
    "references": "Hobeichi, Sanaa, 2018: Linear Optimal Runoff Aggregate v1.0. NCI National Research Data Collection, doi:10.25914/5b612e993d8ea",
    "region": "global_land",
    "source": "LORA 1.1 (2018): Linear Optimal Runoff Aggregate",
    "source_id": "LORA-1-1",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": "https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/LORA/v1-0/catalog.html",
    "source_type": "gridded_insitu",
    "source_label": 'LORA',
    "variant_info":"CMORized product prepared by ILAMB and CMIP IPO",
    "source_version_number": "1.1",
    "title": "Linear Optimal Runoff Aggregate v1.0",
    "variant_label": "REF",
    "version":today,
    "tracking_id": generate_trackingid,
}
del out.attrs['time_coverage_start']
del out.attrs['time_coverage_end']
# Write out files
out.attrs = attrs | {"variable_id": "mrro"}
out = set_ods_global_attributes(out, **out.attrs)
out = set_ods_var_attrs(out, "mrro")
out = set_ods_coords(out)
print(out)

out_path = (
    "/home/users/dhegedus/CMIPplacement/beta/{activity_id}/{institution_id}/{source_id}/{frequency}/{variable_id}/{grid_label}/"
    + today
    + "/{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc"
).format(**out.attrs, time_mark=time_range)

# Ensure the output directory exists
os.makedirs(os.path.dirname(out_path), exist_ok=True)


out.to_netcdf(out_path,
    encoding={'lat': {'zlib': False, '_FillValue': None},
                  'lon': {'zlib': False, '_FillValue': None},
                  'lat_bnds': {'zlib': False, '_FillValue': None, 'chunksizes': (180, 2)},
                  'time_bnds': {'zlib': False, '_FillValue': None},
                  'lon_bnds': {'zlib': False, '_FillValue': None, 'chunksizes': (360, 2)},
                  "time": {"units": "days since 1979-01-01 00:00:00",
                        "calendar": "gregorian",
                        "dtype": "float64",
                        "_FillValue":None},
                  'mrro': {'zlib': True,
                        '_FillValue': np.float32(1.0E20),
                        'chunksizes': (1, 180, 360)},
                  'mrrostdev': {'zlib': True,
                        '_FillValue': np.float32(1.0E20),
                        'chunksizes': (1, 180, 360)}},
)
