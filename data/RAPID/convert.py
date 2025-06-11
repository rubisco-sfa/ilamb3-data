from pathlib import Path
from datetime import datetime

import cftime as cf
import xarray as xr
import os

from ilamb3_data import (
    add_time_bounds2, 
    fix_time,
    fix_lat,
    gen_utc_timestamp,
    set_ods_global_attributes,
    gen_trackingid,
    set_ods_var_attrs,
    set_ods_coords,
    )

today = datetime.now().strftime("%Y%m%d")

# Download source, cannot be downloaded automatically
remote_source = "https://rapid.ac.uk/data/data-download"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / "moc_transports.nc"
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()

generate_trackingid = gen_trackingid()

# Load the dataset for adjustments
ds = xr.open_dataset(local_source).load()

# Adding the 'basin' coordinate
basin = xr.DataArray(["atlantic_ocean"], dims=["basin"], name="basin")

# Adding the 'latitude' coordinate (scalar for 26°N)
latitude = xr.DataArray([26.0], dims=["lat"], name="lat", attrs={"units": "degrees_north"})

# Adding a scalar olevel to indicate the vertically integrated nature of the data
olevel = xr.DataArray([0.0], dims=["olevel"], name="olevel", attrs={"units": "m", "long_name": "integrated depth"})

# Assigning the coordinates to the dataset
ds = ds.assign_coords({'basin':basin, 'lat':latitude, 'olevel':olevel,"YYYYMM": ds["time"].dt.year * 100 + ds["time"].dt.month})
ds = ds.groupby("YYYYMM").mean().rename({"YYYYMM": "time"})
ds["time"] = [
    cf.DatetimeGregorian(int(ym / 100), (ym - int(ym / 100) * 100), 15)
    for ym in ds["time"]
]
ds = add_time_bounds2(ds)
ds = ds.assign_coords({"time_bnds": ds["time_bnds"]})
ds["time"] = fix_time(ds)
ds['time'].attrs['bounds'] = 'time_bnds'

ds = ds.drop_vars([v for v in ds.data_vars if v != "moc_mar_hc10"]).rename_vars(
    {"moc_mar_hc10": "msftmz"}
)


ds["msftmz"].attrs = {
    "standard_name": "ocean_meridional_overturning_mass_streamfunction",
    "long_name": "Ocean Meridional Overturning Mass Streamfunction",
    "comment":"Overturning mass streamfunction arising from all advective mass transport processes, resolved and parameterized.",
    "units": "Sv",
}
ds["msftmz"] = ds[
        "msftmz"
    ].assign_coords(basin="atlantic_ocean", lat=26.0, olevel=0)
ds['lat'] = fix_lat(ds)
ds = set_ods_coords(ds)

# Fix up the dimensions
time_range = f"{ds["time"].min().dt.year:d}{ds["time"].min().dt.month:02d}"
time_range += f"-{ds["time"].max().dt.year:d}{ds["time"].max().dt.month:02d}"

# Populate attributes
ds.attrs = {
    "activity_id": "obs4MIPs",
    "contact": "Ben Moat (ben.moat@noc.ac.uk)",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_date": generate_stamp,
    "comment":"Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet; Units Sv for variable msftmz must be convertible to canonical units kg s-1" ,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "2.5",
    "doi": "10.5285/223b34a32dc5c945e0637086abc0f274",
    "frequency": "mon",
    "grid": "N/A",
    "grid_label": "NA",
    "has_auxdata":"False",
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
    "source_type": "insitu",
    "source_label":"RAPID",
    "source_version_number": "2023.1",
    "title": "RAPID MOC timeseries",
    "variable_id": "msftmz",
    "variant_label": "REF",
    "variant_info":"CMORized product prepared by ILAMB and CMIP IPO",
    "version":today,
    "tracking_id": generate_trackingid,
}

#ds = set_ods_var_attrs(ds, "msftmz")
#ds = set_ods_global_attributes(ds, **ds.attrs)

out_path = (
    "/home/users/dhegedus/CMIPplacement/beta/{activity_id}/{institution_id}/{source_id}/{frequency}/{variable_id}/{grid_label}/"
    + today
    + "/{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc"
).format(**ds.attrs, time_mark=time_range)

os.makedirs(os.path.dirname(out_path), exist_ok=True)

ds.to_netcdf(out_path,
    encoding={'time_bnds':{'_FillValue':None, "dtype": "float64", "units": "days since 2004-04-01 00:00:00"},
                'time':{"dtype": "float64", "units": "days since 2004-04-01 00:00:00",
                        '_FillValue':None, "calendar": "gregorian"},
                'lat': {'zlib': False, '_FillValue': None}},
)
