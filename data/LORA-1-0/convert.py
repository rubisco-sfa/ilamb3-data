from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

import ilamb3_data as ild

# Download the data
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
        ild.download.from_html(remote_source, source)
    local_sources.append(source)

# Set timestamps and tracking id
download_stamp = ild.output.utc_timestamp(local_sources[0].stat().st_mtime)
creation_stamp = ild.output.utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = ild.output.new_tracking_id()

# Load the dataset for adjustments
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_mfdataset(local_sources, engine="netcdf4", decode_times=time_coder)

# Get attribute info for mrro
mrro_info = ild.variable.lookup_cmip6("mrro", "mrro")

# Set correct attribute information for the vars
ds = ild.variable.standardize(
    ds,
    "mrro",
    units=mrro_info["variable_units"],
    standard_name=mrro_info["cf_standard_name"],
    long_name=mrro_info["variable_long_name"],
    ancillary_variables="mrro_sd",
    target_dtype=np.dtype("float32"),
    convert=True,
)

# Assign ancillary variables
ds = ds.assign(mrro_sd=(("time", "lat", "lon"), ds.mrro_sd.values))
ds.mrro_sd.attrs = {
    "long_name": f"{ds.mrro.attrs['standard_name']} standard_deviation",
    "cell_methods": "area: standard_deviation",  # copied from source data
}
ds.mrro_sd.encoding = {
    "_FillValue": None,  # CMOR default
    "dtype": "float32",
}

# Clean up attrs
ds = ild.time.standardize(ds, bounds_frequency="M")
ds = ild.lat.standardize(ds)
ds = ild.lon.standardize(ds)
ds = ild.bounds.build_from_centers(ds, "lat")
ds = ild.bounds.build_from_centers(ds, "lon")
ds = ild.output.order_dimensions(ds)

# Set global attributes and export
out_ds = ild.global_attrs.set_ods26(
    ds,
    aux_uncertainty_id="sd",
    comment="Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet",
    contact="Sanaa Hobeichi (s.hobeichi@student.unsw.edu.au)",
    creation_date=creation_stamp,
    dataset_contributor="Morgan Steckler",
    doi="10.25914/5b612e993d8ea",
    frequency="mon",
    grid="0.5x0.5 degree latitude x longitude",
    grid_label="gn",
    has_aux_unc="TRUE",
    history=f"""
{download_stamp}: downloaded {remote_sources};
{creation_stamp}: converted to obs4MIPs format""",
    institution="ARC Centre of Excellence for Climate System Science, NSW, Australia",
    institution_id="ARCCSS",
    license="https://creativecommons.org/licenses/by-nc-sa/4.0/",
    nominal_resolution="0.5 degree",
    processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/LORA-1-0/convert.py",
    product="observations",
    realm="land",
    references="Hobeichi, Sanaa, 2018: Linear Optimal Runoff Aggregate v1.0. NCI National Research Data Collection, doi:10.25914/5b612e993d8ea",
    region="global_land",
    source="LORA 1.0 (2018): Linear Optimal Runoff Aggregate",
    source_id="LORA-1-0",
    source_data_retrieval_date=download_stamp,
    source_data_url="https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/LORA/v1-0/catalog.html",
    source_label="LORA",
    source_type="gridded_insitu",
    source_version_number="1.0",
    title="Linear Optimal Runoff Aggregate (version v1.0)",
    tracking_id=tracking_id,
    variable_id="mrro",
    variant_label="ILAMB",
    variant_info="CMORized product prepared by ILAMB",
    version=f"v{today_stamp}",
)

# Define chunk sizes based on a ~1–3 MB target per chunk
# (float32 → 4 bytes/element; 6×180×360 ≃ 388 800 elements → ~1.5 MB)
chunks = dict(time=6, lat=180, lon=360)
ds_chunked = out_ds.chunk(chunks)
encoding = {}
for var, da in ds_chunked.data_vars.items():
    encoding_copy = da.encoding.copy()
    # only data variables need chunksizes
    if var in ("mrro", "mrro_sd"):
        encoding_copy["chunksizes"] = tuple(
            chunks.get(dim, da.sizes[dim]) for dim in da.dims
        )
    encoding[var] = encoding_copy

# Prep for export
out_path = ild.output.filename_from_attrs(ds_chunked.attrs)
ds_chunked.to_netcdf(out_path, encoding=encoding, format="NETCDF4")
