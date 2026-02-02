import time
from pathlib import Path

import cftime as cf
import xarray as xr

import ilamb3_data as ild

RAW_PATH = Path("_raw")

# define sources
remote_sources = [
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2003.nc",
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2004.nc",
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2005.nc",
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2006.nc",
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2007.nc",
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2008.nc",
    "http://dapds00.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2009.nc",
]
local_sources = [RAW_PATH / Path(s).name for s in remote_sources]

# ensure we have downloaded the data
for remote_source, local_source in zip(remote_sources, local_sources):
    if not local_source.is_file():
        ild.download_from_html(remote_source, local_source)
    download_stamp = time.strftime(
        "%Y-%m-%d", time.localtime((local_source).stat().st_ctime)
    )

# open and rename some variables
ds = xr.open_mfdataset(local_sources)
ds = ds.rename({"hfds": "hfdsl", "hfds_sd": "hfdsl_sd", "rs": "rns", "rs_sd": "rns_sd"})

# fix up coordinates
ds["time"] = [cf.DatetimeGregorian(t.dt.year, t.dt.month, t.dt.day) for t in ds["time"]]
ds = ild.set_time_attrs(ds, bounds_frequency="M")
ds = ild.set_coord_bounds(ds, "lat")
ds = ild.set_coord_bounds(ds, "lon")
ds = ild.set_lat_attrs(ds)
ds = ild.set_lon_attrs(ds)

# write out files
generate_stamp = time.strftime("%Y-%m-%d")
tracking_id = ild.gen_trackingid()
variables = [v for v in ds if ("_sd" not in v and "_bnds" not in v)]
for v in variables:
    uncert = f"{v}_sd"
    out = ds.drop_vars([d for d in ds if d not in [v, uncert, "time_bnds"]])
    out[v].attrs["ancillary_variables"] = uncert
    out[uncert].attrs = {
        "standard_name": f"{v} standard deviation",
        "units": out[v].attrs["units"],
    }

    out = ild.set_ods26_global_attrs(
        out,
        aux_uncertainty_id="sd",
        comment="",
        contact="Sanaa Hobeichi (s.hobeichi@unsw.edu.au)",
        creation_date=generate_stamp,
        dataset_contributor="Nathan Collier",
        doi="10.25914/5c872258dc183",
        frequency="mon",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="TRUE",
        history=f"""
{download_stamp}: downloaded using https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f4854_2536_6084_5147;
{generate_stamp}: converted to obs4MIP format""",
        institution="University of New South Wales",
        institution_id="UNSW",
        license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution - 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
        nominal_resolution="50 km",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/CLASS-1-1/convert.py",
        product="derived",
        realm="land",
        references="Hobeichi, Sanaa and Abramowitz, Gab and Evans, Jason, Conserving Land-Atmosphere Synthesis Suite (CLASS), Journal of Climate, 33(5), 2020, 10.1175/JCLI-D-19-0036.1.",
        region="global_land",
        source="Ground Heat Flux (GLDAS, MERRALND, MERRAFLX, NCEP_DOII, NCEP_NCAR), Sensible Heat Flux(GLDAS, MERRALND, MERRAFLX, NCEP_DOII, NCEP_NCAR, MPIBGC, Princeton), Latent Heat Flux(DOLCE1.0), Net Radiation (GLDAS, MERRALND, NCEP_DOII, NCEP_NCAR, ERAI, EBAF4.0), Precipitation(REGEN1.1), Runoff(LORA1.0), Change in Water storage(GRACE(GFZ, JPL, CSR))",
        source_id="CLASS-1-1",
        source_data_retrieval_date=download_stamp,
        source_data_url=",".join(remote_sources),
        source_label="CLASS",
        source_type="gridded_insitu",
        source_version_number="1.1",
        title=f"Conserving Land-Atmosphere Synthesis Suite ({v})",
        tracking_id=tracking_id,
        variable_id=v,
        variant_label="ILAMB",
        variant_info="CMORized product prepared by ILAMB",
        version=f"v{generate_stamp.replace('-', '')}",
    )
    out_path = ild.create_output_filename(out.attrs)
    out.to_netcdf(out_path)
