import time
from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr

import ilamb3_data as ild

RAW_PATH = Path("_raw")
if not RAW_PATH.is_dir():
    RAW_PATH.mkdir()

# define sources
remote_sources = [
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2003.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2004.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2005.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2006.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2007.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2008.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2009.nc",
]
local_sources = [str(RAW_PATH / Path(s).name) for s in remote_sources]

# ensure we have downloaded the data
for remote_source, local_source in zip(remote_sources, local_sources):
    if not Path(local_source).is_file():
        ild.download_from_html(remote_source, local_source)
    download_stamp = ild.gen_utc_timestamp(Path(local_source).stat().st_mtime)

# open and rename some variables
ds = xr.open_mfdataset(local_sources)
ds = ds.rename({"hfds": "hfdsl", "hfds_sd": "hfdsl_sd", "rs": "rns", "rs_sd": "rns_sd"})

# fix up coordinates
ds["time"] = [cf.DatetimeGregorian(t.dt.year, t.dt.month, t.dt.day) for t in ds["time"]]
ds = ild.set_time_attrs(
    ds, bounds_frequency="M", ref_date=cf.DatetimeGregorian(2003, 1, 1)
)
ds = ild.set_lat_attrs(ds)
ds = ild.set_lon_attrs(ds)
ds = ild.set_coord_bounds(ds, "lat")
ds = ild.set_coord_bounds(ds, "lon")

# write netcdf for each variable
generate_stamp = time.strftime("%Y%m%d")
tracking_id = ild.gen_trackingid()
variables = [v for v in ds if ("_sd" not in v and "_bnds" not in v)]
for var in variables:
    uncert = f"{var}_sd"
    out = ds.drop_vars(
        [d for d in ds if d not in [var, uncert] and not d.endswith("_bnds")]
    )

    # get standard/long name info and manage when it's not in cmip6 CV or is uncertainty
    if "_sd" not in var:
        try:
            var_info = ild.get_cmip6_variable_info(var, var)
        except Exception:
            var_info = {
                "cf_standard_name": out[var].attrs.get("standard_name", var),
                "variable_long_name": out[var].attrs.get(
                    "long_name", var.replace("_", " ").title()
                ),
                "variable_units": out[var].attrs.get("units", ""),
            }
            var_info["variable_long_name"] = (
                var_info["variable_long_name"].replace("_", " ").title()
            )

    # format the var attrs
    out = ild.set_var_attrs(
        out,
        var=var,
        cmip6_units=var_info["variable_units"],
        cmip6_standard_name=var_info["cf_standard_name"],
        cmip6_long_name=var_info["variable_long_name"],
        ancillary_variables=uncert,
        target_dtype=np.float32,
        convert=False,
    )

    # drop some straggling attrs
    out[var].attrs.pop("ALMA_short_name", None)

    # format the ancillary var attrs
    out[uncert].attrs = {
        "standard_name": f"{var} standard_deviation",
        "units": out[var].attrs["units"],
    }
    out[uncert].encoding["_FillValue"] = np.float32(1.0e20)

    # set the global attrs
    out = ild.set_ods26_global_attrs(
        out,
        aux_uncertainty_id="sd",
        contact="Sanaa Hobeichi (s.hobeichi@unsw.edu.au)",
        creation_date=generate_stamp,
        dataset_contributor="Nathan Collier",
        doi="https://doi.org/10.25914/5c872258dc183",
        frequency="mon",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="TRUE",
        history=f"""
{download_stamp}: downloaded using https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f4854_2536_6084_5147;
{generate_stamp}: converted to obs4MIP format""",
        institution="University of New South Wales, Sydney, New South Wales, AUS",
        institution_id="UNSW",
        license="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        nominal_resolution="0.5 degree",
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
        title=f"Conserving Land-Atmosphere Synthesis Suite ({var})",
        tracking_id=tracking_id,
        variable_id=var,
        variant_label="ILAMB",
        variant_info="CMORized product prepared by ILAMB",
        version=f"v{generate_stamp}",
    )
    out_path = ild.create_output_filename(out.attrs)
    out.to_netcdf(out_path)
