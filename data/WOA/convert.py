import re
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd
import xarray as xr

from ilamb3_data import (
    create_output_filename,
    download_from_html,
    gen_trackingid,
    gen_utc_timestamp,
    get_cmip6_variable_info,
    set_climatology_attrs,
    set_coord_bounds,
    set_depth_attrs,
    set_lat_attrs,
    set_lon_attrs,
    set_ods_global_attrs,
    set_var_attrs,
)

# DATA NOTES:
# Climatologies:
# --------------
# 00 = 1-year
# 01-12 = each month
# 13-16 = each season

# Within-file naming
# ------------------
# statistical mean = average of quality controlled raw observational stats; simple averaging; gridcells subject to missing data/uneven sampling
# objectively analyzed climatology = take statistical means and fill gaps + interpolate unevenly sampled gridcells

# salinity, temperature
# ---------------------
# Decadal averages: 5564 (1955-1964), 6574 (1965-1974), 7584 (1975-1984), 8594 (1985-1994), 95A4 (1995-2004), A5B4 (2005-2014), B5C2 (2015-2022)
# 30-year averages: decav71A0 (1971-2000), decav81B0 (1981-2010), decav91C0 (1991-2020)
# Average of 7 decades (1955-2022): decav

# phosphate, nitrate, oxygen, silicate
# ------------------------------------
# Average of available data (1965-2022): all

AVG_PERIODS = ["decav"]
NAME_CONVERSIONS = {
    "oxygen": {"o": "o2"},
    "phosphate": {"p": "po4"},
    "nitrate": {"n": "no3"},
    "silicate": {"i": "si"},
    "temperature": {"t": "thetao"},
    "salinity": {"s": "so"},
}

# Build remote_source_dict = {full_name: {period: [urls]}}
remote_source_dict = {}

for full_name, abbv_dict in NAME_CONVERSIONS.items():
    code, cmip_var = next(iter(abbv_dict.items()))
    periods = AVG_PERIODS if full_name in ("temperature", "salinity") else ["all"]

    for period in periods:
        urls = [
            f"https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/{full_name}/netcdf/{period}/1.00/"
            f"woa23_{period}_{code}{month:02d}_01.nc"
            for month in range(1, 13)
        ]
        remote_source_dict.setdefault(full_name, {})[period] = urls

# Download + build local_source_dict = {full_name: {period: [local_paths]}}
local_source_dict = {}
for full_name, periods in remote_source_dict.items():
    for period, urls in periods.items():
        local_paths = []
        for url in urls:
            # derive a stable local path: _raw/<full_name>/<period>/<filename>
            fname = Path(urlparse(url).path).name
            dest = Path("_raw") / full_name / period / fname
            dest.parent.mkdir(parents=True, exist_ok=True)

            if not dest.is_file():
                download_from_html(url, str(dest))

            local_paths.append(str(dest))
        local_source_dict.setdefault(full_name, {})[period] = local_paths

# Set timestamps and tracking id
local_source_ex = local_source_dict["temperature"]["decav"][0]
download_stamp = gen_utc_timestamp(Path(local_source_ex).stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

for tuple in local_source_dict.items():
    name, period_dict = iter(tuple)
    period, files = next(iter(period_dict.items()))
    var, std_var = next(iter(NAME_CONVERSIONS[name].items()))
    print(f"\nWorking on var '{var}' ({std_var})...")

    ds = xr.open_mfdataset(files, decode_times=False, coords="all")
    # get the starting datetime from the time units attribute
    ref_dt = pd.Timestamp(
        re.split(r"\s+since\s+", ds.time.attrs["units"], 1)[-1].strip()
    )
    print(ref_dt)

    # set climatology bounds
    # convert "months since" to "days since"
    cb_days = np.array(
        [
            [
                ((ref_dt + pd.DateOffset(months=int(lower))) - ref_dt)
                / np.timedelta64(1, "D"),
                ((ref_dt + pd.DateOffset(months=int(upper))) - ref_dt)
                / np.timedelta64(1, "D"),
            ]
            for lower, upper in ds["climatology_bounds"].values
        ]
    )
    ds["climatology_bnds"] = xr.DataArray(
        cb_days,
        dims=ds["climatology_bounds"].dims,
        attrs=ds["climatology_bounds"].attrs,
    )

    # set time
    # get midpoints for monthly bounds
    clim_bounds = ds["climatology_bnds"].values  # (time, 2) in days since ref_dt
    months = [(ref_dt + pd.to_timedelta(lo, "D")).month for lo in clim_bounds[:, 0]]
    year_start, year_end = (
        ref_dt.year,
        (ref_dt + pd.to_timedelta(clim_bounds[0, 1], "D")).year,
    )
    year_mid = (
        year_start + (year_end - year_start) // 2
    )  # central year (1955-2022 -> 1988)

    starts = [pd.Timestamp(year_mid, month, 1) for month in months]
    nexts = [start + pd.DateOffset(months=1) for start in starts]
    mid_ts = [start + (next - start) / 2 for start, next in zip(starts, nexts)]
    time_days = np.array(
        [(time - ref_dt) / np.timedelta64(1, "D") for time in mid_ts], dtype="float32"
    )

    ds = ds.assign_coords(
        time=xr.DataArray(time_days, dims=ds["time"].dims, attrs=ds["time"].attrs)
    )
    ds["time"].attrs.update(
        {"units": f"days since {ref_dt:%Y-%m-%d %H:%M:%S}", "calendar": "gregorian"}
    )

    # keep Objectively analyzed climatology (_an) & Standard error of the mean of each variable (_se)
    ds = (
        ds.drop_vars(
            [
                f"{var}_mn",  # Statistical mean
                f"{var}_dd",  # Number of observations
                f"{var}_sd",  # Standard deviation about the statistical mean of each variable
                f"{var}_oa",  # Statistical mean minus the climatological mean
                f"{var}_ma",  # Seasonal or monthly climatology minus the annual climatology
                f"{var}_gp",  # Number of one-degree squares within the smallest radius of influence
                f"{var}_sdo",  # Objectively analyzed standard deviation
                f"{var}_sea",  # Standard error of the analysis
                "crs",
                "climatology_bounds",
            ],
            errors="ignore",
        )
        .rename_vars({f"{var}_an": std_var, f"{var}_se": f"{std_var}_se"})
        .rename_dims({"nbounds": "bnds"})
    )

    # set variable attrs
    ds[std_var].attrs["units"] = "umol kg-1"
    info = get_cmip6_variable_info(std_var)
    ds = set_var_attrs(
        ds,
        var=std_var,
        cmip6_units=info["variable_units"],
        cmip6_standard_name=info["cf_standard_name"],
        cmip6_long_name=info["variable_long_name"],
        ancillary_variables=f"{std_var}_se",
        cell_methods="area: mean depth: mean time: mean within years time: mean over years",
        target_dtype=np.float32,
        convert=False,
    )
    ds[std_var].attrs.pop("grid_mapping", None)

    # set ancillary variable attrs
    ds[f"{std_var}_se"].attrs = {
        "standard_name": f"{info['cf_standard_name']} standard_error",
        "units": "umol kg-1",
    }
    ds[f"{std_var}_se"].encoding = {"_FillValue": None}

    # standardize the dimensions
    ds = set_climatology_attrs(ds, bounds_frequency="M")
    ds = set_lat_attrs(ds)
    ds = set_lon_attrs(ds)
    assert (ds.depth_bnds == ds.depth_bnds.isel(time=0)).all()
    ds = set_depth_attrs(
        ds,
        bounds=ds.depth_bnds.isel(time=0).to_numpy(),
        units="meters",
        positive="down",
        long_name="depth of sea water",
    )
    ds = set_coord_bounds(ds, "lat")
    ds = set_coord_bounds(ds, "lon")
    ds = ds.sortby(["time", "depth", "lat", "lon", "bnds"])
    ds = ds[sorted(ds.data_vars)]

    # Populate attributes
    title = ds.attrs["title"]
    reference = ds.attrs["references"]
    ds = set_ods_global_attrs(
        ds,
        activity_id="obs4MIPs",
        aux_variable_id=f"{std_var}_se",
        comment="Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet",
        contact="NOAA National Centers for Environmental Information (ncei.info@noaa.gov)",
        conventions="CF-1.12 ODS-2.5",
        creation_date=creation_stamp,
        dataset_contributor="Morgan Steckler",
        data_specs_version="2.5",
        doi="10.25921/va26-hv25",
        external_variables="N/A",
        frequency="monC",
        grid="1x1 degree latitude x longitude",
        grid_label="gn",
        has_auxdata="True",
        history=f"""
    {download_stamp}: downloaded file;
    {creation_stamp}: converted to obs4MIPs format""",
        institution="National Oceanic and Atmospheric Administration, National Centers for Environmental Information, Ocean Climate Laboratory, Asheville, NC, USA",
        institution_id="NOAA-NCEI-OCL",
        license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
        nominal_resolution="1x1 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/WOA/convert.py",
        product="observations",
        realm="ocean",
        references=reference,
        region="global_ocean",
        source="",
        source_id="WOA-23",
        source_data_retrieval_date=download_stamp,
        source_data_url="",
        source_label="WOA",
        source_type="gridded_insitu",
        source_version_number="1.0",
        title=title,
        tracking_id=tracking_id,
        variable_id=std_var,
        variant_label="REF",
        variant_info="CMORized product prepared by ILAMB and CMIP IPO",
        version=f"v{today_stamp}",
    )

    # Prep for export
    out_path = create_output_filename(ds.attrs)
    ds.to_netcdf(out_path, format="NETCDF4")
