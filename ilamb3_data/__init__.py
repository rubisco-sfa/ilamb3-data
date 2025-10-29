import datetime
import json
import os
import re
import urllib.request
import uuid
import warnings
from calendar import monthrange
from typing import Optional

import cftime as cf
import numpy as np
import pandas as pd
import pooch
import requests
import xarray as xr
from cf_units import Unit
from intake_esgf import ESGFCatalog
from pandas.tseries.frequencies import to_offset
from tqdm import tqdm

from . import biblatex_builder


def create_registry(registry_file: str) -> pooch.Pooch:
    """
    Given registry file, return the pooch ilamb reference data catalog.
    Returns: The intake ilamb reference data catalog (pooch.Pooch)
    """

    registry = pooch.create(
        path=pooch.os_cache("ilamb3"),
        base_url="https://www.ilamb.org/ilamb3-data",
        version="0.1",
        env="ILAMB_ROOT",
    )
    registry.load_registry(registry_file)
    return registry


def download_from_html(remote_source: str, local_source: str | None = None) -> str:
    """
    Download a file from a remote URL to a local path.
    If the "content-length" header is missing, it falls back to a simple download.
    """
    if local_source is None:
        local_source = os.path.basename(remote_source)
    if os.path.isfile(local_source):
        return local_source

    resp = requests.get(remote_source, stream=True)
    try:
        total_size = int(resp.headers.get("content-length"))
    except (TypeError, ValueError):
        total_size = 0

    with open(local_source, "wb") as fdl:
        if total_size:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=local_source
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        fdl.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    fdl.write(chunk)
    return local_source


def download_from_zenodo(record: dict, download_dir: str):
    """
    Download all files from a Zenodo record dict into a '_temp' directory.
    Example for getting a Zenodo record:

        # Specify the dataset title you are looking for
        dataset_title = "Global Fire Emissions Database (GFED5) Burned Area"

        # Build the query string to search by title
        params = {
            "q": f'title:"{dataset_title}"'
        }

        # Define the Zenodo API endpoint
        base_url = "https://zenodo.org/api/records"

        # Send the GET request
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print("Error during search:", response.status_code)
            exit(1)

        # Parse the JSON response
        data = response.json()

        # Get record dictionary
        records = data['hits']['hits']
        record = data['hits']['hits'][0]
    """
    os.makedirs(download_dir, exist_ok=True)

    title = record.get("metadata", {}).get("title", "No Title")
    pub_date = record.get("metadata", {}).get("publication_date", "No publication date")
    print(f"Found record:\n  Title: {title}\n  Publication Date: {pub_date}")

    for file_info in record.get("files", []):
        file_name = file_info.get("key")
        file_url = file_info.get("links", {}).get("self")
        local_file = os.path.join(download_dir, file_name)

        if file_url:
            print(f"Downloading {file_name} from {file_url} into {download_dir}...")
            download_from_html(file_url, local_source=local_file)
        else:
            print(f"File URL not found for file: {file_name}")


def create_output_filename(attrs: dict) -> str:
    """
    Generate a NetCDF filename using required attribute dictionary
    Returns: Formatted filename (str)
    """
    required_keys = [
        "activity_id",
        "institution_id",
        "source_id",
        "frequency",
        "variable_id",
        "grid_label",
        "version",
    ]

    missing = [key for key in required_keys if key not in attrs]
    if missing:
        raise ValueError(
            f"Missing required attributes: {', '.join(missing)}. "
            f"Expected keys: {', '.join(required_keys)}"
        )

    filename = "{activity_id}_{institution_id}_{source_id}_{frequency}_{variable_id}_{grid_label}_{version}.nc".format(
        **attrs
    )
    return filename


def get_cmip6_variable_info(variable_id: str) -> dict[str, str]:
    """
    Given a CMIP6 variable_id, return a dictionary of its standard_name, long_name, and units.
    """
    df = ESGFCatalog().variable_info(variable_id)
    if variable_id not in df.index:
        raise ValueError(
            f"Variable ID '{variable_id}' not found in CMIP6 variable info."
        )
    return df.loc[variable_id].to_dict()


def time_bounds_from_frequency(
    freq: str, sdate: cf.datetime, edate: cf.datetime
) -> np.ndarray:
    """
    Build CF-style time bounds [start, end) for a given frequency and start/end dates.
    Returns: np.ndarray of shape (ntime, 2), dtype=object, with cftime.DatetimeGregorian.
    """
    # Validate inputs
    if edate < sdate:
        raise ValueError("edate must be >= sdate.")

    # Convert cftime -> pandas Timestamp
    def to_ts(dt):
        return pd.Timestamp(
            dt.year,
            dt.month,
            dt.day,
            getattr(dt, "hour", 0),
            getattr(dt, "minute", 0),
            getattr(dt, "second", 0),
            getattr(dt, "microsecond", 0),
        )

    s_ts, e_ts = to_ts(sdate), to_ts(edate)

    # Validate/normalize frequency
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError as e:
        raise ValueError(f"Invalid freq {freq!r}: {e}") from e
    if offset is None:
        raise ValueError(f"Invalid freq {freq!r}")

    # Periods covering [sdate, edate] (inclusive of both periods that contain them)
    periods = pd.period_range(start=s_ts, end=e_ts, freq=freq)
    if len(periods) == 0:
        raise ValueError("No periods found for the given bounds and frequency.")

    starts = periods.start_time
    ends = (periods + 1).start_time  # ends[i] == starts[i+1]

    # pandas Timestamp -> cftime.DatetimeGregorian
    def ts_to_cf(ts):
        return cf.DatetimeGregorian(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond
        )

    bnds = np.array(
        [(ts_to_cf(lo), ts_to_cf(hi)) for lo, hi in zip(starts, ends)], dtype=object
    )
    return bnds


def climatology_bounds_from_frequency(
    freq: str,
    sdate: cf.datetime,
    edate: cf.datetime,
) -> np.ndarray:
    """
    Build CF-style climatology bounds [start, end) for a *single* cycle of `freq`.
    Starts use the first climatology year (sdate.year); ends use the last (edate.year),
    with +1 year for wrap-around (e.g., Dec->Jan). Matches CF §7.4 semantics.
    """
    if edate < sdate:
        raise ValueError("edate must be >= sdate.")

    # cftime -> pandas Timestamp (we only need year/month/day)
    def to_ts(dt):
        return pd.Timestamp(
            dt.year,
            dt.month,
            dt.day,
            getattr(dt, "hour", 0),
            getattr(dt, "minute", 0),
            getattr(dt, "second", 0),
            getattr(dt, "microsecond", 0),
        )

    s_ts, e_ts = to_ts(sdate), to_ts(edate)

    try:
        _ = pd.tseries.frequencies.to_offset(freq)
    except ValueError as e:
        raise ValueError(f"Invalid freq {freq!r}: {e}") from e

    # Build ONE template year of periods, then transplant years for bounds
    # Use a non-leap anchor year; monthly/seasonal work fine with 2001.
    base_start = pd.Timestamp(2001, 1, 1)
    base_end = pd.Timestamp(2001, 12, 31, 23, 59, 59)
    periods = pd.period_range(start=base_start, end=base_end, freq=freq)
    if len(periods) == 0:
        raise ValueError("No periods found for the given frequency within a year.")

    starts = periods.start_time
    ends = (periods + 1).start_time  # [start, next_start)

    def rep_year(ts: pd.Timestamp, year: int) -> pd.Timestamp:
        # Safe because period.start_time is always a valid first-of-period timestamp
        return ts.replace(year=year)

    rep_bounds = []
    first_year = s_ts.year  # lower column: first climatology year
    last_year = e_ts.year  # upper column: last climatology year (or +1 on wrap)
    for s_i, e_i in zip(starts, ends):
        s_rep = rep_year(s_i, first_year)
        end_year = last_year
        # If end-of-bin is "earlier" in the year than start-of-bin, it wraps into next year
        if (e_i.month, e_i.day, e_i.hour, e_i.minute, e_i.second, e_i.microsecond) < (
            s_i.month,
            s_i.day,
            s_i.hour,
            s_i.minute,
            s_i.second,
            s_i.microsecond,
        ):
            end_year = last_year + 1
        e_rep = rep_year(e_i, end_year)
        rep_bounds.append((s_rep, e_rep))

    # pandas Timestamp -> cftime.DatetimeGregorian
    def ts_to_cf(ts: pd.Timestamp) -> cf.DatetimeGregorian:
        return cf.DatetimeGregorian(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond
        )

    bnds = np.array(
        [(ts_to_cf(lo), ts_to_cf(hi)) for lo, hi in rep_bounds], dtype=object
    )
    return bnds


def build_time(
    sdate: cf.datetime,
    edate: cf.datetime,
    freq: str,
    ref_date: Optional[cf.datetime] = None,
    climatology: bool = False,
) -> xr.DataArray:
    """
    Build a CF-compliant time coordinate DataArray with attributes and encoding.

    For climatology=False:
      - freq != 'fx': regular bounds & midpoints over [sdate, edate]
      - freq == 'fx' : ONE time at the midpoint of [sdate, edate], bounds = [sdate, edate]

    For climatology=True:
      - freq != 'fx': climatology_bnds span from sdate.year to edate.year (per CF §7.4); time is a representative cycle
      - freq == 'fx' : ONE time at the midpoint of [sdate, edate], climatology_bnds = [sdate, edate]
    """
    # validate inputs
    if edate < sdate:
        raise ValueError("edate must be >= sdate.")
    if ref_date is None:
        raise ValueError("ref_date must be provided.")
    if ref_date > sdate:
        raise ValueError("ref_date must be <= sdate.")
    if ref_date.calendar != sdate.calendar or ref_date.calendar != edate.calendar:
        raise ValueError("sdate, edate, and ref_date must have the same calendar.")

    fx = str(freq).lower() == "fx"

    # --- Build bounds + midpoints ---
    if fx:
        # single midpoint + single bounds pair
        bnds = np.array([(sdate, edate)], dtype=object)
        mid_dts = [sdate + (edate - sdate) / 2]
    else:
        if climatology:
            bnds = climatology_bounds_from_frequency(freq, sdate, edate)
            # representative cycle midpoints: use month from lower bound, 15th in a midpoint year
            e_ts = pd.Timestamp(
                edate.year, getattr(edate, "month", 1), getattr(edate, "day", 1)
            )
            last_full_year = (e_ts - pd.Timedelta(days=1)).year
            mid_year = sdate.year + (last_full_year - sdate.year) // 2
            months = [lo.month for lo, _ in bnds]
            mid_dts = [
                cf.DatetimeGregorian(mid_year, m, 15, 0, 0, 0, 0) for m in months
            ]
        else:
            bnds = time_bounds_from_frequency(freq, sdate, edate)
            mid_dts = [lo + (hi - lo) / 2 for lo, hi in bnds]

    # --- Units & convert midpoints to numeric gregorian (stable for write) ---
    ref_str = f"{ref_date.year:04d}-{ref_date.month:02d}-{ref_date.day:02d} 00:00:00"
    units = f"days since {ref_str}"

    orig_calendar_nums = cf.date2num(mid_dts, units, sdate.calendar)
    gregorian_dts = cf.num2date(orig_calendar_nums.tolist(), units, "gregorian")
    gregorian_nums = cf.date2num(gregorian_dts, units, "gregorian")
    gregorian_nums_arr = np.array(gregorian_nums, dtype="float64")

    # time coordinate
    time_da = xr.DataArray(
        data=gregorian_nums_arr,
        dims=("time",),
        coords={"time": gregorian_nums_arr},
        attrs={
            "axis": "T",
            "standard_name": "time",
            "long_name": "time",
            "bounds": "time_bnds",
            "units": units,
            "calendar": "gregorian",
        },
        name="time",
    )

    # bounds variable
    if climatology:
        time_da.attrs.pop("bounds", None)
        time_da.attrs["climatology"] = "climatology_bnds"
        bnds_da = xr.DataArray(
            data=bnds.astype(object),
            dims=("time", "bnds"),
            coords={"time": time_da},
            name="climatology_bnds",
        )
        bnds_da.attrs = {"units": units, "calendar": "gregorian"}
        bnds_da.encoding = {"dtype": "float64", "_FillValue": None}
    else:
        bnds_da = xr.DataArray(
            data=bnds.astype(object),
            dims=("time", "bnds"),
            coords={"time": time_da},
            name="time_bnds",
        )
        bnds_da.attrs = {"units": units, "calendar": "gregorian"}
        bnds_da.encoding = {"dtype": "float64", "_FillValue": None}

    time_da.encoding = {"dtype": "float64", "_FillValue": None}
    return time_da, bnds_da


def cf_to_num(
    da: xr.DataArray, units: str, calendar: str = "gregorian", dtype="float64"
) -> xr.DataArray:
    """Convert cftime/Python datetimes in a DataArray to numeric CF time,
    keeping dims/coords and updating attrs/encoding."""
    if calendar == "standard":
        calendar = "gregorian"

    # If already numeric, just cast
    if np.issubdtype(da.dtype, np.number):
        out = da.astype(dtype)
    else:
        out = xr.apply_ufunc(
            cf.date2num,
            da,
            kwargs={"units": units, "calendar": calendar},
            vectorize=True,  # elementwise over any shape (time) or (time,2)
            dask="allowed",
            output_dtypes=[np.dtype(dtype)],
        )

    # attrs: copy and enforce updated units/calendar
    out.attrs = {**da.attrs, "units": units, "calendar": calendar}

    # encoding: start from existing, but force dtype and _FillValue=None
    enc = dict(getattr(da, "encoding", {}))  # shallow copy if present
    enc["dtype"] = np.dtype(dtype)
    enc["_FillValue"] = None

    out.encoding = enc
    return out


def set_time_attrs(
    ds: xr.Dataset,
    bounds_frequency: str,
    ref_date: Optional[cf.datetime] = None,
    create_new_time: bool = False,
    sdate: Optional[cf.datetime] = None,
    edate: Optional[cf.datetime] = None,
    climatology: bool = False,
    clim_sdate: Optional[cf.datetime] = None,
    clim_edate: Optional[cf.datetime] = None,
) -> xr.Dataset:
    """
    See docstring in your version; adds support for bounds_frequency='fx':
    - time = midpoint(sdate, edate)
    - bounds = [[sdate, edate]]
    """
    if bounds_frequency is None:
        raise ValueError("bounds_frequency must be provided.")
    if ref_date is None:
        try:
            ref_date = ds["time"].values[0]
            warnings.warn(
                "ref_date not provided; using first time value in dataset as ref_date."
            )
        except Exception as e:
            raise ValueError(
                "ref_date must be provided if dataset has no time coordinate."
            ) from e

    fx = str(bounds_frequency).lower() == "fx"

    # --- Create or reformat ---
    if create_new_time:
        # require explicit sdate/edate for fx or generic build
        if sdate is None or edate is None:
            raise ValueError(
                "sdate and edate must be provided to create time from scratch."
            )
        time_da, bnds_da = build_time(
            sdate, edate, bounds_frequency, ref_date, climatology
        )
        ds = ds.assign_coords({"time": time_da})
        ds = ds.assign(
            {"time_bnds": bnds_da} if not climatology else {"climatology_bnds": bnds_da}
        )
    else:
        if "time" not in ds:
            raise ValueError("Dataset has no 'time' coordinate to reformat.")
        if not isinstance(ds["time"].values[0], cf.datetime):
            raise ValueError("Dataset 'time' coordinate values are not cftime objects.")

        if climatology:
            if fx:
                if clim_sdate is None or clim_edate is None:
                    raise ValueError(
                        "clim_sdate and clim_edate are required when bounds_frequency='fx'."
                    )
                time_da, bnds_da = build_time(
                    clim_sdate, clim_edate, "fx", ref_date, True
                )
            else:
                if clim_sdate is None or clim_edate is None:
                    raise ValueError(
                        "clim_sdate and clim_edate must be provided for climatology=True."
                    )
                time_da, bnds_da = build_time(
                    clim_sdate, clim_edate, bounds_frequency, ref_date, True
                )
        else:
            if fx:
                if sdate is None or edate is None:
                    raise ValueError(
                        "sdate and edate are required when bounds_frequency='fx'."
                    )
                time_da, bnds_da = build_time(sdate, edate, "fx", ref_date, False)
            else:
                time_da, bnds_da = build_time(
                    sdate=cf.datetime(
                        ds["time"].values[0].year,
                        ds["time"].values[0].month,
                        ds["time"].values[0].day,
                        calendar=ds["time"].values[0].calendar,
                    ),
                    edate=cf.datetime(
                        ds["time"].values[-1].year,
                        ds["time"].values[-1].month,
                        ds["time"].values[-1].day,
                        calendar=ds["time"].values[-1].calendar,
                    ),
                    freq=bounds_frequency,
                    ref_date=ref_date,
                    climatology=False,
                )

        ds = ds.assign_coords({"time": time_da})
        ds = ds.assign(
            {"time_bnds": bnds_da} if not climatology else {"climatology_bnds": bnds_da}
        )

    # --- Encode bounds + time numerically while keeping attrs/encoding ---
    time_units = ds["time"].attrs["units"]
    time_cal = ds["time"].attrs.get("calendar", "gregorian")

    if climatology:
        ds["climatology_bnds"] = cf_to_num(
            ds["climatology_bnds"], time_units, time_cal, "float64"
        )
    else:
        ds["time_bnds"] = cf_to_num(ds["time_bnds"], time_units, time_cal, "float64")

    ds = ds.assign_coords(time=cf_to_num(ds["time"], time_units, time_cal, "float64"))
    return ds


def set_lat_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure the xarray dataset's latitude attributes are formatted according to CF-Conventions.
    """
    assert "lat" in ds
    da = ds["lat"]
    da.attrs = {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "Latitude",
    }
    da.encoding.clear()
    da.encoding = {"_FillValue": None, "dtype": "float64"}
    ds["lat"] = da
    return ds


def set_lon_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure the xarray dataset's longitude attributes are formatted according to CF-Conventions.
    """
    assert "lon" in ds
    da = ds["lon"]
    da.attrs = {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "long_name": "Longitude",
    }
    da.encoding.clear()
    da.encoding = {"_FillValue": None, "dtype": "float64"}
    ds["lon"] = da
    return ds


def set_depth_attrs(
    ds: xr.Dataset,
    bounds: np.ndarray,
    units: str,
    positive: str,
    long_name: str,
    *,
    standard_name: str = "depth",
    axis: str = "Z",
    depth_dim: str = "depth",
    bnds_dim: str = "bnds",
) -> xr.Dataset:
    """
    Ensure the xarray dataset's depth attributes are formatted according to CF-Conventions.
    bounds = 2D np.ndarray
    units = e.g., "meters"
    positive = direction; "down" or "up"
    long_name = e.g., "depth of sea water"
    """
    assert "depth" in ds

    # set up
    bounds_name = f"{depth_dim}_bnds"
    midpoints = bounds.mean(axis=1)
    ds = ds.assign_coords({depth_dim: midpoints})

    # update depth variable attrs and encoding
    depth_da = ds[depth_dim]
    depth_da.attrs = {
        "units": units,
        "positive": positive,
        "axis": axis,
        "standard_name": standard_name,
        "long_name": long_name,
        "bounds": bounds_name,
    }
    depth_da.encoding.clear()
    depth_da.encoding = {"_FillValue": None, "dtype": "float32"}
    ds[depth_dim] = depth_da

    # create depth bounds data array
    depth_bounds = xr.DataArray(
        data=bounds.astype(np.float32),
        dims=(depth_dim, bnds_dim),
        coords={depth_dim: ds[depth_dim]},
        name=bounds_name,
    )

    # assign bounds
    depth_bounds.attrs.clear()
    depth_bounds.encoding.clear()
    depth_bounds.encoding = {"_FillValue": None, "dtype": "float32"}
    ds[bounds_name] = depth_bounds

    return ds


def convert_units(
    da: xr.DataArray,
    target_units: str,
) -> xr.DataArray:
    """
    Convert a DataArray from its current units to target_units using cf_units.
    - Reads current_units = da.attrs['units']
    - Raises if no units or not convertible
    - Returns a new DataArray with converted data and units attr updated
    """
    current_units = da.attrs.get("units")
    if current_units is None:
        raise ValueError(f"Cannot convert: '{da.name}' has no 'units' attribute.")
    orig_u = Unit(current_units)
    target_u = Unit(target_units)
    if not orig_u.is_convertible(target_u):
        raise ValueError(
            f"Units '{current_units}' are not convertible to '{target_units}'."
        )

    # do the conversion
    new_vals = orig_u.convert(da.values, target_u)

    # build a new DataArray, preserving dims/coords/name and other attrs
    new_attrs = dict(da.attrs)
    new_attrs["units"] = target_units

    return xr.DataArray(
        data=new_vals,
        coords=da.coords,
        dims=da.dims,
        name=da.name,
        attrs=new_attrs,
        # encoding=da.encoding,  # preserve encoding (you may want to clear dtype and calendar)
    )


# default CF _FillValue options (see https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#classic_format_spec)
_FILL_VALUES = {
    np.dtype("S1"): np.bytes_(b"\x00"),  # char
    np.int8: np.int8(-127),  # byte
    np.int16: np.int16(-32767),  # short
    np.int32: np.int32(2147483647),  # int
    np.float32: np.float32(1.0e20),  # float
    np.float64: np.float64(1.0e20),  # double
}


def _sanitize_units(u: str) -> str:
    """
    Turn things like "gC/m^2/day" into "gC m-2 day-1",
    which cf_units will happily parse.
    """
    # 1) squared denominators: /foo^2 → " foo-2"
    u = re.sub(r"/([A-Za-z]+)\^?2", r" \1-2", u)
    # 2) any remaining single denominators: /foo → " foo-1"
    u = re.sub(r"/([A-Za-z]+)", r" \1-1", u)
    # 3) strip any stray carets
    return u.replace("^", "")


def set_var_attrs(
    ds: xr.Dataset,
    var: str,
    cmip6_units: str,
    cmip6_standard_name: str,
    cmip6_long_name: str,
    ancillary_variables: str | None = None,
    cell_methods: str | None = None,
    *,
    target_dtype: str | np.dtype | None = None,
    convert: bool = False,
) -> xr.Dataset:
    """
    Ensure ds[var] has CF-compliant attrs and correct _FillValue:
      - Optionally convert units
      - Strip any “[unit]” from the long_name
      - Set units, standard_name, long_name
      - Optionally cast to target_dtype
      - Pick and set the default _FillValue based on final dtype
      - For ints/bytes/shorts: replace existing missing_value markers
        with the CF fill and then write _FillValue into encoding
    """
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    da = ds[var]

    # capture existing missing_value marker
    mv = da.encoding.get("missing_value", da.attrs.get("missing_value", None))

    # optional unit conversion
    # ——————— normalize any “unit”/“units” attr ———————
    # look for a key that lowercased is “unit” or “units”
    units_key = next((k for k in da.attrs if k.lower() in ("unit", "units")), None)
    if units_key:
        # pop it out under the exact name “units”
        da.attrs["units"] = da.attrs.pop(units_key)

        # sanitize any funky notation so cf_units can parse it
        orig = da.attrs["units"]
        sanitized_unit = _sanitize_units(orig)
        if sanitized_unit != orig:
            da.attrs["units"] = sanitized_unit

    current_units = da.attrs.get("units", None)
    effective_units = cmip6_units

    # only convert if there *was* an original units and it differs
    if current_units is not None and current_units != cmip6_units:
        if convert:
            warnings.warn(
                f"Converting {var} units from {current_units} to {cmip6_units}"
            )
            da = convert_units(da, cmip6_units)
        else:
            warnings.warn(
                f"Variable '{var}' has units '{current_units}', "
                f"requested '{cmip6_units}'. Keeping existing units."
            )
            effective_units = current_units

    # remove "[unit]" from long name; could be confusing if it doesn't match actual unit
    clean_long_name = re.sub(r"\s*\[[^\]]+\]", "", cmip6_long_name).strip()

    # create CF attrs
    attrs = {
        "units": effective_units,
        "standard_name": cmip6_standard_name,
        "long_name": clean_long_name,
    }

    # assign ancillary variables attr if needed
    if ancillary_variables is not None:
        attrs["ancillary_variables"] = ancillary_variables
    if cell_methods is not None:
        attrs["cell_methods"] = cell_methods

    # assign the attrs
    da = da.assign_attrs(attrs)

    # set final dtype
    final_dt = np.dtype(target_dtype) if target_dtype is not None else da.dtype
    da = da.astype(final_dt)

    # determine final dtype and CF _FillValue
    final_dt = np.dtype(target_dtype) if target_dtype is not None else da.dtype
    fill = _FILL_VALUES.get(final_dt, None)
    if fill is None:
        # fallback by kind and size
        if final_dt.kind in ("i", "u"):
            fill = _FILL_VALUES.get(
                {1: np.int8, 2: np.int16, 4: np.int32}[final_dt.itemsize]
            )
        elif final_dt.kind == "f":
            fill = _FILL_VALUES.get({4: np.float32, 8: np.float64}[final_dt.itemsize])
        elif final_dt.kind == "S":
            fill = _FILL_VALUES[np.dtype("S1")]

    # handle data and encoding based on dtype
    if np.issubdtype(final_dt, np.floating):
        # Floats: leave NaNs, just set encoding
        da.encoding["_FillValue"] = fill
    else:
        # Ints/bytes: replace NaNs and old markers, cast to final_dtype
        if np.issubdtype(da.dtype, np.floating):
            da = da.fillna(fill)
        if mv is not None:
            da = da.where(da != mv, fill)
        da = da.astype(final_dt)
        da.encoding["_FillValue"] = fill

    # remove any old missing_value attributes; not required by CF anymore
    da.attrs.pop("missing_value", None)
    da.encoding.pop("missing_value", None)

    # reassign back to dataset
    ds[var] = da
    return ds


def set_coord_bounds(ds: xr.Dataset, coord: str) -> xr.Dataset:
    """
    Compute and attach 1D cell boundaries for `coord` using a 'bnds'=2
    dimension. Ensures float32 output, no _FillValue, and correct ordering.
    """
    # make sure we have a 'bnds' dimension of length 2
    if "nv" in ds.dims:
        ds = ds.rename_dims({"nv": "bnds"})

    # cast and grab values
    ds[coord] = ds[coord].astype(np.float64)
    vals = ds[coord].values
    n = vals.shape[0]

    # build midpoint array
    b = np.empty((n, 2), dtype=np.float64)
    # interior midpoints
    b[1:-1, 0] = 0.5 * (vals[:-2] + vals[1:-1])
    b[1:-1, 1] = 0.5 * (vals[1:-1] + vals[2:])
    # edge extrapolation
    d0 = vals[1] - vals[0]
    dn = vals[-1] - vals[-2]
    b[0] = [vals[0] - d0 / 2, vals[0] + d0 / 2]
    b[-1] = [vals[-1] - dn / 2, vals[-1] + dn / 2]

    # sort so column 0 is lower bound, column 1 is upper
    b = np.sort(b, axis=1)

    # assign into the dataset
    name = f"{coord}_bnds"
    ds[name] = ((coord, "bnds"), b)

    # reset encoding and attrs
    ds[name].encoding = {"_FillValue": None, "dtype": "float64"}
    ds[coord].attrs["bounds"] = name
    ds[coord].encoding = {"_FillValue": None, "dtype": "float64"}

    return ds


def gen_utc_timestamp(time: float | None = None) -> str:
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    else:
        time = datetime.datetime.fromtimestamp(time)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def gen_trackingid() -> str:
    return "hdl:21.14102/" + str(uuid.uuid4())


def standardize_dim_order(ds, order=("time", "depth", "lat", "lon", "bnds")):
    return ds.transpose(*order, missing_dims="ignore")


def set_cf_global_attributes(
    ds: xr.Dataset,
    *,  # keyword only for the following args
    title: str,
    institution: str,
    source: str,
    history: str,
    references: str,
    comment: str,
    conventions: str,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12.

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        title (str): Short description of the file contents.
        institution (str): Where the original data was produced.
        source (str): Method of production of the original data.
        history (str): List of applications that have modified the original data.
        references (str): References describing the data or methods used to produce it.
        comment (str): Miscellaneous information about the data or methods used.
        conventions (str): The name of the conventions followed by the dataset.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If a required global attribute is missing.
    """

    # Build and validate attributes
    attrs = {
        "title": title,
        "institution": institution,
        "source": source,
        "history": history,
        "references": references,
        "comment": comment,
        "Conventions": conventions,
    }

    # Ensure all values are explicitly set (None not allowed)
    missing = [k for k, v in attrs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    ds.attrs.update(attrs)
    return ds


def load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def get_nested_dict(data, path, default=None):
    for key in path:
        try:
            if isinstance(data, dict):
                data = data.get(key, default)
            elif isinstance(data, list) and isinstance(key, int):
                data = data[key]
            else:
                return default
        except (IndexError, TypeError):
            return default
    return data


def set_ods_global_attrs(
    ds: xr.Dataset,
    *,
    activity_id="obs4MIPs",
    aux_variable_id: Optional[str] = "N/A",
    comment: Optional[str] = "N/A",
    contact: str,
    conventions="CF-1.12 ODS-2.5",
    creation_date: str,
    dataset_contributor: str,
    data_specs_version: str,
    doi: str = "N/A",
    external_variables: Optional[str] = "N/A",
    frequency: str,
    grid: str,
    grid_label: str,
    has_auxdata: bool,
    history: str,
    institution: str,
    institution_id: str,
    license: str,
    nominal_resolution: str,
    processing_code_location: str,
    product: str,
    realm: str,
    references: str,
    region: str,
    source: str,
    source_id: str,
    source_data_retrieval_date: Optional[str] = "N/A",
    source_data_url: Optional[str] = "N/A",
    source_label: str,
    source_type: str,
    source_version_number: str,
    title: Optional[str] = "N/A",
    tracking_id: str,
    variable_id: str,
    variant_label: str,
    variant_info: str,
    version: Optional[str] = str,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12 and ODS-2.5.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to None.

    Special behavior:
        - Attributes with default values must match those defaults unless explicitly overridden.
        - Some attributes must be selected from a predefined list.
        - Some attributes are validated against controlled vocabularies loaded from the official online JSON files.
        - Nested vocabularies (e.g., obs4MIPs_CV.json) are accessed by key lookup (e.g., search "source_id").

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If any required attribute is missing or not valid.
    """

    valid_external_variables = ["areacella", "areacello", "volcella", "volcello"]

    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    freq_cv = load_json_from_url(base_url + "obs4MIPs_frequency.json")
    institution_cv = load_json_from_url(base_url + "obs4MIPs_institution_id.json")
    nominal_res_cv = load_json_from_url(base_url + "obs4MIPs_nominal_resolution.json")
    realm_cv = load_json_from_url(base_url + "obs4MIPs_realm.json")
    region_cv = load_json_from_url(base_url + "obs4MIPs_region.json")
    source_id_cv = load_json_from_url(base_url + "obs4MIPs_source_id.json")
    source_type_cv = load_json_from_url(base_url + "obs4MIPs_source_type.json")
    top_level_cv = load_json_from_url(base_url + "Tables/obs4MIPs_CV.json")
    grid_labels_cv = load_json_from_url(base_url + "obs4MIPs_grid_label.json")
    products_cv = load_json_from_url(base_url + "obs4MIPs_product.json")
    if frequency == "mon":
        mip_tables = [
            "Tables/obs4MIPs_Lmon.json",
            "Tables/obs4MIPs_Omon.json",
            "Tables/obs4MIPs_Amon.json",
        ]
        for table in mip_tables:
            if variable_id in get_nested_dict(
                load_json_from_url(base_url + table), ["variable_entry"]
            ):
                realm = get_nested_dict(
                    load_json_from_url(base_url + table), ["Header"]
                )["realm"]
                variable_cv = load_json_from_url(base_url + table)
                table_name = table.split("_")[-1].split(".")[0]

    errors = []
    if has_auxdata:
        if aux_variable_id == "None":
            errors.append("must specify ancillary variable_ids if included")

    # Check vals dependent on "valid_" lists hard-coded above
    if grid_label not in grid_labels_cv["grid_label"]:
        errors.append("grid_label must match a key in obs4MIPs_grid_label.json")
    if product not in products_cv["product"]:
        errors.append("product must match a key in obs4MIPs_product.json")
    # if external_variables:
    #     if external_variables not in valid_external_variables:
    #         errors.append(
    #             f"external_variables must be one of {valid_external_variables}"
    #         )
    # Check vals present in Github json files
    if frequency not in freq_cv["frequency"]:
        errors.append("frequency must match a key in obs4MIPs_frequency.json")
    # if institution_id not in institution_cv["institution_id"]:
    #     errors.append("institution_id must match a key in obs4MIPs_institution_id.json")
    # if nominal_resolution not in nominal_res_cv["nominal_resolution"]:
    #     errors.append(
    #         "nominal_resolution must match a key in obs4MIPs_nominal_resolution.json"
    #     )
    if realm not in realm_cv["realm"]:
        errors.append("realm must match a key in obs4MIPs_realm.json")
    if region not in region_cv["region"]:
        errors.append("region must match a key in obs4MIPs_region.json")
    # if source_id not in source_id_cv["source_id"]:
    #     errors.append("source_id must match a key in obs4MIPs_source_id.json")
    # if source_type not in source_type_cv["source_type"]:
    #     errors.append("source_type must match a key in obs4MIPs_source_type.json")
    # Check vals *nested* within Github json files
    # if source_label != get_nested_dict(
    #     source_id_cv, ["source_id", source_id, "source_label"]
    # ):
    #     errors.append(
    #         "source_label must match the label inside obs4MIPs_source_id.json[source_id]"
    #     )
    # if source_version_number not in get_nested_dict(
    #     source_id_cv, ["source_id", source_id, "source_version_number"]
    # ):
    #     errors.append("source_version_number must match a value inside source_id entry")
    # if source_id not in get_nested_dict(top_level_cv, ["CV", "source_id"]):
    #     errors.append(
    #         "source_id must be present in obs4MIPs_CV.json under 'source_id' key"
    #     )
    # if source not in get_nested_dict(
    #     top_level_cv, ["CV", "source_id", source_id, "source"]
    # ):
    #     errors.append(
    #         "source must match a attribute in the 'source_id' section of obs4MIPs_CV.json"
    #     )
    # if variable_id not in get_nested_dict(variable_cv, ["variable_entry"]):
    #     errors.append(
    #         f"variable_id must match a key in the 'variable_entry' section of {table_name}"
    #     )

    if errors:
        raise ValueError("\n".join(errors))

    attrs = {
        "activity_id": activity_id,
        "aux_variable_id": aux_variable_id,
        "comment": comment,
        "contact": contact,
        "Conventions": conventions,
        "creation_date": creation_date,
        "dataset_contributor": dataset_contributor,
        "data_specs_version": data_specs_version,
        "doi": doi,
        "frequency": frequency,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_frequency.json
        "grid": grid,
        "grid_label": grid_label,  # ["gn", "gr1"]
        "has_auxdata": has_auxdata,
        "history": history,
        "institution": institution,
        "institution_id": institution_id,  # have to be registered on https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json
        "license": license,
        "nominal_resolution": nominal_resolution,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_nominal_resolution.json
        "processing_code_location": processing_code_location,
        "product": product,  # [“observations”, “reanalysis”, “in_situ”, “exploratory_product”]
        "realm": realm,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_realm.json
        "references": references,
        "region": region,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_region.json
        "source": source,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/Tables/obs4MIPs_CV.json (search source_id)
        "source_data_retrieval_date": source_data_retrieval_date,
        "source_data_url": source_data_url,
        "source_id": source_id,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json
        "source_label": source_label,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json (nested source_label)
        "source_type": source_type,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_type.json
        "source_version_number": source_version_number,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json (nested source_version_number)
        "title": title,
        "tracking_id": tracking_id,  # automatically detected by CMOR (hdl:21.14102/<uuid>)
        "variable_id": variable_id,  # https://clipc-services.ceda.ac.uk/dreq/index/CMORvar.html
        "variant_info": variant_info,  # description of who prepared the data, describe obs data variance if applicable
        "variant_label": variant_label,  # same as source_id (if prepped by them), or "BE" if source_id unknown, ensemble member identified via -r1, -r2, ..., -rN
        "version": version,
    }

    missing = [k for k, v in attrs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    ds.attrs = attrs

    return ds


def set_ods26_global_attrs(
    ds: xr.Dataset,
    *,
    activity_id: str = "obs4MIPs",
    aux_uncertainty_id: str = "N/A",
    comment: Optional[str],
    contact: str = "N/A",  # First Last (email)
    Conventions: str = "CF-1.12 ODS-2.6",
    creation_date: str = "N/A",
    dataset_contributor: Optional[str],
    data_specs_version: str = "2.6",
    doi: Optional[str],
    frequency: str = "N/A",
    grid: str = "N/A",
    grid_label: str = "N/A",
    has_aux_unc: str = "FALSE",  # must be TRUE or FALSE
    history: Optional[str],
    institution: str = "N/A",
    institution_id: str = "N/A",
    license: str = "N/A",
    nominal_resolution: str = "N/A",
    processing_code_location: str = "N/A",
    product: str = "N/A",
    realm: str = "N/A",
    references: str = "N/A",
    region: str = "N/A",
    site_id: str = "N/A",
    site_location: str = "N/A",
    source: str = "N/A",
    source_data_retrieval_date: Optional[str],
    source_data_url: str = "N/A",
    source_id: str = "N/A",
    source_label: str = "N/A",
    source_type: str = "N/A",
    source_version_number: str = "N/A",
    table_id: str = "N/A",
    title: Optional[str],
    tracking_id: str = "N/A",
    variable_id: str = "N/A",
    variant_label: str = "N/A",
    variant_info: Optional[str],
    version: Optional[str],
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12 and ODS-2.6.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to None.
    """

    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    # Load controlled vocabularies from online JSON files
    # auxuncid_cv = load_json_from_url(base_url + "obs4MIPs_aux_uncertainty_id.json")
    freq_cv = load_json_from_url(base_url + "obs4MIPs_frequency.json")
    gridlabel_cv = load_json_from_url(base_url + "obs4MIPs_grid_label.json")
    # hasauxunc_cv = load_json_from_url(base_url + "obs4MIPs_has_aux_unc.json")
    instid_cv = load_json_from_url(base_url + "obs4MIPs_institution_id.json")
    license_cv = load_json_from_url(base_url + "obs4MIPs_license.json")
    nomres_cv = load_json_from_url(base_url + "obs4MIPs_nominal_resolution.json")
    product_cv = load_json_from_url(base_url + "obs4MIPs_product.json")
    realm_cv = load_json_from_url(base_url + "obs4MIPs_realm.json")
    region_cv = load_json_from_url(base_url + "obs4MIPs_region.json")
    reqattrs_cv = load_json_from_url(
        base_url + "obs4MIPs_required_global_attributes.json"
    )
    siteid_cv = load_json_from_url(base_url + "obs4MIPs_site_id.json")
    sourceid_cv = load_json_from_url(base_url + "obs4MIPs_source_id.json")
    sourcetype_cv = load_json_from_url(base_url + "obs4MIPs_source_type.json")
    tableid_cv = load_json_from_url(base_url + "obs4MIPs_table_id.json")

    # Fill in required global attributes
    attrs = {
        "activity_id": activity_id,
        "aux_uncertainty_id": aux_uncertainty_id,
        "contact": contact,
        "Conventions": Conventions,
        "creation_date": creation_date,
        "data_specs_version": data_specs_version,
        "frequency": frequency,
        "grid": grid,
        "grid_label": grid_label,
        "has_aux_unc": has_aux_unc,
        "institution": institution,
        "institution_id": institution_id,
        "license": license,
        "nominal_resolution": nominal_resolution,
        "processing_code_location": processing_code_location,
        "product": product,
        "realm": realm,
        "references": references,
        "region": region,
        "site_id": site_id,
        "site_location": site_location,
        "source": source,
        "source_data_url": source_data_url,
        "source_id": source_id,
        "source_label": source_label,
        "source_type": source_type,
        "source_version_number": source_version_number,
        "table_id": table_id,
        "tracking_id": tracking_id,
        "variable_id": variable_id,
        "variant_label": variant_label,
    }

    # Add optional attributes if provided
    optional_attrs = {
        "comment": comment,
        "dataset_contributor": dataset_contributor,
        "doi": doi,
        "history": history,
        "source_data_retrieval_date": source_data_retrieval_date,
        "title": title,
        "variant_info": variant_info,
        "version": version,
    }
    for key, value in optional_attrs.items():
        if value is not None:
            attrs[key] = value

    # Sort the keys alphabetically
    attrs = dict(sorted(attrs.items()))

    # Validate required attributes
    missing = []
    for attr in reqattrs_cv["required_global_attributes"]:
        if attr not in attrs or attrs[attr] is None:
            missing.append(attr)
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    # Validate controlled vocabularies
    cv_checks = {
        # "aux_uncertainty_id": auxuncid_cv["aux_uncertainty_id"],
        "frequency": freq_cv["frequency"],
        "grid_label": gridlabel_cv["grid_label"],
        # "has_aux_unc": hasauxunc_cv["has_aux_unc"],
        "institution_id": instid_cv["institution_id"],
        "nominal_resolution": nomres_cv["nominal_resolution"],
        "product": product_cv["product"],
        "realm": realm_cv["realm"],
        "region": region_cv["region"],
        "site_id": siteid_cv["site_id"],
        "source_id": sourceid_cv["source_id"],
        "source_type": sourcetype_cv["source_type"],
    }
    for attr, cv_obj in cv_checks.items():
        if attrs[attr] not in cv_obj:
            # Warn but don't fail
            warnings.warn(
                f"Attribute '{attr}' has value '{attrs[attr]}' "
                f"which is not in the controlled vocabulary."
            )
            continue

    # set global attributes
    ds.attrs = attrs
    return ds


def set_ods_var_attrs(ds: xr.Dataset, variable_id: str) -> xr.Dataset:
    """
    Set required NetCDF variable level attributes according to CF-Conventions 1.12 and ODS-2.5.

    This function validates that all required attributes are provided. Optional fields may be set to None.

    Special behavior:
        - Attributes with default values must match those defaults unless explicitly overridden.
        - Some attributes must be selected from a predefined list.
        - Some attributes are validated against controlled vocabularies loaded from the official online JSON files.
        - Nested vocabularies (e.g., obs4MIPs_CV.json) are accessed by key lookup (e.g., search "source_id").

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If any required attribute is missing or not valid.
    """
    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    if ds.attrs["frequency"] == "mon":
        if ds.attrs["realm"] == "atmos":
            mip_table = "Tables/obs4MIPs_Amon.json"
        elif ds.attrs["realm"] == "land":
            mip_table = "Tables/obs4MIPs_Lmon.json"
        elif ds.attrs["realm"] == "ocean":
            mip_table = "Tables/obs4MIPs_Omon.json"
        varattrs = get_nested_dict(
            load_json_from_url(base_url + mip_table), ["variable_entry"]
        )[ds.attrs["variable_id"]]
        if Unit(varattrs["units"]) != Unit(ds[variable_id].attrs["units"]):
            ds[variable_id].values = Unit(ds[variable_id].attrs["units"]).convert(
                ds[variable_id].values,
                Unit(varattrs["units"]),
                inplace=True,
            )
            ds[variable_id].attrs["history"] = (
                f"{gen_utc_timestamp()} altered by ILAMB: Converted units from '{ds[variable_id].attrs['units']}' to '{varattrs['units']}'."
            )
            ds[variable_id].attrs["original_units"] = ds[variable_id].attrs["units"]
            ds[variable_id].attrs["units"] = varattrs["units"]
        else:
            ds[variable_id].attrs["units"] = varattrs["units"]
        if varattrs["positive"]:
            ds[variable_id].attrs["positive"] = varattrs["positive"]
        print(ds[variable_id].attrs)
        ds[variable_id].attrs.update(
            {
                key: varattrs[key]
                for key in [
                    "standard_name",
                    "long_name",
                    "comment",
                    "cell_methods",
                    "cell_measures",
                ]
            }
        )
        print(ds[variable_id].attrs)
    return ds


def set_ods_calendar(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.convert_calendar("gregorian")
    return ds


def set_ods_coords(ds: xr.Dataset) -> xr.Dataset:
    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    possible_bounds = ["bounds", "lat_bounds", "lon_bounds", "time_bounds"]
    replaced_bounds = ["bnds", "lat_bnds", "lon_bnds", "time_bnds"]
    for bound, rbound in zip(possible_bounds, replaced_bounds):
        if bound in ds:
            ds = ds.rename({bound: rbound})
            if "_" in bound:
                coord = bound.split("_")[0]
                ds[coord].attrs.update({"bounds": rbound})
    coord_table = load_json_from_url(base_url + "Tables/obs4MIPs_coordinate.json")[
        "axis_entry"
    ]

    def find_coord_key(nested_json, coord):
        if isinstance(nested_json, dict):
            for key, value in nested_json.items():
                if isinstance(value, dict) and value.get("out_name") == coord:
                    return key
                result = find_coord_key(value, coord)
                if result:
                    return result
        elif isinstance(nested_json, list):
            for item in nested_json:
                result = find_coord_key(item, coord)
                if result:
                    return result
        return None

    for coord in ds.coords:
        key = find_coord_key(coord_table, coord)
        if key and key != "time":
            ds[coord].attrs.update(
                {
                    k: coord_table[key][k]
                    for k in ["units", "axis", "long_name", "standard_name"]
                }
            )
    return ds
