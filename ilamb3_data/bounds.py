import cftime as cf
import numpy as np
import pandas as pd
import xarray as xr


def build_time_from_frequency(
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
        # Datetime offsets call month-end "ME" while PeriodIndex still uses "M".
        offset = pd.tseries.frequencies.to_offset("ME" if freq == "M" else freq)
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


def build_climatology_from_frequency(
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
        _ = pd.tseries.frequencies.to_offset("ME" if freq == "M" else freq)
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


def build_from_centers(ds: xr.Dataset, coord: str) -> xr.Dataset:
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
