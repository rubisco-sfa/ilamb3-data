import cftime as cf
import numpy as np
import pandas as pd
import xarray as xr


def create_time_bounds(freq: str, sdate: cf.datetime, edate: cf.datetime) -> np.ndarray:
    """
    Create an ndarray of CF-style time bounds from a frequency and date extent.

    Returns an object array with shape ``(ntime, 2)`` whose intervals cover every period
    containing a date from ``sdate`` through ``edate``.

    Parameters
    ----------
    freq : str
        Frequency string, e.g., "M" for monthly, "Q" for quarterly, "A" for annual, etc.
        See pandas documentation for valid frequency strings.
    sdate : cf.datetime
        Start date of the time bounds.
    edate : cf.datetime
        End date of the time bounds.

    Returns
    -------
    np.ndarray
        An array of CF-style time bounds for the given frequency and date extent.
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
    ends = periods.end_time + pd.Timedelta(nanoseconds=1)

    # pandas Timestamp -> cftime.DatetimeGregorian
    def ts_to_cf(ts):
        return cf.DatetimeGregorian(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond
        )

    bnds = np.array(
        [(ts_to_cf(lo), ts_to_cf(hi)) for lo, hi in zip(starts, ends)], dtype=object
    )
    return bnds


def create_climatology_bounds(
    freq: str,
    sdate: cf.datetime,
    edate: cf.datetime,
) -> np.ndarray:
    """
    Create an ndarray of CF-style bounds for climatological periods over a date range.

    ``freq`` determines the intervals represented by the bounds. Lower bounds use the
    year of ``sdate``, and upper bounds use the year of ``edate``. If a period wraps
    into the next calendar year, the upper bound will use ``edate.year + 1``.

    Parameters
    ----------
    freq : str
        Frequency string, e.g., "M" for monthly, "Q" for quarterly, "A" for annual, etc.
        See pandas documentation for valid frequency strings.
    sdate : cf.datetime
        Start date of the climatology period.
    edate : cf.datetime
        End date of the climatology period.

    Returns
    -------
    np.ndarray
        An array of CF-style bounds for the climatological periods.
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
    ends = periods.end_time + pd.Timedelta(nanoseconds=1)

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


def add_rectilinear_bounds(
    ds: xr.Dataset,
    *,
    latitude: str = "lat",
    longitude: str = "lon",
    bounds_dim: str = "bnds",
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Compute and attach bounds to xr.Dataset for rectilinear lat/lon coordinates.

    Bounds are placed halfway between adjacent coordinate values. The outer bounds are
    estimated using the spacing between the two nearest values.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the latitude and longitude coordinates.
    latitude : str, default: "lat"
        Name of the 1D latitude coordinate.
    longitude : str, default: "lon"
        Name of the 1D longitude coordinate.
    bounds_dim : str, default: "bnds"
        Name of the trailing, size-two vertex dimension. The dimension is
        created when absent and reused when present.
    overwrite : bool, default: False
        Replace bounds already identified by a coordinate's ``bounds``
        attribute. By default, valid existing bounds are preserved.

    Returns
    -------
    xr.Dataset
        The dataset with ``<latitude>_bnds`` and ``<longitude>_bnds`` added.

    Notes
    -----
    This function does not support curvilinear or unstructured grids.
    """
    # Validate inputs
    if latitude == longitude:
        raise ValueError("latitude and longitude must name different coordinates")
    if not isinstance(bounds_dim, str) or not bounds_dim:
        raise ValueError("bounds_dim must be a non-empty string")
    if bounds_dim in ds.sizes and ds.sizes[bounds_dim] != 2:
        raise ValueError(
            f"Bounds dimension {bounds_dim!r} must have size 2, "
            f"not {ds.sizes[bounds_dim]}"
        )

    # Create a "plan" for building bounds for each coordinate
    plans = []
    for coord in (latitude, longitude):
        # Validate coordinate shape, type, and monotonicity
        if coord not in ds.coords:
            raise KeyError(f"Coordinate {coord!r} not found in dataset")
        if ds[coord].ndim != 1:
            raise ValueError(f"Coordinate {coord!r} must be one-dimensional")
        if ds[coord].size < 2:
            raise ValueError(f"Coordinate {coord!r} must contain at least two values")
        if not np.issubdtype(ds[coord].dtype, np.number):
            raise TypeError(f"Coordinate {coord!r} must be numeric")
        coord_dim = ds[coord].dims[0]
        if coord_dim == bounds_dim:
            raise ValueError(
                f"bounds_dim {bounds_dim!r} cannot also be the dimension of "
                f"coordinate {coord!r}"
            )
        vals = np.asarray(ds[coord].values, dtype=np.float64)
        if not np.isfinite(vals).all():
            raise ValueError(f"Coordinate {coord!r} must contain only finite values")
        deltas = np.diff(vals)
        if not (np.all(deltas > 0) or np.all(deltas < 0)):
            raise ValueError(f"Coordinate {coord!r} must be strictly monotonic")

        # Validate existing bounds if present
        existing_name = ds[coord].attrs.get("bounds")
        if existing_name is not None and not isinstance(existing_name, str):
            raise TypeError(
                f"The bounds attribute of coordinate {coord!r} must be a string"
            )

        # Check for existing bounds and validity
        if existing_name and not overwrite:
            if existing_name not in ds.variables:
                raise ValueError(
                    f"Coordinate {coord!r} references missing bounds variable "
                    f"{existing_name!r}"
                )
            existing = ds[existing_name]
            expected_leading_dims = ds[coord].dims
            if (
                existing.ndim != ds[coord].ndim + 1
                or existing.dims[:-1] != expected_leading_dims
                or existing.sizes[existing.dims[-1]] != 2
                or not np.issubdtype(existing.dtype, np.number)
                or not np.isfinite(existing.values).all()
            ):
                raise ValueError(
                    f"Bounds variable {existing_name!r} for coordinate {coord!r} "
                    "must be numeric with the coordinate dimensions followed by "
                    "a size-two vertex dimension"
                )
            continue

        # Don't overwrite existing bounds if they are already valid
        bounds_name = existing_name or f"{coord}_bnds"
        if bounds_name in ds.variables and not overwrite:
            raise ValueError(
                f"Bounds variable {bounds_name!r} already exists but is not "
                f"identified by coordinate {coord!r}; use overwrite=True to replace it"
            )

        n = vals.shape[0]

        # Build interior bounds from adjacent centers and extrapolate the edges
        bnds = np.empty((n, 2), dtype=np.float64)
        bnds[1:-1, 0] = 0.5 * (vals[:-2] + vals[1:-1])
        bnds[1:-1, 1] = 0.5 * (vals[1:-1] + vals[2:])
        bnds[0] = [
            vals[0] - (vals[1] - vals[0]) / 2,
            0.5 * (vals[0] + vals[1]),
        ]
        bnds[-1] = [
            0.5 * (vals[-2] + vals[-1]),
            vals[-1] + (vals[-1] - vals[-2]) / 2,
        ]
        bnds = np.sort(bnds, axis=1)
        plans.append((coord, coord_dim, bounds_name, bnds))

    # Apply changes only after both coordinates and all existing bounds validate
    out = ds.copy()
    for coord in (latitude, longitude):
        out[coord] = out[coord].astype(np.float64)
        out[coord].encoding = {"_FillValue": None, "dtype": "float64"}
    for coord, coord_dim, bounds_name, bnds in plans:
        out[bounds_name] = ((coord_dim, bounds_dim), bnds)
        out[bounds_name].encoding = {"_FillValue": None, "dtype": "float64"}
        out[coord].attrs["bounds"] = bounds_name

    return out
