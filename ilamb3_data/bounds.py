from datetime import timedelta

import cftime as cf
import numpy as np
import xarray as xr


def _period_boundaries(
    freq: str,
    sdate: cf.datetime,
    edate: cf.datetime,
    calendar: str,
) -> list[cf.datetime]:
    """Return calendar-aware boundaries for the periods containing two dates."""
    frequency = str(freq)
    if frequency not in ("D", "MS", "QS-JAN", "YS"):
        raise ValueError("freq must be one of 'D', 'MS', 'QS-JAN', or 'YS'")

    start_fields = (
        sdate.year,
        sdate.month,
        sdate.day,
        sdate.hour,
        sdate.minute,
        sdate.second,
        sdate.microsecond,
    )
    end_fields = (
        edate.year,
        edate.month,
        edate.day,
        edate.hour,
        edate.minute,
        edate.second,
        edate.microsecond,
    )
    if end_fields < start_fields:
        raise ValueError("edate must be >= sdate.")

    if frequency == "D":
        try:
            lower = cf.datetime(
                sdate.year, sdate.month, sdate.day, calendar=calendar
            )
            upper = cf.datetime(
                edate.year, edate.month, edate.day, calendar=calendar
            ) + timedelta(days=1)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"The daily extent cannot be represented in calendar {calendar!r}"
            ) from exc
    elif frequency == "MS":
        lower = cf.datetime(sdate.year, sdate.month, 1, calendar=calendar)
        year = edate.year + (1 if edate.month == 12 else 0)
        month = 1 if edate.month == 12 else edate.month + 1
        upper = cf.datetime(year, month, 1, calendar=calendar)
    elif frequency == "QS-JAN":
        month = 3 * ((sdate.month - 1) // 3) + 1
        lower = cf.datetime(sdate.year, month, 1, calendar=calendar)
        month = 3 * ((edate.month - 1) // 3) + 4
        year = edate.year
        if month > 12:
            month -= 12
            year += 1
        upper = cf.datetime(year, month, 1, calendar=calendar)
    else:
        lower = cf.datetime(sdate.year, 1, 1, calendar=calendar)
        upper = cf.datetime(edate.year + 1, 1, 1, calendar=calendar)

    return list(
        xr.date_range(
            start=lower,
            end=upper,
            freq=frequency,
            calendar=calendar,
            use_cftime=True,
        )
    )


def create_time_bounds(
    freq: str,
    sdate: cf.datetime,
    edate: cf.datetime,
    calendar: str = "standard",
) -> np.ndarray:
    """
    Create an ndarray of CF-style time bounds from a frequency and date extent.

    Returns an object array with shape ``(ntime, 2)`` whose intervals cover every period
    containing a date from ``sdate`` through ``edate``.

    Parameters
    ----------
    freq : {"D", "MS", "QS-JAN", "YS"}
        Daily, monthly, quarterly, or yearly interval frequency.
    sdate : cf.datetime
        Start date of the time bounds.
    edate : cf.datetime
        End date of the time bounds.
    calendar : str, default: "standard"
        CF calendar used for the returned bounds.

    Returns
    -------
    np.ndarray
        An array of CF-style time bounds for the given frequency and date extent.
    """
    boundaries = _period_boundaries(freq, sdate, edate, calendar)
    return np.array(list(zip(boundaries[:-1], boundaries[1:])), dtype=object)


def create_climatology_bounds(
    freq: str,
    sdate: cf.datetime,
    edate: cf.datetime,
    calendar: str = "standard",
) -> np.ndarray:
    """
    Create an ndarray of CF-style bounds for climatological periods over a date range.

    ``freq`` determines the intervals represented by the bounds. Lower bounds use the
    year of ``sdate``, and upper bounds use the year of ``edate``. If a period wraps
    into the next calendar year, the upper bound will use ``edate.year + 1``.

    Parameters
    ----------
    freq : {"D", "MS", "QS-JAN", "YS"}
        Daily, monthly, quarterly, or yearly interval frequency.
    sdate : cf.datetime
        Start date of the climatology period.
    edate : cf.datetime
        End date of the climatology period.
    calendar : str, default: "standard"
        CF calendar used for the returned bounds.

    Returns
    -------
    np.ndarray
        An array of CF-style bounds for the climatological periods.
    """
    if (
        edate.year,
        edate.month,
        edate.day,
        edate.hour,
        edate.minute,
        edate.second,
        edate.microsecond,
    ) < (
        sdate.year,
        sdate.month,
        sdate.day,
        sdate.hour,
        sdate.minute,
        sdate.second,
        sdate.microsecond,
    ):
        raise ValueError("edate must be >= sdate.")

    template_year = sdate.year
    if freq == "D" and calendar.lower() in {
        "standard",
        "gregorian",
        "proleptic_gregorian",
        "julian",
    }:
        while cf.is_leap_year(template_year, calendar=calendar):
            template_year += 1
    first = cf.datetime(template_year, 1, 1, calendar=calendar)
    last = cf.datetime(template_year + 1, 1, 1, calendar=calendar) - timedelta(days=1)
    template = create_time_bounds(freq, first, last, calendar)

    bounds = []
    for lower, upper in template:
        bounds.append(
            (
                cf.datetime(
                    sdate.year,
                    lower.month,
                    lower.day,
                    lower.hour,
                    lower.minute,
                    lower.second,
                    lower.microsecond,
                    calendar=calendar,
                ),
                cf.datetime(
                    edate.year + (upper.year - lower.year),
                    upper.month,
                    upper.day,
                    upper.hour,
                    upper.minute,
                    upper.second,
                    upper.microsecond,
                    calendar=calendar,
                ),
            )
        )
    return np.array(bounds, dtype=object)


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
