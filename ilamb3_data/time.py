import cftime as cf
import numpy as np
import pandas as pd
import xarray as xr

from .bounds import create_climatology_bounds, create_time_bounds


def _as_cftime(value, calendar: str) -> cf.datetime:
    """Convert a datetime-like value to cftime.datetime with the specified calendar."""
    if isinstance(value, np.datetime64):
        if np.isnat(value):
            raise ValueError("Time coordinates cannot contain missing dates")
        value = pd.Timestamp(value)
    try:
        return cf.datetime(
            value.year,
            value.month,
            value.day,
            getattr(value, "hour", 0),  # get value or default to 0 if it doesn't exist
            getattr(value, "minute", 0),
            getattr(value, "second", 0),
            getattr(value, "microsecond", 0),
            calendar=calendar,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        raise ValueError(
            f"Date {value!r} cannot be represented in calendar {calendar!r}"
        ) from exc


def _read_time(ds: xr.Dataset) -> list[cf.datetime]:
    """Read a one-dimensional time coordinate as cftime values."""

    # Check that the dataset has a valid time coordinate called "time"
    if "time" not in ds.coords:
        raise ValueError(
            "Dataset has no coordinate named 'time.' Rename existing time axis to "
            "'time' before calling this function."
        )
    time = ds["time"]
    if time.dims != ("time",):
        raise ValueError("The 'time' coordinate must be one-dimensional")
    if time.size == 0:
        raise ValueError("The 'time' coordinate cannot be empty")

    # Get calendar, assume "standard" (according to CF conventions) if not specified
    calendar = time.attrs.get("calendar") or time.encoding.get("calendar") or "standard"

    # If the time values are numeric, convert to cftime using units and calendar attrs
    values = np.asarray(time.values)
    if np.issubdtype(values.dtype, np.number):
        units = time.attrs.get("units") or time.encoding.get("units")
        if units is None:
            raise ValueError("Numeric time coordinates must define CF time units")
        return list(
            np.asarray(
                cf.num2date(
                    values,
                    units=units,
                    calendar=calendar,
                    only_use_cftime_datetimes=True,
                ),
                dtype=object,
            )
        )

    # If the time values are already datetime-like, convert to cftime given calendar
    if np.issubdtype(values.dtype, np.datetime64):
        return [_as_cftime(value, calendar) for value in values]

    # If the time values are already cftime.datetime, ensure calendar consistency
    first_calendar = getattr(values[0], "calendar", calendar)
    dates = [_as_cftime(value, first_calendar) for value in values]
    if any(date.calendar != dates[0].calendar for date in dates[1:]):
        raise ValueError("All time coordinate values must use the same calendar")
    return dates


def _assign_time_axis(
    ds: xr.Dataset,
    centers: list[cf.datetime],
    bounds: np.ndarray,
    *,
    calendar: str,
    ref_date: cf.datetime | None,
    bounds_dim: str,
    climatology: bool,
    add_bounds: bool = True,
) -> xr.Dataset:
    """Attach a prepared time coordinate to a xarray Dataset and its optional bounds."""

    # Validate inputs
    if bounds.shape != (len(centers), 2):
        raise ValueError("Time centers and bounds must have matching lengths")
    if "time" in ds.sizes and ds.sizes["time"] != len(centers):
        raise ValueError(
            f"Generated {len(centers)} time values, but the dataset's time "
            f"dimension has length {ds.sizes['time']}"
        )
    if not isinstance(bounds_dim, str) or not bounds_dim:
        raise ValueError("bounds_dim must be a non-empty string")
    if bounds_dim == "time":
        raise ValueError("bounds_dim cannot be 'time'")
    if add_bounds and bounds_dim in ds.sizes and ds.sizes[bounds_dim] != 2:
        raise ValueError(
            f"Bounds dimension {bounds_dim!r} must have size 2, "
            f"not {ds.sizes[bounds_dim]}"
        )

    # Use the first lower bound as the reference date if not provided
    reference = bounds[0, 0] if ref_date is None else _as_cftime(ref_date, calendar)
    ref_text = (
        f"{reference.year:04d}-{reference.month:02d}-{reference.day:02d} "
        f"{reference.hour:02d}:{reference.minute:02d}:{reference.second:02d}"
    )
    if reference.microsecond:
        ref_text += f".{reference.microsecond:06d}"
    units = f"days since {ref_text}"

    # Get existing bounds/climatology variable names
    old_bounds = []
    if "time" in ds.coords:
        for attr in ("bounds", "climatology"):
            name = ds["time"].attrs.get(attr)
            if isinstance(name, str) and name in ds.variables:
                old_bounds.append(name)

    # Drop existing bounds/climatology variables and assign new time coordinate
    out = ds.drop_vars(old_bounds, errors="ignore")
    out = out.assign_coords(time=("time", np.asarray(centers, dtype=object)))
    out["time"].attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
    }

    # Add time or climatology bounds as attributes to the time coordinate
    bounds_name = "climatology_bnds" if climatology else "time_bnds"
    if add_bounds:
        out["time"].attrs["climatology" if climatology else "bounds"] = bounds_name

    # Set encoding for time and bounds variables
    out["time"].encoding = {
        "units": units,
        "calendar": calendar,
        "dtype": "float64",  # obs4MIPs uses float64 for time
        "_FillValue": None,
    }

    # Add the bounds variable (if requested) with appropriate encoding
    if add_bounds:
        out[bounds_name] = (("time", bounds_dim), bounds)
        out[bounds_name].encoding = {
            "units": units,
            "calendar": calendar,
            "dtype": "float64",
            "_FillValue": None,
        }
    return out


def create_time_axis(
    ds: xr.Dataset,
    sdate: cf.datetime,
    edate: cf.datetime,
    frequency: str,
    *,
    calendar: str = "standard",
    ref_date: cf.datetime | None = None,
    add_bounds: bool = True,
    bounds_dim: str = "bnds",
) -> xr.Dataset:
    """
    Create and attach a time coordinate to an xarray Dataset using a start date, end
    date, and frequency. Optionally define a calendar, reference date for the time
    units, and whether to add bounds. The bounds dimension name to look for or create
    can also be specified.

    Time values are set to the exact midpoint of each interval. By default, bounds are
    attached, and the first lower bound is used as the reference date. Drops existing
    time coordinate and bounds/climatology variables if they exist.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to which the time coordinate will be attached.
    sdate : cf.datetime
        The start date of the time axis.
    edate : cf.datetime
        The end date of the time axis.
    frequency : str
        The frequency of the time axis; only "D", "MS", "QS-JAN", "YS", or "fx" are
        supported. "fx" is a special case for fixed time intervals, where the start and
        end dates define a single interval.
    calendar : str, optional
        The calendar to use for the time axis. Default is "standard".
    ref_date : cf.datetime, optional
        The reference date for the time units. If not provided, the first lower bound
        is used.
    add_bounds : bool, optional
        Whether to add bounds to the time coordinate. Default is True.
    bounds_dim : str, optional
        The name of the bounds dimension. Default is "bnds".

    Returns
    -------
    xr.Dataset
        The dataset with the new time coordinate and optional bounds attached.
    """
    if frequency == "fx":
        lower = _as_cftime(sdate, calendar)
        upper = _as_cftime(edate, calendar)
        if upper < lower:
            raise ValueError("edate must be >= sdate")
        bounds = np.array([(lower, upper)], dtype=object)
    else:
        bounds = create_time_bounds(frequency, sdate, edate, calendar)
    centers = [lower + (upper - lower) / 2 for lower, upper in bounds]
    return _assign_time_axis(
        ds,
        centers,
        bounds,
        calendar=calendar,
        ref_date=ref_date,
        bounds_dim=bounds_dim,
        climatology=False,
        add_bounds=add_bounds,
    )


def standardize(
    ds: xr.Dataset,
    bounds_frequency: str,
    *,
    calendar: str = "standard",
    ref_date: cf.datetime | None = None,
    bounds_dim: str = "bnds",
) -> xr.Dataset:
    """
    Standardize an xarray Dataset's existing time coordinate and add interval bounds.

    Time values are set to the exact midpoint of each interval if they are not already.
    The bounds frequency determines the length of each interval.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with an existing time coordinate to standardize.
    bounds_frequency : str
        The frequency of the time bounds; only "D", "MS", "QS-JAN", "YS", are supported.
        If the time coordinate is fixed ("fx"), use create_time_axis() instead.
    calendar : str, optional
        The calendar to use for the time axis. Default is "standard".
    ref_date : cf.datetime, optional
        The reference date for the time units. If not provided, the first lower bound
        is used.
    bounds_dim : str, optional
        The name of the bounds dimension. Default is "bnds".

    Returns
    -------
    xr.Dataset
        The dataset with the new standardized time coordinate and bounds attached.
    """

    # If fixed time interval, sdate and edate are required to define the bounds
    # So recommend using create_time_axis() instead of standardize() for fixed intervals
    if bounds_frequency == "fx":
        raise ValueError("Use create_time_axis() to define a fixed time interval")

    # Read the time coordinate and validate that it contains one value per interval
    dates = _read_time(ds)
    source_calendar = dates[0].calendar

    # Validate the existing time coordinates and current calendar
    source_bounds = create_time_bounds(
        bounds_frequency, dates[0], dates[-1], source_calendar
    )
    if len(source_bounds) != len(dates) or any(
        not (lower <= date < upper)
        for date, (lower, upper) in zip(dates, source_bounds)
    ):
        raise ValueError(
            "The time coordinate must contain one ordered value per "
            f"{bounds_frequency!r} interval"
        )

    # Convert to new calendar if necessary, and compute the new centers and bounds
    bounds = source_bounds
    if calendar != source_calendar:
        bounds = create_time_bounds(bounds_frequency, dates[0], dates[-1], calendar)
    centers = [lower + (upper - lower) / 2 for lower, upper in bounds]
    return _assign_time_axis(
        ds,
        centers,
        bounds,
        calendar=calendar,
        ref_date=ref_date,
        bounds_dim=bounds_dim,
        climatology=False,
    )


def standardize_climatology(
    ds: xr.Dataset,
    bounds_frequency: str,
    sdate: cf.datetime,
    edate: cf.datetime,
    *,
    calendar: str = "standard",
    ref_date: cf.datetime | None = None,
    bounds_dim: str = "bnds",
) -> xr.Dataset:
    """
    Standardize a climatological time coordinate and add climatology bounds.

    ``sdate`` and ``edate`` describe the period used to calculate the climatology. Time
    values are representative exact midpoints of the first occurrence of each
    climatological interval.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset with an existing time coordinate to standardize.
    bounds_frequency : str
        The frequency of the time bounds; only "D", "MS", "QS-JAN", "YS", and "fx" are
        supported. Use "fx" for fixed climatology intervals, where ``sdate`` and
        ``edate`` define a single interval.
    sdate : cf.datetime
        The start date of the climatology interval.
    edate : cf.datetime
        The end date of the climatology interval.
    calendar : str, optional
        The calendar to use for the time axis. Default is "standard".
    ref_date : cf.datetime, optional
        The reference date for the time units. If not provided, the first lower bound
        is used.
    bounds_dim : str, optional
        The name of the bounds dimension. Default is "bnds".

    Returns
    -------
    xr.Dataset
        The dataset with the new climatological time coordinate and bounds attached.
    """
    _read_time(ds)
    bounds = create_climatology_bounds(bounds_frequency, sdate, edate, calendar)

    centers = []
    for lower, upper in bounds:
        lower_fields = (
            lower.month,
            lower.day,
            lower.hour,
            lower.minute,
            lower.second,
            lower.microsecond,
        )
        upper_fields = (
            upper.month,
            upper.day,
            upper.hour,
            upper.minute,
            upper.second,
            upper.microsecond,
        )
        upper_year = lower.year + (1 if upper_fields <= lower_fields else 0)
        representative_upper = cf.datetime(
            upper_year,
            upper.month,
            upper.day,
            upper.hour,
            upper.minute,
            upper.second,
            upper.microsecond,
            calendar=calendar,
        )
        centers.append(lower + (representative_upper - lower) / 2)

    return _assign_time_axis(
        ds,
        centers,
        bounds,
        calendar=calendar,
        ref_date=ref_date,
        bounds_dim=bounds_dim,
        climatology=True,
    )
