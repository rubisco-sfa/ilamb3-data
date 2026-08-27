import warnings

import cftime as cf
import numpy as np
import pandas as pd
import xarray as xr

from .bounds import create_climatology_bounds, create_time_bounds


def build(
    sdate: cf.datetime,
    edate: cf.datetime,
    freq: str,
    ref_date: cf.datetime | None = None,
    climatology: bool = False,
) -> tuple[xr.DataArray, xr.DataArray]:
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
            bnds = create_climatology_bounds(freq, sdate, edate)
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
            bnds = create_time_bounds(freq, sdate, edate)
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


def to_numeric(
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


def standardize(
    ds: xr.Dataset,
    bounds_frequency: str,
    ref_date: cf.datetime | None = None,
    create_new_time: bool = False,
    sdate: cf.datetime | None = None,
    edate: cf.datetime | None = None,
    climatology: bool = False,
    clim_sdate: cf.datetime | None = None,
    clim_edate: cf.datetime | None = None,
) -> xr.Dataset:
    """
    Set time-related attributes for an xarray Dataset, including time bounds.

    Parameters
    ----------
    ds : xr.Dataset
        The dataset to modify.
    bounds_frequency : str
        Frequency of the time bounds (e.g., "M" for monthly, "fx" for fixed).
    ref_date : cf.datetime | None, optional
        Reference date for CF time conversion. If None, the first time value in the dataset is used.
    create_new_time : bool, default False
        Whether to create a new time coordinate from scratch.
    sdate : cf.datetime | None, optional
        Start date for creating new time coordinate.
    edate : cf.datetime | None, optional
        End date for creating new time coordinate.
    climatology : bool, default False
        Whether to set climatology bounds.
    clim_sdate : cf.datetime | None, optional
        Start date for climatology bounds.
    clim_edate : cf.datetime | None, optional
        End date for climatology bounds.

    Returns
    -------
    xr.Dataset
        The dataset with updated time attributes.
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
        time_da, bnds_da = build(
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
                time_da, bnds_da = build(
                    clim_sdate, clim_edate, "fx", ref_date, True
                )
            else:
                if clim_sdate is None or clim_edate is None:
                    raise ValueError(
                        "clim_sdate and clim_edate must be provided for climatology=True."
                    )
                time_da, bnds_da = build(
                    clim_sdate, clim_edate, bounds_frequency, ref_date, True
                )
        else:
            if fx:
                if sdate is None or edate is None:
                    raise ValueError(
                        "sdate and edate are required when bounds_frequency='fx'."
                    )
                time_da, bnds_da = build(sdate, edate, "fx", ref_date, False)
            else:
                time_da, bnds_da = build(
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
        ds["climatology_bnds"] = to_numeric(
            ds["climatology_bnds"], time_units, time_cal, "float64"
        )
    else:
        ds["time_bnds"] = to_numeric(
            ds["time_bnds"], time_units, time_cal, "float64"
        )

    ds = ds.assign_coords(
        time=to_numeric(ds["time"], time_units, time_cal, "float64")
    )
    return ds
