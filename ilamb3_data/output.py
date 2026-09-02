import datetime
import uuid

import xarray as xr


def filename_from_attrs(attrs: dict) -> str:
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

def utc_timestamp(time: float | None = None) -> str:
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    else:
        time = datetime.datetime.fromtimestamp(time, datetime.UTC)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")

def new_tracking_id() -> str:
    return "hdl:21.14102/" + str(uuid.uuid4())

def order_dimensions(ds: xr.Dataset, order=("time", "depth", "lat", "lon", "bnds")) -> xr.Dataset:
    return ds.transpose(*order, missing_dims="ignore")
