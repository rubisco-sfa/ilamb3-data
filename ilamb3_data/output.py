import datetime
import uuid

import xarray as xr


# To-Do: We may want to use the nested directory structure that obs4MIPs uses
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

    # If variable_label is present and not equal to activity_id, append it to variable_id
    variable_id = str(attrs["variable_id"])
    variant_label = attrs.get("variant_label")
    if variant_label and variant_label != attrs["activity_id"]:
        variable_id = f"{variable_id}_{variant_label}"

    filename_attrs = {**attrs, "variable_id": variable_id}
    return (
        "{activity_id}_{institution_id}_{source_id}_{frequency}_"
        "{variable_id}_{grid_label}_{version}.nc"
    ).format(**filename_attrs)


def utc_timestamp(time: float | None = None) -> str:
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    else:
        time = datetime.datetime.fromtimestamp(time, datetime.UTC)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def new_tracking_id() -> str:
    return "hdl:21.14102/" + str(uuid.uuid4())


def order_dimensions(
    ds: xr.Dataset, order=("time", "depth", "lat", "lon", "bnds")
) -> xr.Dataset:
    """Order dimensions in variables and in the serialized dataset.

    ``xarray.Dataset.transpose`` changes the dimension order of each variable,
    but it retains the dataset's variable insertion order. NetCDF dimension
    declarations follow the order in which dimensions first occur in the
    dataset, so transposition alone does not necessarily affect their order.
    """
    dimension_order = [dim for dim in order if dim in ds.dims]
    dimension_order.extend(dim for dim in ds.dims if dim not in dimension_order)
    transposed = ds.transpose(*dimension_order, missing_dims="ignore")

    # Insert one-dimensional dimension coordinates first so dimensions are
    # registered in the requested order. Preserve auxiliary coordinates after
    # the dimension coordinates, in their existing order.
    coordinate_names = [
        dim
        for dim in dimension_order
        if dim in transposed.coords and transposed[dim].dims == (dim,)
    ]
    coordinate_names.extend(
        name for name in transposed.coords if name not in coordinate_names
    )
    result = xr.Dataset(
        coords={name: transposed.coords[name] for name in coordinate_names},
        attrs=transposed.attrs,
    )
    # NetCDF writers serialize data variables before coordinate variables. Put
    # multidimensional primary variables first and bounds variables last so the
    # former establish the main dimension order and ``bnds`` is introduced last.
    bounds_names = {
        name
        for coord in transposed.coords.values()
        for attr in ("bounds", "climatology")
        if isinstance((name := coord.attrs.get(attr)), str)
        and name in transposed.data_vars
    }
    original_position = {
        name: position for position, name in enumerate(transposed.data_vars)
    }
    data_names = sorted(
        transposed.data_vars,
        key=lambda name: (
            name in bounds_names or "bnds" in transposed[name].dims,
            -len(transposed[name].dims),
            original_position[name],
        ),
    )
    result = result.assign({name: transposed[name] for name in data_names})
    result.encoding = transposed.encoding.copy()
    return result
