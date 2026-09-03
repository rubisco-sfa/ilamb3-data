import numpy as np
import xarray as xr


def standardize(
    ds: xr.Dataset,
    units: str,
    positive: str,
    long_name: str,
    *,
    standard_name: str = "depth",
    axis: str = "Z",
    depth_dim: str = "depth",
    compression: dict | None = None,
) -> xr.Dataset:
    """Standardize an existing depth coordinate according to CF conventions.

    Parameters
    ----------
    ds : xr.Dataset
        Dataset containing the depth coordinate to standardize.
    units : str
        Units of the depth coordinate, for example ``"m"``.
    positive : {"up", "down"}
        Direction in which depth coordinate values increase.
    long_name : str
        Descriptive name for the depth coordinate, for example
        ``"depth of sea water"``.
    standard_name : str, default: "depth"
        CF standard name for the vertical coordinate.
    axis : str, default: "Z"
        CF axis attribute.
    depth_dim : str, default: "depth"
        Name of the depth coordinate.
    compression : dict, optional
        Compression settings to add to the depth coordinate and bounds encodings.

    Returns
    -------
    xr.Dataset
        The dataset with standardized depth coordinate metadata and encoding.

    Notes
    -----
    If the coordinate identifies a bounds variable through its ``bounds``
    attribute, that reference is preserved and the bounds encoding is standardized
    as well.
    """
    if depth_dim not in ds.coords:
        raise KeyError(f"Coordinate {depth_dim!r} not found in dataset")
    if ds[depth_dim].dims != (depth_dim,):
        raise ValueError(
            f"Coordinate {depth_dim!r} must be one-dimensional and use a dimension "
            "of the same name"
        )
    if positive not in ("up", "down"):
        raise ValueError("positive must be either 'up' or 'down'")

    depth_da = ds[depth_dim]
    bounds_name = depth_da.attrs.get("bounds")
    if bounds_name is not None and not isinstance(bounds_name, str):
        raise ValueError("The depth coordinate's bounds attribute must be a string")

    depth_da.attrs = {
        "units": units,
        "positive": positive,
        "axis": axis,
        "standard_name": standard_name,
        "long_name": long_name,
    }
    if bounds_name is not None:
        depth_da.attrs["bounds"] = bounds_name

    encoding = {"_FillValue": None, "dtype": "float32"}
    if compression is not None:
        encoding.update(compression)
    depth_da.encoding = encoding
    ds[depth_dim] = depth_da

    if bounds_name is not None and bounds_name in ds:
        bounds_da = ds[bounds_name]
        bounds_da.attrs = {}
        bounds_da.encoding = encoding.copy()
        ds[bounds_name] = bounds_da

    return ds


def build_from_bounds(
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
