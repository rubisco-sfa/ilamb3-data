import numpy as np
import xarray as xr


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
