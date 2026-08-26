import xarray as xr


def standardize(ds: xr.Dataset, compression: dict | None = None) -> xr.Dataset:
    """
    Ensure the xarray dataset's latitude attributes are formatted according to CF-Conventions.
    """

    assert "lat" in ds
    da = ds["lat"]

    bounds = da.attrs.get("bounds")

    da.attrs = {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "Latitude",
    }

    if bounds is not None:
        da.attrs["bounds"] = bounds

    da.encoding.clear()
    da.encoding = {"_FillValue": None, "dtype": "float64"}
    if compression is not None:
        da.encoding.update(compression)
    ds["lat"] = da
    return ds
