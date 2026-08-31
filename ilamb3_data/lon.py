import xarray as xr


def standardize(ds: xr.Dataset, compression: dict | None = None) -> xr.Dataset:
    """
    Ensure the xarray dataset's longitude attributes are formatted according to CF-Conventions.
    """

    assert "lon" in ds
    da = ds["lon"]

    bounds = da.attrs.get("bounds")

    da.attrs = {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "long_name": "Longitude",
    }

    if bounds is not None:
        da.attrs["bounds"] = bounds

    da.encoding.clear()
    da.encoding = {"_FillValue": None, "dtype": "float64"}
    if compression is not None:
        da.encoding.update(compression)
    ds["lon"] = da
    return ds
