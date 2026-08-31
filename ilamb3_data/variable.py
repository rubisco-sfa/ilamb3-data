import re
import warnings

import numpy as np
import xarray as xr
from cf_units import Unit
from intake_esgf import ESGFCatalog


def lookup_cmip6(search_key: str, variable_id: str | None = None) -> dict:
    """
    Find CMIP6 variable information using the intake-esgf catalog.

    Parameters
    ----------
    search_key : str
        The search key for the CMIP6 variable. Could be a phrase or best guess at a
        variable ID.
    variable_id : str, optional
        The variable ID to select from the returned CMIP6 variable info query results.

    Returns
    -------
    dict
        A dictionary containing "cf_standard_name", "variable_long_name", and
        "variable_units".

    Note
    ----
    This needs to be able to return a df with keys or a dictionary depending on if the
    variable_id is provided. Not providing variable_id shouldn't cause the fun to fail,
    but I haven't implemented this yet.
    """
    df = ESGFCatalog().variable_info(search_key)
    if variable_id:
        if variable_id not in df.index:
            raise ValueError(
                f"Variable ID '{variable_id}' not found in CMIP6 variable info. "
                f"Use existing information from the {variable_id} attrs instead."
            )
        return df.loc[variable_id].to_dict()
    else:
        raise ValueError(
            "variable_id must be provided to get a dict for your search key. If you "
            "just want to see query results, run "
            "intake-esgf.ESGFCatalog().variable_info(<search_key>) instead."
        )


def convert_units(
    da: xr.DataArray,
    target_units: str,
) -> xr.DataArray:
    """
    Convert a DataArray from its current units to target_units using cf_units.
    - Reads current_units = da.attrs['units']
    - Raises if no units or not convertible
    - Returns a new DataArray with converted data and units attr updated
    """
    current_units = da.attrs.get("units")
    if current_units is None:
        raise ValueError(f"Cannot convert: '{da.name}' has no 'units' attribute.")
    orig_u = Unit(current_units)
    target_u = Unit(target_units)
    if not orig_u.is_convertible(target_u):
        raise ValueError(
            f"Units '{current_units}' are not convertible to '{target_units}'."
        )

    # do the conversion
    new_vals = orig_u.convert(da.values, target_u)

    # build a new DataArray, preserving dims/coords/name and other attrs
    new_attrs = dict(da.attrs)
    new_attrs["units"] = target_units

    return xr.DataArray(
        data=new_vals, coords=da.coords, dims=da.dims, name=da.name, attrs=new_attrs
    )


# default CF _FillValue options (see https://docs.unidata.ucar.edu/netcdf-c/current/file_format_specifications.html#classic_format_spec)
_FILL_VALUES = {
    np.dtype("S1"): np.bytes_(b"\x00"),  # char (fixed-length)
    np.dtype("U"): "",  # string (variable-length)
    np.int8: np.int8(-127),  # byte
    np.int16: np.int16(-32767),  # short
    np.int32: np.int32(2147483647),  # int
    np.float32: np.float32(1.0e20),  # float
    np.float64: np.float64(1.0e20),  # double
}


def _sanitize_units(u: str) -> str:
    """
    Turn things like "gC/m^2/day" into "gC m-2 day-1",
    which cf_units will happily parse.
    """
    # 1) squared denominators: /foo^2 → " foo-2"
    u = re.sub(r"/([A-Za-z]+)\^?2", r" \1-2", u)
    # 2) any remaining single denominators: /foo → " foo-1"
    u = re.sub(r"/([A-Za-z]+)", r" \1-1", u)
    # 3) strip any stray carets
    return u.replace("^", "")


def standardize(
    ds: xr.Dataset,
    var: str,
    *,
    units: str,
    standard_name: str,
    long_name: str,
    ancillary_variables: str | None = None,
    cell_methods: str | None = None,
    flag_values: np.ndarray | None = None,
    flag_meanings: list[str] | None = None,
    extra_attrs: dict | None = None,
    target_dtype: str | np.dtype | None = None,
    nodata_value: float | None = None,
    convert: bool = False,
    compression: dict | None = None,
    overwrite: bool = False,
) -> xr.Dataset:
    """
    Set CF-compliant attributes for a variable in the dataset, with optional unit
    conversion and dtype handling. The user can add extra attributes via `extra_attrs`
    that may or may not be CF compliant.

    Parameters
    ----------
    ds : xr.Dataset
        The input dataset containing the variable to update.
    var : str
        The name of the variable in the dataset to update.
    units : str
        The desired CF-compliant units for the variable (e.g., "kg m-2 s-1").
    standard_name : str
        The CF standard_name for the variable (e.g., "surface_air_pressure"). Ideally,
        users should choose a CMIP6 or MIP standard_name if possible.
    long_name : str
        A descriptive long name for the variable. The long_name should be
        CMIP6/MIP-compliant if possible.
    ancillary_variables : str, optional
        A space-separated string of ancillary variable names, if applicable.
        Ancillary variables are usually used to reference an uncertainty variable.
    cell_methods : str, optional
        A string describing the cell methods applied to the variable, if applicable.
        E.g., if the data are observations aggregated over time -> "time: mean".
    flag_values : np.ndarray, optional
        An array of flag values for categorical data. Must be provided alongside
        `flag_meanings`.
    flag_meanings : list[str], optional
        A list of human-readable meanings corresponding to each flag value. Must be
        provided alongside `flag_values`.
    extra_attrs : dict, optional
        Any additional attributes to add to the variable that may not be CF compliant.
    target_dtype : str or np.dtype, optional
        The desired final dtype for the variable (e.g., "float32", "int16").
        If not provided, keeps existing dtype.
    nodata_value : int, float, or None, optional
        The value that is currently used to denote missing data for the variable. This
        value will be converted to the appropriate CF _FillValue based on the current
        dtype or the target_dtype if provided. If nodata_value is None, the function
        will look for existing missing value markers in attrs/encoding.
    convert : bool, default False
        Whether to convert the variable to the target unit.
        If False and the variable has existing units that differ from the target,
        a warning is issued and units are left unchanged.
    compression : dict, optional
        A dictionary of compression settings to add to the variable's encoding
        (e.g., {"zlib": True, "complevel": 4}).
    overwrite : bool, default False
        Whether to overwrite all existing attributes and encoding. If False, existing
        attributes are preserved and updated with any new ones provided here. If True,
        all existing attributes and encoding are cleared before setting the new ones.

    Returns
    -------
    xr.Dataset
        The updated dataset with CF-compliant attributes set for the specified variable.
    """
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    da = ds[var]

    # capture existing missing_value marker
    _NODATA_KEYS = ("missing_value", "_FillValue", "nodata")

    def _get_nodata(da: xr.DataArray) -> int | float | None:
        for key in _NODATA_KEYS:
            val = da.encoding.get(key, da.attrs.get(key, None))
            if val is not None:
                return val
        return None

    mv = nodata_value if nodata_value is not None else _get_nodata(da)

    # optionally overwrite all existing attrs
    if overwrite:
        da.attrs = {}
        da.encoding = {}

    # optional unit conversion
    # look for a key that lowercased is “unit” or “units”
    units_key = next((k for k in da.attrs if k.lower() in ("unit", "units")), None)
    if units_key:
        # pop it out under the exact name “units”
        da.attrs["units"] = da.attrs.pop(units_key)

        # sanitize any funky notation so cf_units can parse it
        orig = da.attrs["units"]
        sanitized_unit = _sanitize_units(orig)
        if sanitized_unit != orig:
            da.attrs["units"] = sanitized_unit

    current_units = da.attrs.get("units", None)  # current units attribute
    effective_units = units  # desired units

    # if current units exist and differ from the target...
    if current_units is not None and current_units != units:
        # if wanting to convert, do so and set the unit attribute
        if convert:
            warnings.warn(f"Converting {var} units from {current_units} to {units}")
            da = convert_units(da, units)
        # if not wanting to convert, keep the original units and unit attribute
        else:
            warnings.warn(
                f"Variable '{var}' has units '{current_units}', "
                f"requested '{units}'. Keeping existing units."
            )
            effective_units = current_units

    # do some cleaning
    clean_long_name = re.sub(r"\s*\[[^\]]+\]", "", long_name).strip()
    if not clean_long_name:
        raise ValueError("long_name cannot be empty.")
    # create CF attrs
    attrs = {
        "units": effective_units,
        "standard_name": standard_name,
        "long_name": clean_long_name,
    }

    # assign other variables attr if needed
    if ancillary_variables is not None:
        attrs["ancillary_variables"] = ancillary_variables
    if cell_methods is not None:
        attrs["cell_methods"] = cell_methods
    if (flag_values is None) != (flag_meanings is None):
        raise ValueError("flag_values and flag_meanings must be provided together.")
    if flag_values is not None and flag_meanings is not None:
        if len(flag_values) != len(flag_meanings):
            raise ValueError("flag_values and flag_meanings must have the same length.")
        final_dt = np.dtype(target_dtype) if target_dtype is not None else da.dtype
        attrs["flag_values"] = np.asarray(flag_values, dtype=final_dt)
        attrs["flag_meanings"] = " ".join(flag_meanings)
    if extra_attrs is not None:
        attrs.update(extra_attrs)

    # assign the attrs
    da = da.assign_attrs(attrs)

    # set final dtype
    final_dt = np.dtype(target_dtype) if target_dtype is not None else da.dtype
    da = da.astype(final_dt)

    # determine final dtype and CF _FillValue
    fill = _FILL_VALUES.get(final_dt, None)
    if fill is None:
        # fallback by kind and size
        if final_dt.kind in ("i", "u"):
            fill = _FILL_VALUES.get(
                {1: np.int8, 2: np.int16, 4: np.int32}[final_dt.itemsize]
            )
        elif final_dt.kind == "f":
            fill = _FILL_VALUES.get({4: np.float32, 8: np.float64}[final_dt.itemsize])
        elif final_dt.kind == "S":
            fill = _FILL_VALUES[np.dtype("S1")]
    if fill is None:
        raise ValueError(f"No CF _FillValue defined for dtype '{final_dt}'.")

    # handle data and encoding based on dtype
    if np.issubdtype(final_dt, np.floating):
        # Floats: leave NaNs, just set encoding
        da.encoding["_FillValue"] = fill
    else:
        # Ints/bytes: replace NaNs and old markers, cast to final_dtype
        data = da.values.copy()
        if np.issubdtype(data.dtype, np.floating):
            data = np.where(np.isnan(data), fill, data)
        if mv is not None:
            data[data == mv] = fill
        da = da.copy(data=data.astype(final_dt))
        da.encoding["_FillValue"] = fill

    # remove any old missing_value attributes; not required by CF anymore
    da.attrs.pop("missing_value", None)
    da.encoding.pop("missing_value", None)

    # set compression encoding
    if compression is not None:
        da.encoding.update(compression)

    # reassign back to dataset
    ds[var] = da
    return ds
