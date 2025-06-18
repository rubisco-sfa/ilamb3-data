import datetime
import json
import os
import urllib.request
import uuid
import warnings
from calendar import monthrange
from typing import Optional

import cftime as cf
import numpy as np
import pandas as pd
import pooch
import requests
import xarray as xr
from cf_units import Unit
from intake_esgf import ESGFCatalog
from tqdm import tqdm


def create_registry(registry_file: str) -> pooch.Pooch:
    """
    Return the pooch ilamb reference data catalog.

    Returns
    -------
    pooch.Pooch
        The intake ilamb reference data catalog.
    """

    registry = pooch.create(
        path=pooch.os_cache("ilamb3"),
        base_url="https://www.ilamb.org/ilamb3-data",
        version="0.1",
        env="ILAMB_ROOT",
    )
    registry.load_registry(registry_file)
    return registry


def download_from_html(remote_source: str, local_source: str | None = None) -> str:
    """
    Download a file from a remote URL to a local path.
    If the "content-length" header is missing, it falls back to a simple download.
    """
    if local_source is None:
        local_source = os.path.basename(remote_source)
    if os.path.isfile(local_source):
        return local_source

    resp = requests.get(remote_source, stream=True)
    try:
        total_size = int(resp.headers.get("content-length"))
    except (TypeError, ValueError):
        total_size = 0

    with open(local_source, "wb") as fdl:
        if total_size:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=local_source
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        fdl.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    fdl.write(chunk)
    return local_source


def download_file(remote_source: str, local_source: str | None = None) -> str:
    """
    Download a file from a remote URL to a local path.
    If the "content-length" header is missing, it falls back to a simple download.
    """
    if local_source is None:
        local_source = os.path.basename(remote_source)
    if os.path.isfile(local_source):
        return local_source

    resp = requests.get(remote_source, stream=True)
    try:
        total_size = int(resp.headers.get("content-length"))
    except (TypeError, ValueError):
        total_size = 0

    with open(local_source, "wb") as fdl:
        if total_size:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=local_source
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        fdl.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in resp.iter_content(chunk_size=1024):
                if chunk:
                    fdl.write(chunk)
    return local_source


def download_from_zenodo(record: dict):
    """
    Download all files from a Zenodo record dict into a '_temp' directory.
    Example for getting a Zenodo record:

        # Specify the dataset title you are looking for
        dataset_title = "Global Fire Emissions Database (GFED5) Burned Area"

        # Build the query string to search by title
        params = {
            "q": f'title:"{dataset_title}"'
        }

        # Define the Zenodo API endpoint
        base_url = "https://zenodo.org/api/records"

        # Send the GET request
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            print("Error during search:", response.status_code)
            exit(1)

        # Parse the JSON response
        data = response.json()

        # Get record dictionary
        records = data['hits']['hits']
        record = data['hits']['hits'][0]
    """
    download_dir = "_temp"
    os.makedirs(download_dir, exist_ok=True)

    title = record.get("metadata", {}).get("title", "No Title")
    pub_date = record.get("metadata", {}).get("publication_date", "No publication date")
    print(f"Found record:\n  Title: {title}\n  Publication Date: {pub_date}")

    for file_info in record.get("files", []):
        file_name = file_info.get("key")
        file_url = file_info.get("links", {}).get("self")
        local_file = os.path.join(download_dir, file_name)

        if file_url:
            print(f"Downloading {file_name} from {file_url} into {download_dir}...")
            download_from_html(file_url, local_source=local_file)
        else:
            print(f"File URL not found for file: {file_name}")


# I think this can be useful to make sure people export the netcdfs the same way every time
def create_output_filename(attrs: dict, time_range: str) -> str:
    """
    Generate a NetCDF filename using required attributes and a time range.

    Args:
        attrs (dict): Dictionary of global attributes.
        time_range (str): Time range string to embed in the filename.

    Returns:
        str: Formatted filename.

    Raises:
        ValueError: If any required attributes are missing from `attrs`.
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
        **attrs, time_mark=time_range
    )
    return filename


def get_cmip6_variable_info(variable_id: str) -> dict[str, str]:
    """ """
    df = ESGFCatalog().variable_info(variable_id)
    return df.iloc[0].to_dict()


def generate_time_bounds_var(time_da: xr.DataArray) -> xr.DataArray:
    """
    From CF-Conventions: "The boundary variable time_bnds associates a
    time point i with the time interval whose boundaries are time_bnds(i,0)
    and time_bnds(i,1). The instant time(i) should be contained within the
    interval, or be at one end of it. For instance, with i=2 we might have
    time(2)=10.5, time_bnds(2,0)=10.0, time_bnds(2,1)=11.0."
    """
    units = time_da.encoding["units"]
    calendar = time_da.encoding["calendar"]

    nums = cf.date2num(list(time_da.values), units, calendar)  # [365, 730, 1095, ...]
    half = np.diff(nums) / 2.0  # [182.5, 182.5, 182.5, ...]

    lowers = np.empty_like(nums)
    uppers = np.empty_like(nums)
    for i in range(len(nums)):
        lowers[i] = nums[i] - (
            half[0] if i == 0 else half[i - 1]
        )  # [182.5, 547.5, ...]
        uppers[i] = nums[i] + (
            half[i] if i < len(half) else half[-1]
        )  # [547.5, 912.5, ...]
    bounds_nums = np.stack(
        [lowers, uppers], axis=1
    )  # [[182.5, 547.5], [547.5, 912.5], ...]

    bounds_cfdates = cf.num2date(bounds_nums.tolist(), units, calendar)
    cfdates_arr = np.array([tuple(pair) for pair in bounds_cfdates], dtype=object)

    return cfdates_arr


def _to_cftime(time_da: xr.DataArray, calendar: str = "gregorian") -> xr.DataArray:
    """
    Convert a DataArray whose values are np.datetime64 or strings
    into one whose values are cftime.Datetime* objects.
    """
    # parse with pandas
    pd_times = pd.to_datetime(time_da.values, utc=False)
    # convert to plain Python datetimes
    py_times = pd_times.to_pydatetime().tolist()

    # re-build the list of datetimes with cf.datetime objects
    cftime_vals = [
        cf.DatetimeGregorian(dt.year, dt.month, dt.day, dt.hour, dt.minute, dt.second)
        for dt in py_times
    ]

    # build a new data array from list
    return xr.DataArray(
        data=np.array(cftime_vals, dtype=object),
        dims=time_da.dims,
        coords={time_da.name: time_da.coords[time_da.name]},
        name=time_da.name,
    )


def set_time_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    1) Recast ds.time into cftime.Datetime* under 'standard'
    2) Stamp CF attributes + encoding (with _FillValue=None)
    3) Drop & regenerate time_bnds (with matching encoding)
    """

    # select time coordinate values as data array
    if "time" not in ds.coords:
        raise ValueError("Dataset must have a 'time' coordinate.")
    time_da = ds.coords["time"]

    # ensure time coordinate values are of type cf.datetime
    if np.issubdtype(time_da.dtype, np.datetime64) or (
        time_da.dtype == object and isinstance(time_da.values[0], str)
    ):
        time_da = _to_cftime(time_da, calendar="gregorian")

    elif isinstance(time_da.values[0], cf.datetime):
        pass
    else:
        raise TypeError(f"Unsupported time type: {time_da.dtype}")

    # convert cf.datetime objects of <calendar> to gregorian
    calendar = time_da.values[0].calendar
    ref_date = time_da.values[0]
    ref_str = f"{ref_date.year:04d}-{ref_date.month:02d}-{ref_date.day:02d} {ref_date.hour:02d}:{ref_date.minute:02d}:{ref_date.second:02d}"
    units = f"days since {ref_str}"
    if calendar != "gregorian":
        raw = list(time_da.values)
        date_nums = cf.date2num(raw, units, calendar)
        cf_date_arr = cf.num2date(date_nums.tolist(), units, "gregorian")
        time_da = xr.DataArray(
            cf_date_arr,
            dims=("time",),
            coords={"time": cf_date_arr},
            name="time",
        )

    # set CF attrs
    attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
        "bounds": "time_bnds",
    }

    # assign time attributes
    new_time_da = time_da.assign_attrs(**attrs)

    # set encoding and re-assign coordinates
    new_time_da.encoding = {
        "units": units,
        "calendar": "gregorian",
        "dtype": "float64",
        "_FillValue": None,
    }
    ds = ds.assign_coords(time=new_time_da)

    # drop & regenerate bounds with matching encoding
    if "time_bnds" in ds.data_vars:
        ds = ds.drop_vars("time_bnds")
    bnds_arr = generate_time_bounds_var(ds.coords["time"])

    # add to the dataset
    ds["time_bnds"] = (("time", "bnds"), bnds_arr)

    # ensure no attrs and set encoding
    ds["time_bnds"].encoding.clear()
    ds["time_bnds"].attrs.clear()
    ds["time_bnds"].encoding = {"dtype": "float64"}

    return ds


def set_lat_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure the xarray dataset's latitude attributes are formatted according to CF-Conventions.
    """
    assert "lat" in ds
    da = ds["lat"]
    da.attrs = {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "latitude",
    }
    ds["lat"] = da
    return ds


def set_lon_attrs(ds: xr.Dataset) -> xr.Dataset:
    """
    Ensure the xarray dataset's longitude attributes are formatted according to CF-Conventions.
    """
    assert "lon" in ds
    da = ds["lon"]
    da.attrs = {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "long_name": "longitude",
    }
    ds["lon"] = da
    return ds


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
        data=new_vals,
        coords=da.coords,
        dims=da.dims,
        name=da.name,
        attrs=new_attrs,
        encoding=da.encoding,  # preserve encoding (you may want to clear dtype and calendar)
    )


def set_var_attrs(
    ds: xr.Dataset,
    var: str,
    cmip6_units: str,
    cmip6_standard_name: str,
    cmip6_long_name: str,
    *,
    convert: bool = False,
) -> xr.Dataset:
    """
    Ensure ds[var] has CF‐compliant attrs:
      - Optionally convert from its existing units to cmip6_units
      - Warn (but do not fail) if units don’t match and convert=False;
        in that case, keep the original units.
      - Then set units (either converted or original), standard_name, long_name
    """
    if var not in ds:
        raise KeyError(f"Variable '{var}' not found in dataset.")
    da = ds[var]

    current_units = da.attrs.get("units")
    effective_units = cmip6_units

    if current_units is not None and current_units != cmip6_units:
        if convert:
            da = convert_units(da, cmip6_units)
        else:
            warnings.warn(
                f"Variable '{var}' has units '{current_units}', "
                f"requested '{cmip6_units}'. Keeping existing units."
            )
            effective_units = current_units

    # now safe to overwrite attrs
    da.assign_attrs(
        {
            "units": effective_units,
            "standard_name": cmip6_standard_name,
            "long_name": cmip6_long_name,
        }
    )

    ds[var] = da
    return ds


def gen_utc_timestamp(time: float | None = None) -> str:
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    else:
        time = datetime.datetime.fromtimestamp(time)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def gen_trackingid() -> str:
    return "hdl:21.14102/" + str(uuid.uuid4())


def add_time_bounds2(ds: xr.Dataset) -> xr.Dataset:
    """
    Adds a time_bounds variable to the xarray dataset according to CF conventions.
    Assumes time intervals are consecutive.
    """
    # Temporarily convert cftime to numpy datetime64 for bounds calculation
    if isinstance(ds["time"].values[0], cf.DatetimeGregorian):
        times = [
            np.datetime64(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in ds["time"].values
        ]
        ds = ds.assign_coords(time=("time", times))

    time = ds["time"]
    encoding = time.encoding

    # Generate time bounds
    bounds = np.empty((len(time), 2), dtype="datetime64[ns]")
    for i, t in enumerate(pd.to_datetime(time.values)):
        # Start of the month
        start = t.replace(day=1)
        # End of the month (last day)
        end = (start + pd.DateOffset(months=1) - pd.DateOffset(days=1)).normalize()

        bounds[i, 0] = np.datetime64(start)
        bounds[i, 1] = np.datetime64(end)

    # Create the bounds DataArray with the same encoding as time
    time_bounds = xr.DataArray(
        bounds,
        dims=["time", "bounds"],
    )
    time_bounds.encoding.update(encoding)
    # Attach the bounds to the dataset
    ds["time_bnds"] = time_bounds
    ds.time.attrs["bounds"] = "time_bnds"

    return ds


def fix_time(ds: xr.Dataset) -> xr.DataArray:
    """
    Ensure the xarray dataset's time attributes are formatted according to CF-Conventions.
    """
    assert "time" in ds
    da = ds["time"]

    # Ensure time is an accepted xarray time dtype
    if np.issubdtype(da.dtype, np.datetime64):
        ref_date = np.datetime_as_string(da.min().values, unit="s")
    elif isinstance(da.values[0], cf.datetime):
        ref_date = da.values[0].strftime("%Y-%m-%d %H:%M:%S")
    else:
        raise TypeError(
            f"Unsupported xarray time format: {type(da.values[0])}. Accepted types are np.datetime64 or cftime.datetime."
        )

    da.encoding = {
        "units": f"days since {ref_date}",
        "calendar": da.encoding.get("calendar"),
    }
    da.attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
    }
    ds["time"] = da
    return ds


def add_time_bounds(ds: xr.Dataset) -> xr.Dataset:
    """
    Add bounds to the time coordinate using numpy.datetime64 format.
    """
    # Temporarily convert cftime to numpy datetime64 for bounds calculation
    if isinstance(ds["time"].values[0], cf.DatetimeGregorian):
        times = [
            np.datetime64(t.strftime("%Y-%m-%dT%H:%M:%S")) for t in ds["time"].values
        ]
        ds = ds.assign_coords(time=("time", times))

    # Add bounds using cf_xarray
    ds = ds.cf.add_bounds("time")
    return ds


def add_time_bounds_monthly(ds: xr.Dataset) -> xr.Dataset:
    """
    Add CF-compliant monthly time bounds and preserve existing mid-month time values.

    Parameters:
        ds: xarray.Dataset with a 'time' coordinate of monthly midpoints.

    Returns:
        Dataset with 'time_bnds' for start/end of each month and unchanged 'time'.
    """
    time_units = ds["time"].attrs.get("units", "days since 2000-01-01")
    calendar = ds["time"].attrs.get("calendar", "standard")

    # Convert time values to cftime objects if needed
    times = ds["time"].values
    if not isinstance(times[0], cf.DatetimeNoLeap):
        if np.issubdtype(times.dtype, np.number):
            times = xr.coding.times.decode_cf_datetime(
                times, units=time_units, calendar=calendar
            )

    lower_bounds = []
    upper_bounds = []

    for t in times:
        # Get current month start
        month_start = cf.DatetimeGregorian(t.year, t.month, 1)
        # Next month start
        if t.month == 12:
            next_month = cf.DatetimeGregorian(t.year + 1, 1, 1)
        else:
            next_month = cf.DatetimeGregorian(t.year, t.month + 1, 1)

        lower_bounds.append(month_start)
        upper_bounds.append(next_month)

    bounds_array = list(zip(lower_bounds, upper_bounds))
    ds["time_bnds"] = xr.DataArray(
        data=bounds_array,
        dims=("time", "bounds"),
        coords={"time": ds.time},
        attrs={"units": time_units, "calendar": calendar},
    )

    # Ensure CF-compliant metadata and encoding
    ds["time"].attrs.update(
        {
            "units": time_units,
            "calendar": calendar,
            "bounds": "time_bnds",
            "standard_name": "time",
            "long_name": "Time",
        }
    )
    ds["time"].encoding.update({"units": time_units, "calendar": calendar})
    ds["time_bnds"].encoding.update({"units": time_units, "calendar": calendar})

    return ds


def set_cf_global_attributes(
    ds: xr.Dataset,
    *,  # keyword only for the following args
    title: str,
    institution: str,
    source: str,
    history: str,
    references: str,
    comment: str,
    conventions: str,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12.

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        title (str): Short description of the file contents.
        institution (str): Where the original data was produced.
        source (str): Method of production of the original data.
        history (str): List of applications that have modified the original data.
        references (str): References describing the data or methods used to produce it.
        comment (str): Miscellaneous information about the data or methods used.
        conventions (str): The name of the conventions followed by the dataset.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If a required global attribute is missing.
    """

    # Build and validate attributes
    attrs = {
        "title": title,
        "institution": institution,
        "source": source,
        "history": history,
        "references": references,
        "comment": comment,
        "Conventions": conventions,
    }

    # Ensure all values are explicitly set (None not allowed)
    missing = [k for k, v in attrs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    ds.attrs.update(attrs)
    return ds


def load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def get_nested_dict(data, path, default=None):
    for key in path:
        try:
            if isinstance(data, dict):
                data = data.get(key, default)
            elif isinstance(data, list) and isinstance(key, int):
                data = data[key]
            else:
                return default
        except (IndexError, TypeError):
            return default
    return data


def set_ods_global_attrs(
    ds: xr.Dataset,
    *,
    activity_id="obs4MIPs",
    aux_variable_id: Optional[str] = "N/A",
    comment: Optional[str] = "N/A",
    contact: str,
    conventions="CF-1.12 ODS-2.5",
    creation_date: str,
    dataset_contributor: str,
    data_specs_version: str,
    doi: str = "N/A",
    external_variables: Optional[str] = "N/A",
    frequency: str,
    grid: str,
    grid_label: str,
    has_auxdata: bool,
    history: str,
    institution: str,
    institution_id: str,
    license: str,
    nominal_resolution: str,
    processing_code_location: str,
    product: str,
    realm: str,
    references: str,
    region: str,
    source: str,
    source_id: str,
    source_data_retrieval_date: Optional[str] = "N/A",
    source_data_url: Optional[str] = "N/A",
    source_label: str,
    source_type: str,
    source_version_number: str,
    title: Optional[str] = "N/A",
    tracking_id: str,
    variable_id: str,
    variant_label: str,
    variant_info: str,
    version: Optional[str] = str,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12 and ODS-2.5.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to None.

    Special behavior:
        - Attributes with default values must match those defaults unless explicitly overridden.
        - Some attributes must be selected from a predefined list.
        - Some attributes are validated against controlled vocabularies loaded from the official online JSON files.
        - Nested vocabularies (e.g., obs4MIPs_CV.json) are accessed by key lookup (e.g., search "source_id").

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If any required attribute is missing or not valid.
    """

    valid_external_variables = ["areacella", "areacello", "volcella", "volcello"]

    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    freq_cv = load_json_from_url(base_url + "obs4MIPs_frequency.json")
    institution_cv = load_json_from_url(base_url + "obs4MIPs_institution_id.json")
    nominal_res_cv = load_json_from_url(base_url + "obs4MIPs_nominal_resolution.json")
    realm_cv = load_json_from_url(base_url + "obs4MIPs_realm.json")
    region_cv = load_json_from_url(base_url + "obs4MIPs_region.json")
    source_id_cv = load_json_from_url(base_url + "obs4MIPs_source_id.json")
    source_type_cv = load_json_from_url(base_url + "obs4MIPs_source_type.json")
    top_level_cv = load_json_from_url(base_url + "Tables/obs4MIPs_CV.json")
    grid_labels_cv = load_json_from_url(base_url + "obs4MIPs_grid_label.json")
    products_cv = load_json_from_url(base_url + "obs4MIPs_product.json")
    if frequency == "mon":
        mip_tables = [
            "Tables/obs4MIPs_Lmon.json",
            "Tables/obs4MIPs_Omon.json",
            "Tables/obs4MIPs_Amon.json",
        ]
        for table in mip_tables:
            if variable_id in get_nested_dict(
                load_json_from_url(base_url + table), ["variable_entry"]
            ):
                realm = get_nested_dict(
                    load_json_from_url(base_url + table), ["Header"]
                )["realm"]
                variable_cv = load_json_from_url(base_url + table)
                table_name = table.split("_")[-1].split(".")[0]

    errors = []
    if has_auxdata:
        if aux_variable_id == "None":
            errors.append("must specify ancillary variable_ids if included")

    # Check vals dependent on "valid_" lists hard-coded above
    if grid_label not in grid_labels_cv["grid_label"]:
        errors.append("grid_label must match a key in obs4MIPs_grid_label.json")
    if product not in products_cv["product"]:
        errors.append("product must match a key in obs4MIPs_product.json")
    # if external_variables:
    #     if external_variables not in valid_external_variables:
    #         errors.append(
    #             f"external_variables must be one of {valid_external_variables}"
    #         )
    # Check vals present in Github json files
    if frequency not in freq_cv["frequency"]:
        errors.append("frequency must match a key in obs4MIPs_frequency.json")
    # if institution_id not in institution_cv["institution_id"]:
    #     errors.append("institution_id must match a key in obs4MIPs_institution_id.json")
    # if nominal_resolution not in nominal_res_cv["nominal_resolution"]:
    #     errors.append(
    #         "nominal_resolution must match a key in obs4MIPs_nominal_resolution.json"
    #     )
    if realm not in realm_cv["realm"]:
        errors.append("realm must match a key in obs4MIPs_realm.json")
    if region not in region_cv["region"]:
        errors.append("region must match a key in obs4MIPs_region.json")
    # if source_id not in source_id_cv["source_id"]:
    #     errors.append("source_id must match a key in obs4MIPs_source_id.json")
    # if source_type not in source_type_cv["source_type"]:
    #     errors.append("source_type must match a key in obs4MIPs_source_type.json")
    # Check vals *nested* within Github json files
    # if source_label != get_nested_dict(
    #     source_id_cv, ["source_id", source_id, "source_label"]
    # ):
    #     errors.append(
    #         "source_label must match the label inside obs4MIPs_source_id.json[source_id]"
    #     )
    # if source_version_number not in get_nested_dict(
    #     source_id_cv, ["source_id", source_id, "source_version_number"]
    # ):
    #     errors.append("source_version_number must match a value inside source_id entry")
    # if source_id not in get_nested_dict(top_level_cv, ["CV", "source_id"]):
    #     errors.append(
    #         "source_id must be present in obs4MIPs_CV.json under 'source_id' key"
    #     )
    # if source not in get_nested_dict(
    #     top_level_cv, ["CV", "source_id", source_id, "source"]
    # ):
    #     errors.append(
    #         "source must match a attribute in the 'source_id' section of obs4MIPs_CV.json"
    #     )
    # if variable_id not in get_nested_dict(variable_cv, ["variable_entry"]):
    #     errors.append(
    #         f"variable_id must match a key in the 'variable_entry' section of {table_name}"
    #     )

    if errors:
        raise ValueError("\n".join(errors))

    attrs = {
        "activity_id": activity_id,
        "aux_variable_id": aux_variable_id,
        "comment": comment,
        "contact": contact,
        "Conventions": conventions,
        "creation_date": creation_date,
        "dataset_contributor": dataset_contributor,
        "data_specs_version": data_specs_version,
        "doi": doi,
        "frequency": frequency,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_frequency.json
        "grid": grid,
        "grid_label": grid_label,  # ["gn", "gr1"]
        "has_auxdata": has_auxdata,
        "history": history,
        "institution": institution,
        "institution_id": institution_id,  # have to be registered on https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json
        "license": license,
        "nominal_resolution": nominal_resolution,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_nominal_resolution.json
        "processing_code_location": processing_code_location,
        "product": product,  # [“observations”, “reanalysis”, “in_situ”, “exploratory_product”]
        "realm": realm,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_realm.json
        "references": references,
        "region": region,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_region.json
        "source": source,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/Tables/obs4MIPs_CV.json (search source_id)
        "source_data_retrieval_date": source_data_retrieval_date,
        "source_data_url": source_data_url,
        "source_id": source_id,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json
        "source_label": source_label,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json (nested source_label)
        "source_type": source_type,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_type.json
        "source_version_number": source_version_number,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json (nested source_version_number)
        "title": title,
        "tracking_id": tracking_id,  # automatically detected by CMOR (hdl:21.14102/<uuid>)
        "variable_id": variable_id,  # https://clipc-services.ceda.ac.uk/dreq/index/CMORvar.html
        "variant_info": variant_info,  # description of who prepared the data, describe obs data variance if applicable
        "variant_label": variant_label,  # same as source_id (if prepped by them), or "BE" if source_id unknown, ensemble member identified via -r1, -r2, ..., -rN
        "version": version,
    }

    missing = [k for k, v in attrs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    ds.attrs.update(attrs)

    return ds


def set_ods_var_attrs(ds: xr.Dataset, variable_id: str) -> xr.Dataset:
    """
    Set required NetCDF variable level attributes according to CF-Conventions 1.12 and ODS-2.5.

    This function validates that all required attributes are provided. Optional fields may be set to None.

    Special behavior:
        - Attributes with default values must match those defaults unless explicitly overridden.
        - Some attributes must be selected from a predefined list.
        - Some attributes are validated against controlled vocabularies loaded from the official online JSON files.
        - Nested vocabularies (e.g., obs4MIPs_CV.json) are accessed by key lookup (e.g., search "source_id").

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If any required attribute is missing or not valid.
    """
    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    if ds.attrs["frequency"] == "mon":
        if ds.attrs["realm"] == "atmos":
            mip_table = "Tables/obs4MIPs_Amon.json"
        elif ds.attrs["realm"] == "land":
            mip_table = "Tables/obs4MIPs_Lmon.json"
        elif ds.attrs["realm"] == "ocean":
            mip_table = "Tables/obs4MIPs_Omon.json"
        varattrs = get_nested_dict(
            load_json_from_url(base_url + mip_table), ["variable_entry"]
        )[ds.attrs["variable_id"]]
        if Unit(varattrs["units"]) != Unit(ds[variable_id].attrs["units"]):
            ds[variable_id].values = Unit(ds[variable_id].attrs["units"]).convert(
                ds[variable_id].values,
                Unit(varattrs["units"]),
                inplace=True,
            )
            ds[variable_id].attrs["history"] = (
                f"{gen_utc_timestamp()} altered by ILAMB: Converted units from '{ds[variable_id].attrs['units']}' to '{varattrs['units']}'."
            )
            ds[variable_id].attrs["original_units"] = ds[variable_id].attrs["units"]
            ds[variable_id].attrs["units"] = varattrs["units"]
        else:
            ds[variable_id].attrs["units"] = varattrs["units"]
        if varattrs["positive"]:
            ds[variable_id].attrs["positive"] = varattrs["positive"]
        print(ds[variable_id].attrs)
        ds[variable_id].attrs.update(
            {
                key: varattrs[key]
                for key in [
                    "standard_name",
                    "long_name",
                    "comment",
                    "cell_methods",
                    "cell_measures",
                ]
            }
        )
        print(ds[variable_id].attrs)
    return ds


def set_ods_calendar(ds: xr.Dataset) -> xr.Dataset:
    ds = ds.convert_calendar("gregorian")
    return ds


def set_ods_coords(ds: xr.Dataset) -> xr.Dataset:
    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    possible_bounds = ["bounds", "lat_bounds", "lon_bounds", "time_bounds"]
    replaced_bounds = ["bnds", "lat_bnds", "lon_bnds", "time_bnds"]
    for bound, rbound in zip(possible_bounds, replaced_bounds):
        if bound in ds:
            ds = ds.rename({bound: rbound})
            if "_" in bound:
                coord = bound.split("_")[0]
                ds[coord].attrs.update({"bounds": rbound})
    coord_table = load_json_from_url(base_url + "Tables/obs4MIPs_coordinate.json")[
        "axis_entry"
    ]

    def find_coord_key(nested_json, coord):
        if isinstance(nested_json, dict):
            for key, value in nested_json.items():
                if isinstance(value, dict) and value.get("out_name") == coord:
                    return key
                result = find_coord_key(value, coord)
                if result:
                    return result
        elif isinstance(nested_json, list):
            for item in nested_json:
                result = find_coord_key(item, coord)
                if result:
                    return result
        return None

    for coord in ds.coords:
        key = find_coord_key(coord_table, coord)
        if key and key != "time":
            ds[coord].attrs.update(
                {
                    k: coord_table[key][k]
                    for k in ["units", "axis", "long_name", "standard_name"]
                }
            )
    return ds
