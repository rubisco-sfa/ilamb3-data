import datetime
import json
import os
import urllib.request
from typing import Optional

import cftime as cf
import numpy as np
import pooch
import requests
import xarray as xr
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
def get_filename(attrs: dict, time_range: str) -> str:
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
        "variable_id",
        "frequency",
        "source_id",
        "variant_label",
        "grid_label",
    ]

    missing = [key for key in required_keys if key not in attrs]
    if missing:
        raise ValueError(
            f"Missing required attributes: {', '.join(missing)}. "
            f"Expected keys: {', '.join(required_keys)}"
        )

    filename = "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
        **attrs, time_mark=time_range
    )
    return filename


def get_cmip6_variable_info(variable_id: str) -> dict[str, str]:
    """ """
    df = ESGFCatalog().variable_info(variable_id)
    return df.iloc[0].to_dict()


def set_time_attrs(ds: xr.Dataset) -> xr.Dataset:
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


def set_var_attrs(
    ds: xr.Dataset, var: str, units: str, standard_name: str, long_name: str
) -> xr.Dataset:
    """
    Ensure the xarray dataset's variable attributes are formatted according to CF-Conventions.
    """
    assert var in ds
    da = ds[var]
    da.attrs = {"units": units, "standard_name": standard_name, "long_name": long_name}
    ds[var] = da
    return ds


def gen_utc_timestamp(time: float | None = None) -> str:
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    else:
        time = datetime.datetime.fromtimestamp(time)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def add_time_bounds_monthly(ds: xr.Dataset) -> xr.Dataset:
    """
    Add monthly time bounds to an xarray Dataset.

    For each timestamp in the dataset's 'time' coordinate, this function adds a new
    coordinate called 'time_bounds' with the first day of the month and the first
    day of the next month. These bounds follow CF conventions.

    Args:
        ds (xr.Dataset): Dataset with a 'time' coordinate of monthly timestamps.

    Returns:
        xr.Dataset: Modified dataset with a 'time_bounds' coordinate and updated
                    attributes on the 'time' coordinate.
    """

    def _ymd_tuple(da: xr.DataArray) -> tuple[int, int, int]:
        """Extract (year, month, day) from a single-element datetime DataArray."""
        if da.size != 1:
            raise ValueError("Expected a single-element datetime for conversion.")
        return int(da.dt.year), int(da.dt.month), int(da.dt.day)

    def _make_timestamp(t: xr.DataArray, ymd: tuple[int, int, int]) -> np.datetime64:
        """Construct a timestamp matching the type of the input time value."""
        try:
            return type(t.item())(*ymd)  # try using the same class as the input
        except Exception:
            # fallback to datetime64 if direct construction fails
            return np.datetime64(f"{ymd[0]:04d}-{ymd[1]:02d}-{ymd[2]:02d}")

    lower_bounds = []
    upper_bounds = []

    for t in ds["time"]:
        year, month, _ = _ymd_tuple(t)
        lower_bounds.append(_make_timestamp(t, (year, month, 1)))

        # First day of the next month (verbose-ified for easier readability)
        if month == 12:
            next_month = (year + 1, 1, 1)
        else:
            next_month = (year, month + 1, 1)
        upper_bounds.append(_make_timestamp(t, next_month))

    bounds_array = np.array([lower_bounds, upper_bounds]).T
    ds = ds.assign_coords(time_bounds=(("time", "bounds"), bounds_array))
    ds["time_bounds"].attrs["long_name"] = "time_bounds"
    ds["time"].attrs["bounds"] = "time_bounds"

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


# FUNCTION IS UNDER CONSTRUCTION
def set_ods_global_attributes(
    ds: xr.Dataset,
    *,
    activity_id="obs4MIPs",
    comment: Optional[str] = None,
    contact: str,
    conventions="CF-1.12 ODS-2.5",
    creation_date: str,
    dataset_contributor: str,
    data_specs_version: str,
    external_variables: str,
    frequency: str,
    grid: str,
    grid_label: str,
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
    source_data_notes: Optional[str] = None,
    source_data_retrieval_date: Optional[str] = None,
    source_data_url: Optional[str] = None,
    source_label: str,
    source_type: str,
    source_version_number: str,
    title: Optional[str] = None,
    tracking_id: str,
    variable_id: str,
    variant_label: str,
    variant_info: str,
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

    valid_grid_labels = ["gn", "gr1"]
    valid_products = ["observations", "reanalysis", "in_situ", "exploratory_product"]
    valid_external_variables = ["areacella", "areacello", "volcella", "volcello"]

    def load_json_from_url(url):
        with urllib.request.urlopen(url) as response:
            return json.load(response)

    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    freq_cv = load_json_from_url(base_url + "obs4MIPs_frequency.json")
    institution_cv = load_json_from_url(base_url + "obs4MIPs_institution_id.json")
    nominal_res_cv = load_json_from_url(base_url + "obs4MIPs_nominal_resolution.json")
    realm_cv = load_json_from_url(base_url + "obs4MIPs_realm.json")
    region_cv = load_json_from_url(base_url + "obs4MIPs_region.json")
    source_id_cv = load_json_from_url(base_url + "obs4MIPs_source_id.json")
    source_type_cv = load_json_from_url(base_url + "obs4MIPs_source_type.json")
    top_level_cv = load_json_from_url(base_url + "Tables/obs4MIPs_CV.json")

    errors = []
    # Check vals dependent on "valid_" lists hard-coded above
    if grid_label not in valid_grid_labels:
        errors.append(f"grid_label must be one of {valid_grid_labels}")
    if product not in valid_products:
        errors.append(f"product must be one of {valid_products}")
    if external_variables not in valid_external_variables:
        errors.append(f"external_variables must be one of {valid_external_variables}")

    # Check vals present in Github json files
    if frequency not in freq_cv:
        errors.append("frequency must match a key in obs4MIPs_frequency.json")
    if institution_id not in institution_cv:
        errors.append("institution_id must match a key in obs4MIPs_institution_id.json")
    if nominal_resolution not in nominal_res_cv:
        errors.append(
            "nominal_resolution must match a key in obs4MIPs_nominal_resolution.json"
        )
    if realm not in realm_cv:
        errors.append("realm must match a key in obs4MIPs_realm.json")
    if region not in region_cv:
        errors.append("region must match a key in obs4MIPs_region.json")
    if source_id not in source_id_cv:
        errors.append("source_id must match a key in obs4MIPs_source_id.json")
    if source_type not in source_type_cv:
        errors.append("source_type must match a key in obs4MIPs_source_type.json")

    # Check vals *nested* within Github json files
    if source_label != source_id_cv.get(source_id, {}).get("source_label"):
        errors.append(
            "source_label must match the label inside obs4MIPs_source_id.json[source_id]"
        )
    if source_version_number not in source_id_cv.get(source_id, {}).get(
        "source_version_number", []
    ):
        errors.append("source_version_number must match a value inside source_id entry")
    if source_id not in top_level_cv.get("source_id", {}):
        errors.append(
            "source_id must be present in obs4MIPs_CV.json under 'source_id' key"
        )
    if source not in top_level_cv.get("source_id", {}):
        errors.append(
            "source must match a key in the 'source_id' section of obs4MIPs_CV.json"
        )
    if variable_id not in top_level_cv.get("CMORvar", {}):
        errors.append(
            "variable_id must match a key in the 'CMORvar' section of obs4MIPs_CV.json"
        )

    if errors:
        raise ValueError("\n".join(errors))

    attrs = {
        "activity_id": activity_id,
        "comment": comment,
        "contact": contact,
        "Conventions": conventions,
        "creation_date": creation_date,
        "dataset_contributor": dataset_contributor,
        "data_specs_version": data_specs_version,
        "external_variables": external_variables,  # [“areacella”, “areacello”, “volcella”, “volcello”]
        "frequency": frequency,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_frequency.json
        "grid": grid,
        "grid_label": grid_label,  # ["gn", "gr1"]
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
        "source_id": source_id,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json
        "source_data_notes": source_data_notes,
        "source_data_retrieval_date": source_data_retrieval_date,
        "source_data_url": source_data_url,
        "source_label": source_label,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json (nested source_label)
        "source_type": source_type,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_type.json
        "source_version_number": source_version_number,  # https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json (nested source_version_number)
        "title": title,
        "tracking_id": tracking_id,  # automatically detected by CMOR (hdl:21.14102/<uuid>)
        "variable_id": variable_id,  # https://clipc-services.ceda.ac.uk/dreq/index/CMORvar.html
        "variant_label": variant_label,  # same as source_id (if prepped by them), or "BE" if source_id unknown, ensemble member identified via -r1, -r2, ..., -rN
        "variant_info": variant_info,  # description of who prepared the data, describe obs data variance if applicable
    }

    missing = [k for k, v in attrs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    ds.attrs.update(attrs)
    return ds
