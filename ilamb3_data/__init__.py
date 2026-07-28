import datetime
import fnmatch
import json
import os
import re
import urllib.request
import uuid
import warnings
import zipfile
from pathlib import Path
from typing import Optional
from urllib.parse import unquote, urljoin, urlparse

import cftime as cf
import numpy as np
import pandas as pd
import pooch
import requests
import s3fs
import xarray as xr
from cf_units import Unit
from intake_esgf import ESGFCatalog
from tqdm import tqdm


def create_registry(registry_file: str) -> pooch.Pooch:
    """
    Given registry file, return the pooch ilamb reference data catalog.
    Returns: The intake ilamb reference data catalog (pooch.Pooch)
    """

    registry = pooch.create(
        path=pooch.os_cache("ilamb3"),
        base_url="https://www.ilamb.org/ilamb3-data",
        version="0.1",
        env="ILAMB_ROOT",
    )
    registry.load_registry(registry_file)
    return registry


def download_from_html(
    remote_source: str,
    local_source: Path | str | None = None,
    *,
    pattern: str | None = None,
) -> Path | list[Path]:
    """
    Download a file, or files matching a wildcard, from a remote URL.

    When ``pattern`` is provided, ``remote_source`` must be an HTML page with
    links and ``local_source`` is treated as a destination directory. Matching
    uses shell-style wildcards (for example, ``*.zip``), and a list of
    downloaded or extracted paths is returned.

    If the "content-length" header is missing, it falls back to a simple download.
    If the downloaded file is a .zip, it is extracted into a directory of the same name
    and the extraction directory is returned.
    Spaces in the filename are replaced with underscores.
    """
    CHUNKSIZE = 2**20  # 1 [Mb]
    if not isinstance(remote_source, str):
        raise TypeError(
            f"remote_source must be a str, not {type(remote_source).__name__}"
        )
    if pattern is not None:
        if not local_source:
            raise ValueError("local_source must be a directory when pattern is used")

        destination = Path(local_source)
        destination.mkdir(parents=True, exist_ok=True)
        page = requests.get(remote_source)
        page.raise_for_status()
        from bs4 import BeautifulSoup

        soup = BeautifulSoup(page.text, "html.parser")
        urls = [
            urljoin(remote_source, href)
            for link in soup.find_all("a", href=True)
            if (href := link.get("href"))
            and fnmatch.fnmatch(Path(urlparse(href).path).name, pattern)
        ]

        # Preserve listing order while ignoring duplicate links.
        urls = list(dict.fromkeys(urls))
        if not urls:
            raise FileNotFoundError(
                f"No files matching {pattern!r} found at {remote_source}"
            )
        downloaded_files = []
        for url in urls:
            downloaded_file = download_from_html(
                url,
                destination / Path(unquote(urlparse(url).path)).name,
            )
            if isinstance(downloaded_file, list):
                raise TypeError("Expected a single downloaded file")
            downloaded_files.append(downloaded_file)
        return downloaded_files

    if not local_source:
        unquoted_url = unquote(urlparse(remote_source).path)
        filename = Path(unquoted_url).name
        local_source = Path(filename.replace(" ", "_"))
    else:
        local_source = Path(local_source)
        local_source = local_source.with_name(local_source.name.replace(" ", "_"))

    extract_dir = local_source.with_suffix("")
    if local_source.is_file():
        if zipfile.is_zipfile(local_source):
            if not extract_dir.is_dir():
                with zipfile.ZipFile(local_source, "r") as zf:
                    zf.extractall(extract_dir)
            return extract_dir
        return local_source

    resp = requests.get(remote_source, stream=True)
    resp.raise_for_status()
    total_size = [
        int(value)
        for key, value in resp.headers.items()
        if key.lower() == "content-length"
    ]
    total_size = max(total_size) if total_size else 0

    with open(str(local_source), "wb") as fdl:
        if total_size:
            with tqdm(
                total=total_size, unit="B", unit_scale=True, desc=str(local_source)
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=CHUNKSIZE):
                    if chunk:
                        fdl.write(chunk)
                        pbar.update(len(chunk))
        else:
            for chunk in resp.iter_content(chunk_size=CHUNKSIZE):
                if chunk:
                    fdl.write(chunk)

    if zipfile.is_zipfile(local_source):
        with zipfile.ZipFile(local_source, "r") as zf:
            zf.extractall(extract_dir)
        return extract_dir

    return local_source


def download_from_s3(
    remote_source: str,
    local_source: Path | str,
    *,
    pattern: str = "*",
    anon: bool = True,
) -> list[Path]:
    """Download objects matching a wildcard from an S3 bucket or prefix."""
    if not isinstance(remote_source, str):
        raise TypeError(
            f"remote_source must be a str, not {type(remote_source).__name__}"
        )

    destination = Path(local_source)
    destination.mkdir(parents=True, exist_ok=True)
    remote_pattern = f"{remote_source.rstrip('/')}/{pattern}"
    filesystem = s3fs.S3FileSystem(anon=anon)
    remote_files = filesystem.glob(remote_pattern)
    if not remote_files:
        raise FileNotFoundError(f"No files matching {remote_pattern!r}")

    local_files = []
    for remote_file in remote_files:
        local_file = destination / Path(remote_file).name.replace(" ", "_")
        if not local_file.is_file():
            filesystem.get(remote_file, str(local_file))
        local_files.append(local_file)
    return local_files


def download_from_arcgis_rest(
    remote_source: str,
    local_source: Path | str,
    *,
    where: str | list[str] = "1=1",
    out_fields: str = "*",
    out_sr: int = 4326,
    timeout: int = 180,
) -> Path:
    """Query an ArcGIS REST feature layer and save the results as GeoJSON."""
    if not isinstance(remote_source, str):
        raise TypeError(
            f"remote_source must be a str, not {type(remote_source).__name__}"
        )

    local_source = Path(local_source)
    if local_source.is_file():
        return local_source

    queries = [where] if isinstance(where, str) else where
    if not queries:
        raise ValueError("where must contain at least one query")

    features = []
    query_url = f"{remote_source.rstrip('/')}/query"
    for query in queries:
        response = requests.get(
            query_url,
            params={
                "where": query,
                "outFields": out_fields,
                "returnGeometry": "true",
                "outSR": out_sr,
                "f": "geojson",
            },
            timeout=timeout,
        )
        response.raise_for_status()
        result = response.json()
        if "error" in result:
            error = result["error"]
            message = error.get("message", "Unknown ArcGIS REST error")
            details = "; ".join(error.get("details", []))
            if details:
                message = f"{message}: {details}"
            raise RuntimeError(message)
        if result.get("type") != "FeatureCollection" or not isinstance(
            result.get("features"), list
        ):
            raise ValueError("ArcGIS REST response is not a GeoJSON FeatureCollection")
        features.extend(result["features"])

    local_source.parent.mkdir(parents=True, exist_ok=True)
    local_source.write_text(
        json.dumps({"type": "FeatureCollection", "features": features}),
        encoding="utf-8",
    )
    return local_source


def download_from_zenodo(record: dict, download_dir: str):
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


def create_output_filename(attrs: dict) -> str:
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


def get_cmip6_variable_info(
    search_key: str, variable_id: str | None = None
) -> dict[str, str]:
    """
    Given a CMIP6 variable_id, return a dictionary of its standard_name, long_name, and units.
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
        raise ValueError("variable_id must be provided to get specific variable info.")


def time_bounds_from_frequency(
    freq: str, sdate: cf.datetime, edate: cf.datetime
) -> np.ndarray:
    """
    Build CF-style time bounds [start, end) for a given frequency and start/end dates.
    Returns: np.ndarray of shape (ntime, 2), dtype=object, with cftime.DatetimeGregorian.
    """
    # Validate inputs
    if edate < sdate:
        raise ValueError("edate must be >= sdate.")

    # Convert cftime -> pandas Timestamp
    def to_ts(dt):
        return pd.Timestamp(
            dt.year,
            dt.month,
            dt.day,
            getattr(dt, "hour", 0),
            getattr(dt, "minute", 0),
            getattr(dt, "second", 0),
            getattr(dt, "microsecond", 0),
        )

    s_ts, e_ts = to_ts(sdate), to_ts(edate)

    # Validate/normalize frequency
    try:
        offset = pd.tseries.frequencies.to_offset(freq)
    except ValueError as e:
        raise ValueError(f"Invalid freq {freq!r}: {e}") from e
    if offset is None:
        raise ValueError(f"Invalid freq {freq!r}")

    # Periods covering [sdate, edate] (inclusive of both periods that contain them)
    periods = pd.period_range(start=s_ts, end=e_ts, freq=freq)
    if len(periods) == 0:
        raise ValueError("No periods found for the given bounds and frequency.")

    starts = periods.start_time
    ends = (periods + 1).start_time  # ends[i] == starts[i+1]

    # pandas Timestamp -> cftime.DatetimeGregorian
    def ts_to_cf(ts):
        return cf.DatetimeGregorian(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond
        )

    bnds = np.array(
        [(ts_to_cf(lo), ts_to_cf(hi)) for lo, hi in zip(starts, ends)], dtype=object
    )
    return bnds


def climatology_bounds_from_frequency(
    freq: str,
    sdate: cf.datetime,
    edate: cf.datetime,
) -> np.ndarray:
    """
    Build CF-style climatology bounds [start, end) for a *single* cycle of `freq`.
    Starts use the first climatology year (sdate.year); ends use the last (edate.year),
    with +1 year for wrap-around (e.g., Dec->Jan). Matches CF §7.4 semantics.
    """
    if edate < sdate:
        raise ValueError("edate must be >= sdate.")

    # cftime -> pandas Timestamp (we only need year/month/day)
    def to_ts(dt):
        return pd.Timestamp(
            dt.year,
            dt.month,
            dt.day,
            getattr(dt, "hour", 0),
            getattr(dt, "minute", 0),
            getattr(dt, "second", 0),
            getattr(dt, "microsecond", 0),
        )

    s_ts, e_ts = to_ts(sdate), to_ts(edate)

    try:
        _ = pd.tseries.frequencies.to_offset(freq)
    except ValueError as e:
        raise ValueError(f"Invalid freq {freq!r}: {e}") from e

    # Build ONE template year of periods, then transplant years for bounds
    # Use a non-leap anchor year; monthly/seasonal work fine with 2001.
    base_start = pd.Timestamp(2001, 1, 1)
    base_end = pd.Timestamp(2001, 12, 31, 23, 59, 59)
    periods = pd.period_range(start=base_start, end=base_end, freq=freq)
    if len(periods) == 0:
        raise ValueError("No periods found for the given frequency within a year.")

    starts = periods.start_time
    ends = (periods + 1).start_time  # [start, next_start)

    def rep_year(ts: pd.Timestamp, year: int) -> pd.Timestamp:
        # Safe because period.start_time is always a valid first-of-period timestamp
        return ts.replace(year=year)

    rep_bounds = []
    first_year = s_ts.year  # lower column: first climatology year
    last_year = e_ts.year  # upper column: last climatology year (or +1 on wrap)
    for s_i, e_i in zip(starts, ends):
        s_rep = rep_year(s_i, first_year)
        end_year = last_year
        # If end-of-bin is "earlier" in the year than start-of-bin, it wraps into next year
        if (e_i.month, e_i.day, e_i.hour, e_i.minute, e_i.second, e_i.microsecond) < (
            s_i.month,
            s_i.day,
            s_i.hour,
            s_i.minute,
            s_i.second,
            s_i.microsecond,
        ):
            end_year = last_year + 1
        e_rep = rep_year(e_i, end_year)
        rep_bounds.append((s_rep, e_rep))

    # pandas Timestamp -> cftime.DatetimeGregorian
    def ts_to_cf(ts: pd.Timestamp) -> cf.DatetimeGregorian:
        return cf.DatetimeGregorian(
            ts.year, ts.month, ts.day, ts.hour, ts.minute, ts.second, ts.microsecond
        )

    bnds = np.array(
        [(ts_to_cf(lo), ts_to_cf(hi)) for lo, hi in rep_bounds], dtype=object
    )
    return bnds


def build_time(
    sdate: cf.datetime,
    edate: cf.datetime,
    freq: str,
    ref_date: cf.datetime | None = None,
    climatology: bool = False,
) -> xr.DataArray:
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
            bnds = climatology_bounds_from_frequency(freq, sdate, edate)
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
            bnds = time_bounds_from_frequency(freq, sdate, edate)
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


def cf_to_num(
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


def set_time_attrs(
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
    See docstring in your version; adds support for bounds_frequency='fx':
    - time = midpoint(sdate, edate)
    - bounds = [[sdate, edate]]
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
        time_da, bnds_da = build_time(
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
                time_da, bnds_da = build_time(
                    clim_sdate, clim_edate, "fx", ref_date, True
                )
            else:
                if clim_sdate is None or clim_edate is None:
                    raise ValueError(
                        "clim_sdate and clim_edate must be provided for climatology=True."
                    )
                time_da, bnds_da = build_time(
                    clim_sdate, clim_edate, bounds_frequency, ref_date, True
                )
        else:
            if fx:
                if sdate is None or edate is None:
                    raise ValueError(
                        "sdate and edate are required when bounds_frequency='fx'."
                    )
                time_da, bnds_da = build_time(sdate, edate, "fx", ref_date, False)
            else:
                time_da, bnds_da = build_time(
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
        ds["climatology_bnds"] = cf_to_num(
            ds["climatology_bnds"], time_units, time_cal, "float64"
        )
    else:
        ds["time_bnds"] = cf_to_num(ds["time_bnds"], time_units, time_cal, "float64")

    ds = ds.assign_coords(time=cf_to_num(ds["time"], time_units, time_cal, "float64"))
    return ds


def set_lat_attrs(ds: xr.Dataset, compression: dict | None = None) -> xr.Dataset:
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


def set_lon_attrs(ds: xr.Dataset, compression: dict | None = None) -> xr.Dataset:
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


def set_depth_attrs(
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


def _validate_standard_name(name: str) -> None:
    if " " in name:
        raise ValueError(f"standard_name '{name}' contains spaces; use underscores.")
    if name != name.lower():
        raise ValueError(f"standard_name '{name}' must be lowercase.")


def _validate_ancillary_variables(ds: xr.Dataset, ancillary_variables: str) -> None:
    missing = [v for v in ancillary_variables.split() if v not in ds]
    if missing:
        warnings.warn(
            f"Ancillary variables not found in dataset: {missing}. "
            f"Make sure to add them before writing."
        )


_VALID_CELL_METHODS = {
    "point",
    "sum",
    "mean",
    "maximum",
    "minimum",
    "mid_range",
    "standard_deviation",
    "variance",
    "mode",
    "median",
}


def _validate_cell_methods(ds: xr.Dataset, cell_methods: str) -> None:
    # pattern: "dim1: method1 dim2: method2 ..."
    pairs = re.findall(r"(\S+):\s+(\S+)", cell_methods)
    if not pairs:
        raise ValueError(
            f"cell_methods '{cell_methods}' doesn't match expected 'dim: method' "
            "format."
        )
    for dim, method in pairs:
        if dim not in ds.dims and dim not in ("area", "time", "where"):
            warnings.warn(
                f"cell_methods references '{dim}' which is not a dimension in the "
                "dataset."
            )
        if method not in _VALID_CELL_METHODS:
            raise ValueError(
                f"cell_methods method '{method}' is not CF-valid. "
                f"Must be one of: {_VALID_CELL_METHODS}"
            )


def _validate_flags(
    flag_values: np.ndarray,
    flag_meanings: list[str],
    dtype: np.dtype,
) -> None:
    if len(flag_values) != len(flag_meanings):
        raise ValueError(
            f"flag_values ({len(flag_values)}) and flag_meanings ({len(flag_meanings)}) "
            f"must have the same length."
        )
    if len(flag_values) != len(set(flag_values)):
        raise ValueError("flag_values must be unique.")
    for meaning in flag_meanings:
        if " " in meaning:
            raise ValueError(
                f"flag_meanings entry '{meaning}' contains spaces; use underscores."
            )
    if not np.can_cast(flag_values, dtype, casting="same_kind"):
        warnings.warn(
            f"flag_values dtype ({flag_values.dtype}) doesn't match "
            f"variable dtype ({dtype}). Values will be cast."
        )


def set_var_attrs(
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

    current_units = da.attrs.get("units", None)
    effective_units = units

    # only convert if there *was* an original units and it differs
    if current_units is not None and current_units != units:
        if convert:
            warnings.warn(f"Converting {var} units from {current_units} to {units}")
            da = convert_units(da, units)
        else:
            warnings.warn(
                f"Variable '{var}' has units '{current_units}', "
                f"requested '{units}'. Keeping existing units."
            )
            effective_units = current_units

    # do some cleaning/validation
    _validate_standard_name(standard_name)
    clean_long_name = re.sub(r"\s*\[[^\]]+\]", "", long_name).strip()
    if not clean_long_name:
        raise ValueError("long_name cannot be empty.")
    if ancillary_variables is not None:
        _validate_ancillary_variables(ds, ancillary_variables)
    if cell_methods is not None:
        _validate_cell_methods(ds, cell_methods)

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
    if flag_values is not None and flag_meanings is not None:
        final_dt = np.dtype(target_dtype) if target_dtype is not None else da.dtype
        _validate_flags(flag_values, flag_meanings, final_dt)
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


def set_coord_bounds(ds: xr.Dataset, coord: str) -> xr.Dataset:
    """
    Compute and attach 1D cell boundaries for `coord` using a 'bnds'=2
    dimension. Ensures float32 output, no _FillValue, and correct ordering.
    """
    # make sure we have a 'bnds' dimension of length 2
    if "nv" in ds.dims:
        ds = ds.rename_dims({"nv": "bnds"})

    # cast and grab values
    ds[coord] = ds[coord].astype(np.float64)
    vals = ds[coord].values
    n = vals.shape[0]

    # build midpoint array
    b = np.empty((n, 2), dtype=np.float64)
    # interior midpoints
    b[1:-1, 0] = 0.5 * (vals[:-2] + vals[1:-1])
    b[1:-1, 1] = 0.5 * (vals[1:-1] + vals[2:])
    # edge extrapolation
    d0 = vals[1] - vals[0]
    dn = vals[-1] - vals[-2]
    b[0] = [vals[0] - d0 / 2, vals[0] + d0 / 2]
    b[-1] = [vals[-1] - dn / 2, vals[-1] + dn / 2]

    # sort so column 0 is lower bound, column 1 is upper
    b = np.sort(b, axis=1)

    # assign into the dataset
    name = f"{coord}_bnds"
    ds[name] = ((coord, "bnds"), b)

    # reset encoding and attrs
    ds[name].encoding = {"_FillValue": None, "dtype": "float64"}
    ds[coord].attrs["bounds"] = name
    ds[coord].encoding = {"_FillValue": None, "dtype": "float64"}

    return ds


def gen_utc_timestamp(time: float | None = None) -> str:
    if time is None:
        time = datetime.datetime.now(datetime.UTC)
    else:
        time = datetime.datetime.fromtimestamp(time)
    return time.strftime("%Y-%m-%dT%H:%M:%SZ")


def gen_trackingid() -> str:
    return "hdl:21.14102/" + str(uuid.uuid4())


def standardize_dim_order(ds, order=("time", "depth", "lat", "lon", "bnds")):
    return ds.transpose(*order, missing_dims="ignore")


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


def set_ods26_global_attrs(
    ds: xr.Dataset,
    *,
    activity_id: str = "obs4MIPs",
    aux_uncertainty_id: str = "N/A",
    comment: str | None = None,
    contact: str = "N/A",  # First Last (email)
    Conventions: str = "CF-1.12 ODS-2.6",
    creation_date: str = "N/A",
    dataset_contributor: str | None = None,
    data_specs_version: str = "2.6",
    doi: str | None = None,
    frequency: str = "N/A",
    grid: str = "N/A",
    grid_label: str = "N/A",
    has_aux_unc: str = "FALSE",  # must be TRUE or FALSE
    history: str | None = None,
    institution: str = "N/A",
    institution_id: str = "N/A",
    license: str = "N/A",
    nominal_resolution: str = "N/A",
    processing_code_location: str = "N/A",
    product: str = "N/A",
    realm: str = "N/A",
    references: str = "N/A",
    region: str = "N/A",
    site_id: str = "N/A",
    site_location: str = "N/A",
    source: str = "N/A",
    source_data_retrieval_date: str | None = None,
    source_data_url: str = "N/A",
    source_id: str = "N/A",
    source_label: str = "N/A",
    source_type: str = "N/A",
    source_version_number: str = "N/A",
    table_id: str = "N/A",
    title: str | None = None,
    tracking_id: str = "N/A",
    variable_id: str = "N/A",
    variant_label: str = "N/A",
    variant_info: str | None = None,
    version: str | None = None,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12 and ODS-2.6.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to None.


    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        activity_id (str, optional): Name of the MIP activity the dataset is part of, default of "obs4MIPs".
        aux_uncertainty_id (str, optional): Identifier for auxiliary uncertainty data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_aux_uncertainty_id.json.
        comment (str, optional): Miscellaneous information about the data or methods used.
        contact (str): Contact information for the dataset written as <First Last (email)>.
        Conventions (str, optional): Name of the conventions followed by the dataset, default of "CF-1.12 ODS-2.6".
        creation_date (str): Date the dataset was created in the format <YYYY-MM-DDThh:mm:ssZ>.
        dataset_contributor (str, optional): Name of the individual or organization contributing the dataset.
        data_specs_version (str, optional): Version of the data specifications followed, default of "2.6".
        doi (str, optional): Digital Object Identifier for the dataset including the "https://doi.org/" prefix.
        frequency (str, optional): Temporal frequency of the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_frequency.json.
        grid (str, optional): Description of the horizontal grid and/or regridding procedure with no standard format. When necessay, provide brief description of native grid and resolution, and if data have been re-gridded, the re-gridding procedure and description of target grid.
        grid_label (str, optional): Label identifying the grid, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_grid_label.json.
        has_aux_unc (str, optional): Indicates if auxiliary uncertainty data is included, must be "TRUE" or "FALSE", default of "FALSE".
        history (str, optional): List of applications/dates that have modified the original data and how.
        institution (str, optional): Where the original data were produced, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json. If not in the JSON already, create one like <Institution Name, City, State, Country>.
        institution_id (str, optional): Identifier for the institution producing the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json. If not in the JSON already, create one as an abbreviation of the institution(s) name(s). E.g., "NASA-JPL" or "EmoryU".
        license (str, optional): License under which the data is shared, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_license.json.
        nominal_resolution (str, optional): Nominal spatial resolution (similar spatial resolution in km) of the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_nominal_resolution.json.
        processing_code_location (str, optional): URL or DOI pointing to the script used to process the data.
        product (str, optional): Type of product, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_product.json.
        realm (str, optional): Earth system domain of the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_realm.json.
        references (str, optional): Reference(s) describing the data or methods used to produce the data.
        region (str, optional): Geographic region covered by the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_region.json.
        site_id (str, optional): Identifier for the observation site, if one site, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_site_id.json to guide formatting. If multiple sites, set as "collection".
        site_location (str, optional): Description of the observation site location(s). If multiple sites, set as "collection".
        source (str, optional): Description of how the data were produced; could just be an expansion of abbreviations in source_id.
        source_data_retrieval_date (str, optional): Date the original source data were retrieved in the format <YYYY-MM-DD>.
        source_data_url (str, optional): URL where the original source data can be found.
        source_id (str, optional): Identifier for the source of the data, formatted as an abbreviation + version (e.g., GFED-5-0). See https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json.
        source_label (str, optional): Short label for the source of the data, formatted as an abbreviation (e.g., GFED). See "source_label" inside https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json.
        source_type (str, optional): Type of source that produced the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_type.json.
        source_version_number (str, optional): Version number of the source that produced the data (e.g., "5", "ver1.2", "v1", etc).
        table_id (str, optional): Identifier for the CMOR table the data would fit into, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_table_id.json.
        title (str, optional): A short description of the file contents, usually including what variable is stored (e.g., "Harmonized World Soil Database v2.0 soil carbon").
        tracking_id (str, optional): Unique identifier for the dataset instance, can be generated using gen_trackingid().
        variable_id (str, optional): Identifier for the variable stored in the file, see https://clipc-services.ceda.ac.uk/dreq/index/CMORvar.html or https://clipc-services.ceda.ac.uk/dreq/mipVars.html.
        variant_label (str, optional): Label indicating the party that prepared the obs4MIPs data, if different from the source_id.
        variant_info (str, optional): Description of the party that prepared the obs4MIPs data, if variant_label is not the source_id.
        version (str, optional): Version of the dataset instance created by the data contributor, formatted as "vYYYYMMDD".

    Returns:
        xr.Dataset: The dataset with updated global attributes.
    """

    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    # Load controlled vocabularies from online JSON files
    # auxuncid_cv = load_json_from_url(base_url + "obs4MIPs_aux_uncertainty_id.json")
    freq_cv = load_json_from_url(base_url + "obs4MIPs_frequency.json")
    gridlabel_cv = load_json_from_url(base_url + "obs4MIPs_grid_label.json")
    # hasauxunc_cv = load_json_from_url(base_url + "obs4MIPs_has_aux_unc.json")
    instid_cv = load_json_from_url(base_url + "obs4MIPs_institution_id.json")
    nomres_cv = load_json_from_url(base_url + "obs4MIPs_nominal_resolution.json")
    product_cv = load_json_from_url(base_url + "obs4MIPs_product.json")
    realm_cv = load_json_from_url(base_url + "obs4MIPs_realm.json")
    region_cv = load_json_from_url(base_url + "obs4MIPs_region.json")
    reqattrs_cv = load_json_from_url(
        base_url + "obs4MIPs_required_global_attributes.json"
    )
    siteid_cv = load_json_from_url(base_url + "obs4MIPs_site_id.json")
    sourceid_cv = load_json_from_url(base_url + "obs4MIPs_source_id.json")
    sourcetype_cv = load_json_from_url(base_url + "obs4MIPs_source_type.json")
    # tableid_cv = load_json_from_url(base_url + "obs4MIPs_table_id.json")

    # Fill in required global attributes
    attrs = {
        "activity_id": activity_id,
        "aux_uncertainty_id": aux_uncertainty_id,
        "contact": contact,
        "Conventions": Conventions,
        "creation_date": creation_date,
        "data_specs_version": data_specs_version,
        "frequency": frequency,
        "grid": grid,
        "grid_label": grid_label,
        "has_aux_unc": has_aux_unc,
        "institution": institution,
        "institution_id": institution_id,
        "license": license,
        "nominal_resolution": nominal_resolution,
        "processing_code_location": processing_code_location,
        "product": product,
        "realm": realm,
        "references": references,
        "region": region,
        "site_id": site_id,
        "site_location": site_location,
        "source": source,
        "source_data_url": source_data_url,
        "source_id": source_id,
        "source_label": source_label,
        "source_type": source_type,
        "source_version_number": source_version_number,
        "table_id": table_id,
        "tracking_id": tracking_id,
        "variable_id": variable_id,
        "variant_label": variant_label,
    }

    # Add optional attributes if provided
    optional_attrs = {
        "comment": comment,
        "dataset_contributor": dataset_contributor,
        "doi": doi,
        "history": history,
        "source_data_retrieval_date": source_data_retrieval_date,
        "title": title,
        "variant_info": variant_info,
        "version": version,
    }
    for key, value in optional_attrs.items():
        if value is not None:
            attrs[key] = value

    # Sort the keys alphabetically
    attrs = dict(sorted(attrs.items()))

    # Validate required attributes
    missing = []
    for attr in reqattrs_cv["required_global_attributes"]:
        if attr not in attrs or attrs[attr] is None:
            missing.append(attr)
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    # Validate controlled vocabularies
    cv_checks = {
        # "aux_uncertainty_id": auxuncid_cv["aux_uncertainty_id"],
        "frequency": freq_cv["frequency"],
        "grid_label": gridlabel_cv["grid_label"],
        # "has_aux_unc": hasauxunc_cv["has_aux_unc"],
        "institution_id": instid_cv["institution_id"],
        "nominal_resolution": nomres_cv["nominal_resolution"],
        "product": product_cv["product"],
        "realm": realm_cv["realm"],
        "region": region_cv["region"],
        "site_id": siteid_cv["site_id"],
        "source_id": sourceid_cv["source_id"],
        "source_type": sourcetype_cv["source_type"],
    }
    for attr, cv_obj in cv_checks.items():
        if attrs[attr] not in cv_obj:
            # Warn but don't fail
            warnings.warn(
                f"Attribute '{attr}' has value '{attrs[attr]}' "
                f"which is not in the controlled vocabulary."
            )
            continue

    # set global attributes
    ds.attrs = attrs
    return ds


def set_regions_global_attrs(
    ds: xr.Dataset,
    creation_date: str,
    dataset_contributor: str,
    frequency: str,
    grid: str,
    grid_label: str,
    history: str,
    institution: str,
    institution_id: str,
    license: str,
    nominal_resolution: str,
    references: str,
    region: str,
    source: str,
    source_data_retrieval_date: str,
    source_id: str,
    title: str,
    variable_id: str,
    version: str,
    *,
    activity_id: str = "ILAMB",
    comment: str | None = None,
    conventions: str = "CF-1.12",
    contact: str | None = None,
    processing_code_location: str | None = None,
    source_data_url: str | None = None,
    source_version_number: str | None = None,
    tracking_id: str | None = None,
    variant_label: str | None = None,
) -> xr.Dataset:
    """
    Set NetCDF global attributes according to a hybrid of CF-Conventions 1.12. This also
    blends some useful elements from ODS-2.6 standards and adds some additional custom
    attributes specific to the ILAMB project.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to
    None.

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        activity_id (str): An ID assigned to the activity dictating the creation of the
            dataset.
        comment (str): Any comments related to the dataset.
        contact (str): The First, Last name and (email) of the person or organization
            that generated the dataset.
        conventions (str): The conventions & versions that the NetCDF are aligned to.
        creation_date (str): The date when the dataset was created.
        dataset_contributor (str): The name of the person or organization that prepared
            the NetCDF.
        frequency (str): The temporal frequency of the dataset.
        grid (str): A description of the spatial grid of the dataset, including any
            transformations.
        grid_label (str): A label identifying the grid used in the dataset.
        history (str): A record of the NetCDF processing history.
        institution (str): The name of the institution responsible for generating the
            original dataset.
        institution_id (str): The identifier of the institution responsible for
            generating the original dataset.
        license (str): The license under which the dataset is distributed.
        nominal_resolution (str): The nominal spatial resolution of the dataset.
        processing_code_location (str): The URL or path to the code used for processing
            the dataset.
        references (str): References or citations related to the dataset.
        region (str): The geographical region covered by the dataset.
        source (str): A description of how the dataset was generated.
        source_data_retrieval_date (str): The date when the source data was downloaded.
        source_data_url (str): The URL from which the source data was obtained.
        source_id (str): The identifier of the source dataset, e.g., GFED-1-0.
        source_version_number (str): The version number of the source dataset.
        title (str): The title of the dataset, usually what is displayed on plots.
        tracking_id (str): A unique identifier for this NetCDF.
        variable_id (str): The identifier of the variable within the NetCDF.
        variant_label (str): The label identifying the variant of the dataset, usually
            named after the project or organization generating the dataset.
        version (str): The version of the NetCDF produced.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If any required attribute is missing or not valid.
    """

    # Fill in global attributes
    attrs = {
        "activity_id": activity_id,
        "comment": comment,
        "contact": contact,
        "Conventions": conventions,
        "creation_date": creation_date,
        "dataset_contributor": dataset_contributor,
        "frequency": frequency,
        "grid": grid,
        "grid_label": grid_label,
        "history": history,
        "institution": institution,
        "institution_id": institution_id,
        "license": license,
        "nominal_resolution": nominal_resolution,
        "processing_code_location": processing_code_location,
        "references": references,
        "region": region,
        "source": source,
        "source_data_retrieval_date": source_data_retrieval_date,
        "source_data_url": source_data_url,
        "source_id": source_id,
        "source_version_number": source_version_number,
        "title": title,
        "tracking_id": tracking_id,
        "variable_id": variable_id,
        "variant_label": variant_label,
        "version": version,
    }

    # Set the attrs
    ds.attrs = attrs
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
