"""Utilities for finding and downloading source data.

Public functions use ``destination`` for local output and return the path or
paths they found there. Existing files are reused instead of downloaded again.
"""

from __future__ import annotations

import fnmatch
import inspect
import json
import warnings
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urljoin, urlparse
from xml.etree import ElementTree

import requests
import s3fs
from bs4 import BeautifulSoup
from tqdm.auto import tqdm

CHUNK_SIZE = 2**20


class ExistingDownloadWarning(UserWarning):
    """Warn that a local download was found and will be reused."""


def _default_destination() -> Path:
    """Return ``_raw`` beside the Python file that called this module.

    Interactive sessions do not have a caller file, so they fall back to an
    ``_raw`` directory in the current working directory.
    """
    module_file = Path(__file__).resolve()
    frame = inspect.currentframe()
    try:
        frame = frame.f_back if frame is not None else None
        while frame is not None:
            caller_file = frame.f_globals.get("__file__")
            if caller_file is None:
                return Path.cwd() / "_raw"
            caller_path = Path(caller_file).resolve()
            if caller_path != module_file:
                return caller_path.parent / "_raw"
            frame = frame.f_back
    finally:
        del frame
    return Path.cwd() / "_raw"


def _safe_name(name: str) -> str:
    """Return a safe (no spaces) filename from a URL path or other string."""
    return Path(unquote(name)).name.replace(" ", "_")


def _reuse(path: Path) -> None:
    """Warn that an existing file is being reused instead of downloaded."""
    warnings.warn(
        f"Already downloaded; using existing file: {path}",
        ExistingDownloadWarning,
        stacklevel=3,
    )


def _download_http(
    source: str,
    destination: Path,
    *,
    timeout: float,
    show_progress: bool,
) -> Path:
    """Stream one HTTP resource to disk without leaving a partial output."""

    # Don't re-download if the destination already exists
    if destination.is_file():
        _reuse(destination)
        return destination

    # Make sure the destination directory exists and stream the download
    destination.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(source, stream=True, timeout=timeout)
    response.raise_for_status()
    total = int(response.headers.get("content-length", 0)) or None
    partial = destination.with_name(f".{destination.name}.part")

    # Stream the response to a temporary file and then rename it to the destination
    try:
        with (
            partial.open("wb") as stream,
            tqdm(
                total=total,
                unit="B",
                unit_scale=True,
                desc=destination.name,
                disable=not show_progress,
            ) as progress,
        ):
            for chunk in response.iter_content(chunk_size=CHUNK_SIZE):
                if chunk:
                    stream.write(chunk)
                    progress.update(len(chunk))
        partial.replace(destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return destination


def _extract_zip(archive: Path) -> Path:
    """Extract a zip archive to a directory with the same name."""
    destination = archive.with_suffix("")
    if destination.is_dir():
        return destination
    destination.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive) as zip_file:
        zip_file.extractall(destination)
    return destination


def _from_http_file(
    source: str,
    destination: Path,
    *,
    timeout: float,
    show_progress: bool,
) -> Path:
    """Download one HTTP file and extract it when it is a zip archive."""
    archive = _download_http(
        source,
        destination,
        timeout=timeout,
        show_progress=show_progress,
    )
    return _extract_zip(archive) if zipfile.is_zipfile(archive) else archive


def from_html(
    source: str,
    destination: Path | str | None = None,
    *,
    pattern: str | None = None,
    timeout: float = 180,
    show_progress: bool = True,
) -> Path | list[Path]:
    """Find and download a file, or matching files, over HTTP.

    Parameters
    ----------
    source : str
        A file URL, or an HTML listing URL when ``pattern`` is supplied.
    destination : Path or str, optional
        Output file for a file URL. For a listing, the directory in which matching
        files are saved. If omitted, files are saved in ``_raw`` beside the calling
        Python script, using remote filenames.
    pattern : str, optional
        Shell-style pattern used to select links, such as ``"*.zip"``.
    timeout : float, default: 180
        HTTP request timeout in seconds.
    show_progress : bool, default: True
        Show byte-level download progress with tqdm.

    Returns
    -------
    Path or list of Path
        A downloaded file (or extraction directory for a zip), or paths from an
        HTML listing.
    """
    # Ensure source is a string
    if not isinstance(source, str):
        raise TypeError(f"source must be a str, not {type(source).__name__}")

    # Download file(s) via HTML listing with pattern matching
    if pattern is not None:
        output_dir = (
            _default_destination() if destination is None else Path(destination)
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        response = requests.get(source, timeout=timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        urls = [
            urljoin(source, href)
            for link in soup.find_all("a", href=True)
            if (href := link.get("href"))
            and fnmatch.fnmatch(Path(urlparse(href).path).name, pattern)
        ]
        urls = list(dict.fromkeys(urls))
        if not urls:
            raise FileNotFoundError(f"No files matching {pattern!r} found at {source}")
        return [
            _from_http_file(
                url,
                output_dir / _safe_name(urlparse(url).path),
                timeout=timeout,
                show_progress=show_progress,
            )
            for url in urls
        ]

    # Download a single file without pattern matching
    if destination is None:
        filename = _safe_name(urlparse(source).path)
        if not filename:
            raise ValueError("destination is required when source has no filename")
        output = _default_destination() / filename
    else:
        output = Path(destination)
    return _from_http_file(
        source,
        output,
        timeout=timeout,
        show_progress=show_progress,
    )


def from_thredds(
    source: str,
    destination: Path | str | None = None,
    *,
    pattern: str = "*",
    timeout: float = 180,
    show_progress: bool = True,
) -> list[Path]:
    """Find and download matching files from a THREDDS catalog.

    Parameters
    ----------
    source : str
        A THREDDS ``catalog.xml`` or ``catalog.html`` URL, or the corresponding
        catalog or file-server directory URL.
    destination : Path or str, optional
        Directory in which matching files are saved. If omitted, files are saved
        in ``_raw`` beside the calling Python script.
    pattern : str, default: "*"
        Shell-style dataset-name pattern, such as ``"*.nc"``.
    timeout : float, default: 180
        HTTP request timeout in seconds.
    show_progress : bool, default: True
        Show byte-level download progress with tqdm.

    Returns
    -------
    list of Path
        Local paths corresponding to every matched catalog dataset.
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be a str, not {type(source).__name__}")

    output_dir = _default_destination() if destination is None else Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize common THREDDS browser and file-server URLs to catalog.xml.
    parsed = urlparse(source)
    path = parsed.path
    if "/fileServer/" in path and path.endswith("/"):
        prefix, directory = path.split("/fileServer/", maxsplit=1)
        catalog_path = f"{prefix}/catalog/{directory}catalog.xml"
    elif "/catalog/" in path and path.endswith("/"):
        catalog_path = f"{path}catalog.xml"
    elif path.endswith("/catalog.html") or path.endswith("/catalog.xml"):
        catalog_path = f"{path.rsplit('/', maxsplit=1)[0]}/catalog.xml"
    else:
        raise ValueError(
            "source must be a THREDDS catalog URL or a catalog/fileServer "
            "directory URL"
        )
    catalog_url = parsed._replace(
        path=catalog_path,
        query="",
        fragment="",
    ).geturl()

    response = requests.get(catalog_url, timeout=timeout)
    response.raise_for_status()

    # Resolve the catalog's HTTP download service and matching datasets.
    root = ElementTree.fromstring(response.content)
    http_base = "/thredds/fileServer/"
    for element in root.iter():
        if (
            element.tag.rsplit("}", maxsplit=1)[-1] == "service"
            and element.attrib.get("serviceType") == "HTTPServer"
        ):
            http_base = element.attrib.get("base", http_base)
            break

    urls = []
    for element in root.iter():
        if element.tag.rsplit("}", maxsplit=1)[-1] != "dataset":
            continue
        url_path = element.attrib.get("urlPath")
        name = element.attrib.get("name")
        if url_path and name and fnmatch.fnmatch(name, pattern):
            remote_path = f"{http_base.rstrip('/')}/{url_path.lstrip('/')}"
            urls.append(urljoin(catalog_url, remote_path))
    urls = list(dict.fromkeys(urls))

    if not urls:
        raise FileNotFoundError(f"No files matching {pattern!r} found at {source}")
    return [
        _from_http_file(
            url,
            output_dir / _safe_name(urlparse(url).path),
            timeout=timeout,
            show_progress=show_progress,
        )
        for url in urls
    ]


def from_s3(
    source: str,
    destination: Path | str | None = None,
    *,
    pattern: str = "*",
    anonymous: bool = True,
    show_progress: bool = True,
) -> list[Path]:
    """Find and download objects from an S3 bucket or prefix.

    Parameters
    ----------
    source : str
        S3 bucket or prefix, for example ``"s3://bucket/prefix"``.
    destination : Path or str, optional
        Directory in which matching objects are saved. If omitted, objects are
        saved in ``_raw`` beside the calling Python script.
    pattern : str, default: "*"
        Shell-style object-name pattern.
    anonymous : bool, default: True
        Use anonymous rather than configured S3 credentials.
    show_progress : bool, default: True
        Show object-level progress with tqdm.

    Returns
    -------
    list of Path
        Local paths corresponding to every matched object.
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be a str, not {type(source).__name__}")
    output_dir = _default_destination() if destination is None else Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    remote_pattern = f"{source.rstrip('/')}/{pattern}"
    filesystem = s3fs.S3FileSystem(anon=anonymous)
    remote_files = filesystem.glob(remote_pattern)
    if not remote_files:
        raise FileNotFoundError(f"No files matching {remote_pattern!r}")

    outputs: list[Path] = []
    for remote_file in tqdm(
        remote_files,
        desc="Downloading S3 objects",
        unit="file",
        disable=not show_progress,
    ):
        output = output_dir / _safe_name(remote_file)
        if output.is_file():
            _reuse(output)
        else:
            partial = output.with_name(f".{output.name}.part")
            try:
                filesystem.get(remote_file, str(partial))
                partial.replace(output)
            except BaseException:
                partial.unlink(missing_ok=True)
                raise
        outputs.append(output)
    return outputs


def from_arcgis_rest(
    source: str,
    destination: Path | str,
    *,
    where: str | list[str] = "1=1",
    out_fields: str = "*",
    out_sr: int = 4326,
    timeout: float = 180,
    show_progress: bool = True,
) -> Path:
    """Query an ArcGIS REST feature layer and save one GeoJSON file.

    ``where`` may contain several queries; their features are combined in query
    order. An existing destination is reused without contacting the service.

    Parameters
    ----------
    source : str
        ArcGIS REST feature-layer URL.
    destination : Path or str
        Output GeoJSON file.
    where : str or list of str, default: "1=1"
        One or more SQL-like ArcGIS selection expressions.
    out_fields : str, default: "*"
        Comma-separated attributes to include.
    out_sr : int, default: 4326
        Spatial reference identifier for output geometries.
    timeout : float, default: 180
        HTTP request timeout in seconds.
    show_progress : bool, default: True
        Show query progress with tqdm when multiple queries are used.

    Returns
    -------
    Path
        Path to the GeoJSON file.
    """
    if not isinstance(source, str):
        raise TypeError(f"source must be a str, not {type(source).__name__}")
    output = Path(destination)
    if output.is_file():
        _reuse(output)
        return output

    queries = [where] if isinstance(where, str) else where
    if not queries:
        raise ValueError("where must contain at least one query")
    features: list[dict[str, Any]] = []
    query_url = f"{source.rstrip('/')}/query"
    for query in tqdm(
        queries,
        desc="Querying ArcGIS",
        unit="query",
        disable=not show_progress or len(queries) == 1,
    ):
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
            raise RuntimeError(f"{message}: {details}" if details else message)
        if result.get("type") != "FeatureCollection" or not isinstance(
            result.get("features"), list
        ):
            raise ValueError("ArcGIS REST response is not a GeoJSON FeatureCollection")
        features.extend(result["features"])

    output.parent.mkdir(parents=True, exist_ok=True)
    partial = output.with_name(f".{output.name}.part")
    try:
        partial.write_text(
            json.dumps({"type": "FeatureCollection", "features": features}),
            encoding="utf-8",
        )
        partial.replace(output)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return output


def from_zenodo(
    record: Mapping[str, Any] | str | int,
    destination: Path | str | None = None,
    *,
    timeout: float = 180,
    show_progress: bool = True,
) -> list[Path]:
    """Find and download every file in a Zenodo record.

    Parameters
    ----------
    record : mapping, str, or int
        A Zenodo API record mapping, or a Zenodo record ID to look up.
    destination : Path or str, optional
        Directory in which record files are saved. If omitted, files are saved in
        ``_raw`` beside the calling Python script.
    timeout : float, default: 180
        HTTP request timeout in seconds.
    show_progress : bool, default: True
        Show byte-level download progress with tqdm.

    Returns
    -------
    list of Path
        Paths corresponding to all downloadable files in the record.

    Examples
    --------
    If the Zenodo record ID is known, pass it directly:

    .. code-block:: python

        from ilamb3_data import download

        files = download.from_zenodo(1234567, destination="_raw")

    To find a record by its dataset title, search the Zenodo API and pass the
    resulting record mapping to this function:

    .. code-block:: python

        import requests

        from ilamb3_data import download

        dataset_title = "Global Fire Emissions Database (GFED5) Burned Area"
        response = requests.get(
            "https://zenodo.org/api/records",
            params={"q": f'title:"{dataset_title}"'},
            timeout=180,
        )
        response.raise_for_status()

        records = response.json()["hits"]["hits"]
        if not records:
            raise FileNotFoundError(
                f"No Zenodo record found for {dataset_title!r}"
            )

        files = download.from_zenodo(records[0], destination="_raw")
    """
    if isinstance(record, (str, int)):
        response = requests.get(
            f"https://zenodo.org/api/records/{record}", timeout=timeout
        )
        response.raise_for_status()
        record_data = response.json()
    elif isinstance(record, Mapping):
        record_data = record
    else:
        raise TypeError("record must be a mapping, str, or int")

    output_dir = _default_destination() if destination is None else Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[Path] = []
    for file_info in record_data.get("files", []):
        name = file_info.get("key") or file_info.get("filename")
        links = file_info.get("links", {})
        url = links.get("download") or links.get("self")
        if not name or not url:
            warnings.warn(
                "Skipping a Zenodo file without both a name and download URL",
                UserWarning,
                stacklevel=2,
            )
            continue
        output = _download_http(
            url,
            output_dir / _safe_name(name),
            timeout=timeout,
            show_progress=show_progress,
        )
        outputs.append(output)
    return outputs


def from_figshare(
    article_id: str | int,
    destination: Path | str | None = None,
    *,
    timeout: float = 180,
    show_progress: bool = True,
) -> list[Path]:
    """Find and download every file associated with a Figshare article.

    Parameters
    ----------
    article_id : str or int
        Figshare article identifier used to look up its files.
    destination : Path or str, optional
        Directory in which article files are saved. If omitted, files are saved in
        ``_raw`` beside the calling Python script.
    timeout : float, default: 180
        HTTP request timeout in seconds.
    show_progress : bool, default: True
        Show byte-level download progress with tqdm.

    Returns
    -------
    list of Path
        Paths corresponding to all files in the article.
    """
    output_dir = _default_destination() if destination is None else Path(destination)
    output_dir.mkdir(parents=True, exist_ok=True)
    response = requests.get(
        f"https://api.figshare.com/v2/articles/{article_id}/files",
        timeout=timeout,
    )
    response.raise_for_status()
    outputs: list[Path] = []
    for file_info in response.json():
        output = _download_http(
            file_info["download_url"],
            output_dir / _safe_name(file_info["name"]),
            timeout=timeout,
            show_progress=show_progress,
        )
        outputs.append(output)
    return outputs


__all__ = [
    "ExistingDownloadWarning",
    "from_arcgis_rest",
    "from_figshare",
    "from_html",
    "from_s3",
    "from_thredds",
    "from_zenodo",
]
