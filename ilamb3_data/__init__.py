import os

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


def download_file(remote_source: str, local_source: str | None = None) -> str:
    """
    Download the specified file to a local location.
    """
    if local_source is None:
        local_source = os.path.basename(remote_source)
    if not os.path.isfile(local_source):
        resp = requests.get(remote_source, stream=True)
        total_size = int(resp.headers.get("content-length"))
        with open(local_source, "wb") as fdl:
            with tqdm(
                total=total_size,
                unit="B",
                unit_scale=True,
                desc=local_source,
            ) as pbar:
                for chunk in resp.iter_content(chunk_size=1024):
                    if chunk:
                        fdl.write(chunk)
                        pbar.update(len(chunk))
    return local_source


def get_cmip6_variable_info(variable_id: str) -> dict[str, str]:
    """ """
    df = ESGFCatalog().variable_info(variable_id)
    return {
        key.replace("variable_", ""): val for key, val in df.iloc[0].to_dict().items()
    }


def fix_time(ds: xr.Dataset) -> xr.DataArray:
    assert "time" in ds
    da = ds["time"]
    da.encoding = {"units": "days since 1850-01-01"}
    da.attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
    }
    return da


def fix_lat(ds: xr.Dataset) -> xr.DataArray:
    assert "lat" in ds
    da = ds["lat"]
    da.attrs = {
        "axis": "Y",
        "units": "degrees_north",
        "standard_name": "latitude",
        "long_name": "latitude",
    }
    return da


def fix_lon(ds: xr.Dataset) -> xr.DataArray:
    assert "lon" in ds
    da = ds["lon"]
    da.attrs = {
        "axis": "X",
        "units": "degrees_east",
        "standard_name": "longitude",
        "long_name": "longitude",
    }
    return da
