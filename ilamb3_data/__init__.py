import pooch

from . import bounds, depth, download, global_attrs, lat, lon, output, time, variable


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
