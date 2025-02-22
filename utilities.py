import pooch


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
        base_url="https://www.ilamb.org/ILAMB-Data/DATA",
        version="0.1",
        env="ILAMB_ROOT",
    )
    registry.load_registry(registry_file)
    return registry
