from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr

from ilamb3_data import (
    download_file,
    gen_utc_timestamp,
)

DECADE = "B5C2"
YEAR_INITIAL = 2015
YEAR_FINAL = 2022
GRID = "01"
NOAA_NAME = {
    "o": "oxygen",
    "p": "phosphate",
    "i": "silicate",
    "t": "temperature",
    "s": "salinity",
    "n": "nitrate",
}
VARIABLE_NAME = {
    "o": "o2",
    "p": "po4",
    "i": "sio3",
    "t": "thetao",
    "s": "so",
    "n": "no3",
}

# Download sources
remote_source_dict = {}
# Some are defined by decades
for v, noaa in NOAA_NAME.items():
    if v not in ["t", "s"]:
        continue
    remote_source_dict[VARIABLE_NAME[v]] = [
        f"https://www.ncei.noaa.gov/data/oceans/woa/WOA23/DATA/{NOAA_NAME[v]}/netcdf/{DECADE}/1.00/woa23_{DECADE}_{v}{m:02d}_{GRID}.nc"
        for m in range(1, 13)
    ]
# Others are not
for v, noaa in NOAA_NAME.items():
    if v in ["t", "s"]:
        continue
    remote_source_dict[VARIABLE_NAME[v]] = [
        f"https://www.ncei.noaa.gov/thredds-ocean/fileServer/woa23/DATA/{NOAA_NAME[v]}/netcdf/all/1.00/woa23_all_{v}{m:02d}_{GRID}.nc"
        for m in range(1, 13)
    ]
local_source_dict = {}
for vname, remote_sources in remote_source_dict.items():
    local_source_dict[vname] = []
    for remote_source in remote_sources:
        source = Path("_raw") / Path(remote_source).name
        local_source_dict[vname].append(source)
        source.parent.mkdir(parents=True, exist_ok=True)
        if not source.is_file():
            download_file(remote_source, str(source))
        download_stamp = gen_utc_timestamp(source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()

# Define time/climatological bounds
time_range = f"{YEAR_INITIAL}01-{YEAR_FINAL}12"
time = np.array([cf.DatetimeNoLeap(YEAR_INITIAL, m, 15) for m in range(1, 13)])
climatology_bounds = np.asarray(
    [
        [cf.DatetimeNoLeap(YEAR_INITIAL, m, 1) for m in range(1, 13)],
        [
            cf.DatetimeNoLeap(
                YEAR_FINAL if m < 12 else (YEAR_FINAL + 1), 1 if m == 12 else (m + 1), 1
            )
            for m in range(1, 13)
        ],
    ]
).T

# Load the dataset for adjustments
for v, vname in VARIABLE_NAME.items():
    ds = xr.open_mfdataset(sorted(local_source_dict[vname]), decode_times=False)

    # Drop/rename
    ds = ds.drop_vars(
        [
            "crs",
            f"{v}_mn",
            f"{v}_dd",
            f"{v}_sd",
            f"{v}_oa",
            f"{v}_ma",
            f"{v}_gp",
            f"{v}_sdo",
            f"{v}_sea",
        ],
        errors="ignore",
    ).rename_vars({f"{v}_an": vname, f"{v}_se": f"{vname}_se"})

    # Fix time
    ds[vname].attrs["ancillary_variables"] = f"{vname}_se"
    ds["time"] = time
    ds["time"].encoding = {
        "units": f"days since {YEAR_INITIAL}-01-01 00:00:00",
        "calendar": "noleap",
    }
    ds["time"].attrs = {
        "axis": "T",
        "standard_name": "time",
        "long_name": "time",
        "climatology": "climatology_bounds",
    }
    ds["climatology_bounds"] = (("time", "nbounds"), climatology_bounds)
    ds[vname].attrs["cell_methods"] = "time: mean within years time: mean over years"

    # Populate attributes
    ds.attrs = {
        "activity_id": "obs4MIPs",
        "contact": "NOAA National Centers for Environmental Information (ncei.info@noaa.gov)",
        "Conventions": "CF-1.12 ODS-2.5",
        "creation_data": generate_stamp,
        "dataset_contributor": "Nathan Collier",
        "data_specs_version": "2.5",
        "doi": "10.25921/va26-hv25",
        "frequency": "mon",
        "grid": "1x1 degree",
        "grid_label": "gn",
        "history": """
    %s: downloaded %s;
    %s: converted to obs4MIP format"""
        % (
            download_stamp,
            remote_source_dict[vname],
            generate_stamp,
        ),
        "institution": "National Oceanic and Atmospheric Administration",
        "institution_id": "NOAA",
        "license": "CC-BY-4.0",
        "nominal_resolution": "1x1 degree",
        "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/WOA/convert.py",
        "product": "observations",
        "realm": "ocean",
        "references": f"Reagan, James R.; Boyer, Tim P.; García, Hernán E.; Locarnini, Ricardo A.; Baranova, Olga K.; Bouchard, Courtney; Cross, Scott L.; Mishonov, Alexey V.; Paver, Christopher R.; Seidov, Dan; Wang, Zhankun; Dukhovskoy, Dmitry (2023). World Ocean Atlas 2023. {DECADE}. NOAA National Centers for Environmental Information. Dataset. https://doi.org/10.25921/va26-hv25. Accessed {download_stamp}.",
        "region": "global_ocean",
        "source": "In situ temperature, salinity, dissolved oxygen, Apparent Oxygen Utilization (AOU), percent oxygen saturation, phosphate, silicate, and nitrate at standard depth levels",
        "source_id": "WOA",
        "source_data_retrieval_date": download_stamp,
        "source_data_url": "https://www.ncei.noaa.gov/access/metadata/landing-page/bin/iso?id=gov.noaa.nodc:NCEI-WOA23",
        "source_type": "gridded_insitu",
        "source_version_number": "2023",
        "title": f"World Ocean Atlas - {NOAA_NAME[v].capitalize()}",
        "variable_id": vname,
        "variant_label": "BE",
    }

    # Some additional references depending on the variable
    if vname == "thetao":
        ds.attrs["references"] += (
            "\nLocarnini, R.A., A.V. Mishonov, O.K. Baranova, J.R. Reagan, T.P. Boyer, D. Seidov, Z. Wang, H.E. Garcia, C. Bouchard, S.L. Cross, C.R. Paver, and D. Dukhovskoy (2024). World Ocean Atlas 2023, Volume 1: Temperature. A. Mishonov Technical Editor, NOAA Atlas NESDIS 89. https://doi.org/10.25923/54bh-1613"
        )
    if vname == "so":
        ds.attrs["references"] += (
            "\nReagan, J.R., D. Seidov, Z. Wang, D. Dukhovskoy, T.P. Boyer, R.A. Locarnini, O.K. Baranova, A.V. Mishonov, H.E. Garcia, C. Bouchard, S.L. Cross, and C.R. Paver. (2024). World Ocean Atlas 2023, Volume 2: Salinity. A. Mishonov, Technical Editor, NOAA Atlas NESDIS 90. https://doi.org/10.25923/70qt-9574"
        )
    if vname == "o2":
        ds.attrs["references"] += (
            "\nGarcia, H.E., Z. Wang, C. Bouchard, S.L. Cross, C.R. Paver, J.R. Reagan, T.P. Boyer, R.A. Locarnini, A.V. Mishonov, O. Baranova, D. Seidov, and D. Dukhovskoy (2024). World Ocean Atlas 2023, Volume 3: Dissolved Oxygen, Apparent Oxygen Utilization, Dissolved Oxygen Saturation and 30 year Climate Normal. A. Mishonov, Tech. Ed. NOAA Atlas NESDIS 91. https://doi.org/10.25923/rb67-ns53"
        )
    if vname in ["no3", "po4", "sio3"]:
        ds.attrs["references"] += (
            "\nGarcia, H.E., C. Bouchard, S.L. Cross, C.R. Paver, Z. Wang, J.R. Reagan, T.P. Boyer, R.A. Locarnini, A.V. Mishonov, O. Baranova, D. Seidov, and D. Dukhovskoy (2024). World Ocean Atlas 2023, Volume 4: Dissolved Inorganic Nutrients (phosphate, nitrate, silicate). A. Mishonov, Tech. Ed. NOAA Atlas NESDIS 92. https://doi.org/10.25923/39qw-7j08"
        )

    # Write out files
    ds.to_netcdf(
        "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
            **ds.attrs, time_mark=time_range
        ),
        encoding={vname: {"zlib": True}, f"{vname}_se": {"zlib": True}},
    )
