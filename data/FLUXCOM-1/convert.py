"""
FLUXCOM must be downloaded manually via ftp https://fluxcom.org/CF-Download/
"""

import glob
import time
from pathlib import Path

import cftime as cf
import numpy as np
import xarray as xr
from ilamb3.dataset import compute_cell_measures, convert

import ilamb3_data as ild

# FluxCom provides a land fraction mask which we will incorporate as cell measures for
# ilamb to use in spatial integration.
cm = xr.open_dataset("_raw/landfraction.720.360.nc") * 0.01
cm *= compute_cell_measures(cm)
cm = cm.rename(dict(landfraction="cell_measures", latitude="lat", longitude="lon"))
cm = cm["cell_measures"]
cm.attrs = {"long_name": "land_area", "units": "m2"}

# Loop through the variables and create the datasets.
fluxcom_to_cmip = dict(GPP="gpp", TER="reco", H="hfss", LE="hfls")
preferred_unit = dict(gpp="g m-2 d-1", reco="g m-2 d-1", hfss="W m-2", hfls="W m-2")
generate_stamp = time.strftime("%Y-%m-%d")
for fluxcom, cmip in fluxcom_to_cmip.items():
    # What files are we using?
    files = glob.glob(f"_raw/{fluxcom}.*.nc")
    if not files:
        continue
    download_stamp = time.strftime(
        "%Y-%m-%d", time.localtime(Path(files[0]).stat().st_ctime)
    )

    # Merge into a single dataset
    ds = xr.concat([xr.load_dataset(f) for f in files], data_vars="minimal", dim="time")
    ds = ds.rename_vars({fluxcom: cmip})
    ds["time"] = [cf.DatetimeGregorian(t.dt.year, t.dt.month, 15) for t in ds["time"]]

    # Fix some locations that are all zero for all time
    ds[cmip] = xr.where(
        ~((np.abs(ds[cmip]) < 1e-15).all(dim="time")), ds[cmip], np.nan, keep_attrs=True
    )

    # Handle units, C is not for carbon in unit parsers
    if "gC" in ds[cmip].attrs["units"]:
        ds[cmip].attrs["units"] = ds[cmip].attrs["units"].replace("gC", "g")
    ds = convert(ds, preferred_unit[cmip], cmip)

    # Add measures and bounds
    ds["cell_measures"] = cm
    ds[cmip].attrs["cell_measures"] = "area: cell_measures"
    ds = ild.set_time_attrs(ds, bounds_frequency="M")
    ds = ild.set_lat_attrs(ds)
    ds = ild.set_lon_attrs(ds)
    ds = ild.set_coord_bounds(ds, "lat")
    ds = ild.set_coord_bounds(ds, "lon")
    ds = ild.standardize_dim_order(ds)
    tracking_id = ild.gen_trackingid()

    attrs = {key.lower(): value for key, value in ds.attrs.items()}
    ds = ild.set_ods26_global_attrs(
        ds,
        comment=", ".join(
            [
                f"{key}={attrs[key]}"
                for key in [
                    "forcing",
                    "method",
                    "flux",
                    "setup",
                    "type",
                    "meteorolgical_data_meteo",
                    "machine_learning_method_mlm",
                    "energy_balance_correction_ebc",
                ]
                if key in attrs
            ]
        ),
        contact=attrs["provided_by"]
        .split(" on ")[0]
        .replace("[", "(")
        .replace("]", ")"),
        creation_date=generate_stamp,
        dataset_contributor="Nathan Collier",
        doi="10.1038/s41597-019-0076-8",
        frequency="mon",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="FALSE",
        history=f"""
{download_stamp}: downloaded {[Path(f).name for f in files]};
{generate_stamp}: converted to netCDF, additionally we apply a mask where |var|<1e-15 for all time.""",
        institution="Department Biogeochemical Integration, Max Planck Institute for Biogeochemistry, Germany",
        institution_id="MPI-BGC-BGI",
        license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution - 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
        nominal_resolution="50 km",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/FLUXCOM-1/convert.py",
        product="derived",
        realm="land",
        references=attrs["reference"],
        region="global_land",
        source="globally distributed eddy-covariance-based estimates of carbon/energy fluxes between the biosphere and the atmosphere",
        source_id="FLUXCOM-1",
        source_data_retrieval_date=download_stamp,
        source_data_url="https://fluxcom.org/CF-Download/",
        source_label="FLUXCOM",
        source_type="AI_upscaling",
        source_version_number="1",
        title=attrs["title"].strip(),
        tracking_id=tracking_id,
        variable_id=cmip,
        variant_label="ILAMB",
        variant_info="CMORized product prepared by ILAMB",
        version=f"v{generate_stamp.replace('-', '')}",
    )
    out_path = ild.create_output_filename(ds.attrs)
    ds.to_netcdf(out_path, encoding={cmip: {"zlib": True}})
