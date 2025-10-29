import datetime
import subprocess
from pathlib import Path

import numpy as np
import xarray as xr

from ilamb3_data import (
    create_output_filename,
    gen_trackingid,
    gen_utc_timestamp,
    get_cmip6_variable_info,
    set_coord_bounds,
    set_lat_attrs,
    set_lon_attrs,
    set_ods_global_attrs,
    set_time_attrs,
    set_var_attrs,
    standardize_dim_order,
)

PURGE = True
remote_source = "https://catalogue.ceda.ac.uk/uuid/f4654030223445b0bac63a23aaa60620/"


def download_average_and_coarsen(
    year: int, month: int, remove_source: bool = False
) -> None:
    # Only redownload and run if monthly mean file doesn't already exist
    monthly_mean_dir = Path("_raw/")
    monthly_mean_dir.mkdir(parents=True, exist_ok=True)
    monthly_mean_file = monthly_mean_dir / f"snc-{year:04d}-{month:02d}.nc"
    if monthly_mean_file.exists():
        print(f"Skipping {monthly_mean_file}, already exists")
        return

    # Download the daily files for a year/month
    downloads_dir = Path(f"downloads/{year:04d}-{month:02d}")
    downloads_dir.mkdir(parents=True, exist_ok=True)

    def fetch_days_in_month(year: int, month: int, outdir: Path):
        url = f"https://dap.ceda.ac.uk/neodc/esacci/snow/data/scfg/CryoClim/v1.0/{year:04d}/{month:02d}/"
        cmd = [
            "wget2",
            "--robots=off",  # disable robots (default is on)
            "--recursive",
            "--level",
            "1",  # one level under the month dir
            "--parent=off",  # don't ascend above the given dir
            "--accept",
            "*.nc",  # only NetCDFs
            "--reject",
            "index.html*",  # skip directory index files
            "--tries",
            "20",
            "--waitretry",
            "5",
            "--retry-connrefused",
            "--retry-on-http-error",
            "429,500,502,503,504",
            "--timeout",
            "30",
            "--dns-timeout",
            "20",
            "--connect-timeout",
            "20",
            "--read-timeout",
            "60",
            "--continue",
            "--timestamping",
            "--max-threads",
            "10",
            "--directories=off",  # <— don't create remote path dirs
            "--host-directories=off",  # <— don't create host dir
            "-P",
            str(outdir),  # directory prefix
            url,
        ]
        subprocess.run(cmd, check=True)
        return sorted(outdir.glob("*.nc"))

    daily_files = fetch_days_in_month(year, month, downloads_dir)

    # Check if any files are broken and cant be read
    good_files = []
    bad_files = []
    for filename in daily_files:
        try:
            with xr.open_dataset(filename) as ds:
                pass
            good_files.append(filename)
        except Exception:
            bad_files.append(filename)
    if bad_files:
        # Remove bad file
        print(f"Removing {len(bad_files)} bad files")
        for filename in bad_files:
            filename.unlink(missing_ok=True)

        # Try to redownload
        daily_files = fetch_days_in_month(year, month, downloads_dir)

        # Check again if any files are still bad
        bad_files = []
        for filename in daily_files:
            try:
                with xr.open_dataset(filename) as ds:
                    pass
            except Exception:
                bad_files.append(filename)

        # Create NaN data file if still bad
        if bad_files:
            print(
                f"WARNING: {len(bad_files)} files still bad after redownload, creating NaN files"
            )
            for filename in bad_files:
                with xr.open_dataset(good_files[0]) as ds:
                    ds = ds.copy()
                ds["scfg"] = xr.full_like(ds["scfg"], fill_value=-9999, dtype="int8")
                ds["scfg_unc"] = xr.full_like(
                    ds["scfg_unc"], fill_value=-9999, dtype="uint8"
                )
                ds.to_netcdf(filename)

    # Average all the daily files into a monthly mean
    ds = xr.open_mfdataset(sorted(daily_files)).mean(dim="time", skipna=True)
    ds = ds.drop_vars(["spatial_ref", "lat_bnds", "lon_bnds"])
    ds = ds.rename_vars({"scfg": "snc", "scfg_unc": "snc_se"})
    ds["snc"] = ds["snc"].astype("int8")
    ds["snc_se"] = ds["snc_se"].astype("uint8")

    # Create time dimension and set time steps to midpoint of month using cftime
    ds = ds.expand_dims("time")
    ds["time"] = xr.date_range(
        start=f"{year:04d}-{month:02d}",
        periods=1,
        freq="M",
        calendar="standard",
        use_cftime=True,
    )

    # Coarsen lat/lon to 0.5 degree resolution
    ds = ds.coarsen(lat=5, lon=5, boundary="trim").mean()
    ds = ds.assign_coords(
        lat=ds["lat"].round(2),
        lon=ds["lon"].round(2),
    )
    ds = ds.sortby(["time", "lat", "lon"])

    ds.to_netcdf(str(monthly_mean_file))
    if remove_source:
        for filename in daily_files:
            filename.unlink()


# Initial pass to create monthly mean files
for year in range(1982, 2020):
    for month in range(1, 13):
        if year == 2019 and month > 6:
            continue
        download_average_and_coarsen(year, month, remove_source=PURGE)

# Set timestamps and tracking id
local_source = next(iter(Path("_raw/").glob("*.nc")))
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

ds = xr.open_mfdataset("_raw/snc*.nc")

# Set variable information
var_info = get_cmip6_variable_info("snc")
ds = set_var_attrs(
    ds,
    var="snc",
    cmip6_units=var_info["variable_units"],
    cmip6_standard_name=var_info["cf_standard_name"],
    cmip6_long_name=var_info["variable_long_name"],
    target_dtype=np.int8,
    convert=False,
    ancillary_variables="snc_se",
    cell_methods="time: mean",
)

# Assign ancillary variables
ds = ds.assign(snc_se=(("time", "lat", "lon"), ds.snc_se.values))
ds.snc_se.attrs = {
    "long_name": f"{ds.snc.attrs['standard_name']} standard_error",
    "cell_methods": "area: standard_error",
}
ds.snc_se.encoding = {
    "_FillValue": None,  # CMOR default
    "dtype": "float32",
}

# Clean up attrs
ds = set_time_attrs(ds, bounds_frequency="M")
ds = set_lat_attrs(ds)
ds = set_lon_attrs(ds)
ds = set_coord_bounds(ds, "lat")
ds = set_coord_bounds(ds, "lon")
ds = standardize_dim_order(ds)

# Set global attributes and export
out_ds = set_ods_global_attrs(
    ds,
    activity_id="obs4MIPs",
    aux_variable_id="snc_se",
    comment="Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet",
    contact="Thomas Nagler (thomas.nagler@enveo.at)",
    conventions="CF-1.12 ODS-2.5",
    creation_date=creation_stamp,
    dataset_contributor="Morgan Steckler",
    data_specs_version="2.5",
    doi="10.5285/f4654030223445b0bac63a23aaa60620",
    external_variables="N/A",
    frequency="mon",
    grid="0.5x0.5 degree latitude x longitude",
    grid_label="gr",
    has_auxdata="True",
    history=f"""
{download_stamp}: downloaded {remote_source};
{creation_stamp}: converted to obs4MIPs format""",
    institution="Norwegian Computing Center, Oslo, NO",
    institution_id="NR",
    license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution- 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",  # OG license: Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International Public License
    nominal_resolution="50 km",
    processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/CCI-CryoClim-FSC-1/convert.py",
    product="observations",
    realm="land",
    references="Solberg, R., Rudjord, Ø., Salberg, A.-B., Killie, M.A., Eastwood, S., Sørensen, A., Marin, C., Premier, V., Schwaizer, G., Nagler, T., 2023: ESA Snow Climate Change Initiative (Snow_cci): Fractional Snow Cover in CryoClim, v1.0. NERC EDS Centre for Environmental Data Analysis, doi:10.5285/f4654030223445b0bac63a23aaa60620",
    region="global_land",
    source="Fractional Snow Cover CDR v1 from ESA CCI / CryoClim",
    source_id="CCI-CryoClim-FSC-1",
    source_data_retrieval_date=download_stamp,
    source_data_url=remote_source,
    source_label="CCI/CryoClim FSC",
    source_type="satellite_retrieval",
    source_version_number="1.0",
    title="ESA CCI CryoClim Fractional Snow Cover (version v1.0)",
    tracking_id=tracking_id,
    variable_id="snc",
    variant_label="REF",
    variant_info="CMORized product prepared by ILAMB and CMIP IPO",
    version=f"v{today_stamp}",
)

# Prep for export
out_path = create_output_filename(ds.attrs)
ds.to_netcdf(out_path, format="NETCDF4")
