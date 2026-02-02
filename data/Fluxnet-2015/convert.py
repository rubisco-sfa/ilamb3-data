"""This conversion script will automatically download Fluxnet2015 CC-By-4.0 Data and
reformat it into a CF-compliant netCDF file. However, you must still make the data
request manually. First, sign into your Fluxnet account and navigate to:

https://fluxnet.org/data/download-data/

Choose "SUBSET data product", select all sites, and check "BADM Zip File For All
FLUXNET2015 Dataset Sites". The click the "Download All Files" button. It will take some
time, but eventually the site will present you with a series of links you are supposed
to click to download. Instead, save this HTML page as "manifest.html" into the directory
where you will execute this script.
"""

import time
from glob import glob
from pathlib import Path
from zipfile import ZipFile

import cftime as cf
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

import ilamb3_data as ild

RAW_PATH = "_raw"


def get_fluxnet_variable_conversions() -> pd.DataFrame:
    """Define the Fluxnet to CMOR mapping, hard coding units because their csv
    file is silly. You cannot read units out of the table because they are
    different depending on the temporal resolution.
    """
    dfv = pd.DataFrame(
        [
            {"standard_name": name, "cmor": cmor, "fluxnet": fluxnet, "units": units}
            for name, cmor, fluxnet, units in [
                ["ecosystem_respiration", "reco", "RECO_VUT_MEAN", "g m-2 d-1"],
                ["gross_primary_productivity", "gpp", "GPP_VUT_MEAN", "g m-2 d-1"],
                [
                    "ecosystem_respiration standard_error",
                    "reco_stderr",
                    "RECO_VUT_UNCERT",
                    "g m-2 d-1",
                ],
                [
                    "gross_primary_productivity standard_error",
                    "gpp_stderr",
                    "GPP_VUT_UNCERT",
                    "g m-2 d-1",
                ],
                ["latent_heat", "hfls", "LE_F_MDS", "W m-2"],
                ["net_ecosystem_exchange", "nee", "NEE_VUT_REF", "g m-2 d-1"],
                ["precipitation", "pr", "P_F", "mm d-1"],
                ["sensible_heat", "hfss", "H_F_MDS", "W m-2"],
                ["surface_air_temperature", "tas", "TA_F", "degC"],
                ["surface_downward_longwave_radiation", "rlds", "LW_IN_F", "W m-2"],
                ["surface_upward_longwave_radiation", "rlus", "LW_OUT", "W m-2"],
                ["surface_downward_shortwave_radiation", "rsds", "SW_IN_F", "W m-2"],
                ["surface_upward_shortwave_radiation", "rsus", "SW_OUT", "W m-2"],
                ["surface_net_radiation", "rns", "NETRAD", "W m-2"],
            ]
        ]
    )
    return dfv


def get_site_information() -> pd.DataFrame:
    # Process the site info and store
    site_info = Path("site_info.feather")
    if site_info.is_file():
        dfi = pd.read_feather(site_info)
    else:
        excels = [f for f in glob(f"{RAW_PATH}/*.xlsx") if "_MM_" in f]
        df = pd.read_excel(excels[0])
        df = df.rename(columns={"SITE_ID": "site"})
        q = df[(df.VARIABLE == "LOCATION_LAT") | (df.VARIABLE == "LOCATION_LONG")]
        # sites can have repeated locations in the database
        for _, grp in q.groupby("site"):
            q = q.drop(grp.sort_values("GROUP_ID").iloc[2:].index)
        dfi = q.pivot(columns="VARIABLE", index="site", values="DATAVALUE")
        dfi["LOCATION_LAT"] = dfi["LOCATION_LAT"].astype(float)
        dfi["LOCATION_LONG"] = dfi["LOCATION_LONG"].astype(float)
        dfi.to_feather(site_info)
    return dfi


# Download the files listed in "manifest.html", see instructions above.
html = open("manifest.html").read()
soup = BeautifulSoup(html, "html.parser")
links = [link.attrs["href"] for link in soup.find_all("a", {"class": "download-link"})]
raw_path = Path(RAW_PATH)
raw_path.mkdir(parents=True, exist_ok=True)
for link in links:
    local_source = Path(link).name.split("?")[0]
    ild.download_from_html(link, str(raw_path / local_source))

# Unzip just the monthly data
for zipfile in tqdm(glob(f"{RAW_PATH}/*.zip"), desc="Unzipping"):
    csvfile = Path(zipfile.replace("SUBSET", "SUBSET_MM").replace("zip", "csv"))
    with ZipFile(zipfile) as fzip:
        if csvfile.is_file():
            continue
        if [fi for fi in fzip.filelist if fi.filename == csvfile.name]:
            fzip.extract(csvfile.name, path=RAW_PATH)
        else:
            fzip.extractall(path=RAW_PATH)


# Concat all the csv files into a dataframe
csvs = glob(f"{RAW_PATH}/*.csv")
df = []
for csv in tqdm(csvs, desc="Concatenate csv's"):
    site = (csv.split("/")[-1]).split("_")[1]
    dfs = pd.read_csv(csv, na_values=-9999)
    dfs["site"] = site
    dfs = dfs.set_index(["TIMESTAMP", "site"])
    df.append(dfs)
df = pd.concat(df)

# As per https://fluxnet.org/data/fluxnet2015-dataset/variables-quick-start-guide/ we
# will use the difference in partitioning methods as uncertainty for carbon variables.
for var in ["GPP", "RECO"]:
    df[f"{var}_VUT_MEAN"] = 0.5 * (df[f"{var}_DT_VUT_REF"] + df[f"{var}_NT_VUT_REF"])
    df[f"{var}_VUT_UNCERT"] = 0.5 * np.abs(
        (df[f"{var}_DT_VUT_REF"] - df[f"{var}_NT_VUT_REF"])
    )

# Convert the dataframe to a dataset and cleanup
dfv = get_fluxnet_variable_conversions()
dfi = get_site_information()
df = df[dfv["fluxnet"]]
ds = df.to_xarray()
for _, row in dfv.iterrows():
    ds[row["fluxnet"]].attrs = {
        "standard_name": row["standard_name"],
        "units": row["units"],
    }
ds = ds.rename({fluxnet: cmor for fluxnet, cmor in zip(dfv["fluxnet"], dfv["cmor"])})
ds = ds.rename({"TIMESTAMP": "time"})
ds["time"] = [
    cf.DatetimeGregorian(int(str(t.values)[:4]), int(str(t.values)[-2:]), 15)
    for t in ds["time"]
]
ds["site"].attrs["name"] = "Fluxnet site id"
ds = ds.assign_coords(
    dict(
        lat=("site", dfi.loc[ds["site"], "LOCATION_LAT"]),
        lon=("site", dfi.loc[ds["site"], "LOCATION_LONG"]),
    )
)

# Standardize the dimensions
ds = ild.set_time_attrs(ds, bounds_frequency="M")
ds = ild.set_lat_attrs(ds)
ds = ild.set_lon_attrs(ds)

# Define the global attributes
download_stamp = time.strftime(
    "%Y-%m-%d", time.localtime(Path("manifest.html").stat().st_ctime)
)
generate_stamp = time.strftime("%Y-%m-%d")
tracking_id = ild.gen_trackingid()

for varname in tqdm(ds, desc="Writing netcdf files"):
    if "_stderr" in varname or "_bnds" in varname:
        continue
    to_drop = [
        v
        for v in ds.data_vars
        if (varname not in v) and ("time" not in v) and (not v.endswith("_bnds"))
    ]
    out_ds = ds.drop_vars(to_drop)
    uname = f"{varname}_stderr"
    if uname in ds:
        out_ds[varname].attrs["ancillary_variables"] = uname

    has_stderr = "gpp" in varname or "reco" in varname
    out_ds = ild.set_ods26_global_attrs(
        out_ds,
        aux_uncertainty_id="stderr" if has_stderr else "",
        comment=f"{varname}=(DT+NT)/2 and {varname}_stderr = |DT-NT|/2"
        if has_stderr
        else "",
        contact="Fluxnet Support Team (fluxdata-support@fluxdata.org)",
        creation_date=generate_stamp,
        dataset_contributor="Nathan Collier",
        doi="N/A",
        frequency="mon",
        grid="site",
        grid_label="site",
        has_aux_unc="TRUE" if has_stderr else "FALSE",
        history=f"""
{download_stamp}: downloaded using https://fluxnet.org/data/download-data/;
{generate_stamp}: converted to obs4MIP format""",
        institution="The Fluxnet Community",
        institution_id="Fluxnet",
        license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution - 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
        nominal_resolution="site",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/Fluxnet-2015/convert.py",
        product="site-observations",
        realm="land",
        references="Pastorello, Gilberto and Trotta, Carlo and Canfora, Eleonora and Chu, et al., The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data, Scientific Data, 10.1038/s41597-020-0534-3.",
        region="global_land",
        source="Eddy covariance flux tower measurements",
        source_id="Fluxnet-2015",
        source_data_retrieval_date=download_stamp,
        source_data_url="https://fluxnet.org/data/download-data/",
        source_label="Fluxnet",
        source_type="insitu",
        source_version_number="2015",
        title=f"Fluxnet2015 {varname}",
        tracking_id=tracking_id,
        variable_id=varname,
        variant_label="ILAMB",
        variant_info="CMORized product prepared by ILAMB",
        version=f"v{generate_stamp.replace('-', '')}",
    )

    out_path = ild.create_output_filename(out_ds.attrs)
    out_ds.to_netcdf(out_path)
