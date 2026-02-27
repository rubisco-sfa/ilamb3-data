"""This conversion script reformats AmeriFlux FLUXNET CC-By-4.0 data
into a CF-compliant netCDF file. 
Data request and download must be done manually. using your Fluxnet
account at:

https://ameriflux.lbl.gov/data/download-data/

"""
import sys
import os
import time
from glob import glob
from pathlib import Path
from zipfile import ZipFile
import argparse
import cftime as cf
import numpy as np
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm
import pickle 

import ilamb3_data as ild

RAW_PATH = "_raw_20260222"
frequency = "daily"

REQUIRED_VARIABLES = ["RECO_DT_VUT_REF", "RECO_NT_VUT_REF",
                      "GPP_DT_VUT_REF", "GPP_NT_VUT_REF",
                      "LE_F_MDS", 
                      "NEE_VUT_REF",
                      "P_F",
                      "H_F_MDS",
                      "TA_F",
                      "LW_IN_F",
                      "LW_OUT",
                      "SW_IN_F",
                      "SW_OUT",
                      "NETRAD",
                      ]

def get_ameriflux_variable_conversions() -> pd.DataFrame:
    """Define the AmeriFlux to CMOR mapping, hard coding units because their csv
    file is silly. You cannot read units out of the table because they are
    different depending on the temporal resolution.
    """
    dfv = pd.DataFrame(
        [
            {"standard_name": name, "cmor": cmor, "ameriflux": ameriflux, "units": units}
            for name, cmor, ameriflux, units in [
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
        #excels = [f for f in glob(f"{RAW_PATH}/*.xlsx") if "_MM_" in f]
        excels = [f for f in glob(f"{RAW_PATH}/*.xlsx") if "BIF" in f]
        print(f"Reading BADM file: {excels[0]}")
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

def download_data():
   # Download the files listed in "manifest.html", see instructions above.
   html = open("manifest.html").read()
   soup = BeautifulSoup(html, "html.parser")
   links = []
   file_links_div = soup.find(id='file-links')
   if file_links_div:
       for link in file_links_div.find_all('a', href=True):
           links.append(link['href'])
   raw_path = Path(RAW_PATH)
   raw_path.mkdir(parents=True, exist_ok=True)
   for link in links:
       local_source = Path(link).name.split("?")[0]
       ild.download_from_html(link, str(raw_path / local_source))
   # download the Ameriflux BADM file -- this was most up to date
   # version on 02/22/2026
   # BADM file is missing PR-xLA and US-xPU and I had to move the zip
   # files for them out to skip them
   link = "https://ftp.fluxdata.org/.ameriflux_downloads/BADM-SGI/AMF_AA-Flx_FLUXNET-BIF_CCBY4_20250804.xlsx?=jkumar"
   local_source = Path(link).name.split("?")[0]
   ild.download_from_html(link, str(raw_path / local_source))

def unzip_data(frequency=frequency):
    # Unzip just the monthly data
    for zipfile in tqdm(glob(f"{RAW_PATH}/*.zip"), desc="Unzipping"):
        if frequency == "monthly":
            print(f"Extracting MM file -- {zipfile}")
            csvfile = Path(zipfile.replace("FULLSET", "FULLSET_MM").replace("zip", "csv"))
            if csvfile.is_file():
                continue
            else:
                csvfile = Path(zipfile.replace("FLUXNET", "FLUXNET_FLUXMET_MM").replace("zip", "csv"))
        if frequency == "daily":
            csvfile = Path(zipfile.replace("FULLSET", "FULLSET_DD").replace("zip", "csv"))
            if csvfile.is_file():
                continue
            else:
                csvfile = Path(zipfile.replace("FLUXNET", "FLUXNET_FLUXMET_DD").replace("zip", "csv"))
        if frequency == "30min":
            print(f"Extracting HH file -- {zipfile}")
            csvfile = Path(zipfile.replace("FULLSET", "FULLSET_HH").replace("zip", "csv"))
            if csvfile.is_file():
                continue
            else:
                csvfile = Path(zipfile.replace("FLUXNET", "FLUXNET_FLUXMET_HH").replace("zip", "csv"))
        with ZipFile(zipfile) as fzip:
            if csvfile.is_file():
                continue
            if [fi for fi in fzip.filelist if fi.filename == csvfile.name]:
                fzip.extract(csvfile.name, path=RAW_PATH)
            else:
                fzip.extractall(path=RAW_PATH)

def process_data(frequency=frequency):
    # Get BADM file 
    badm_file = [f for f in glob(f"{RAW_PATH}/*.xlsx") if "BIF" in f][0]
    
    # Concat all the csv files into a dataframe
    # As of 02/22/2026 ameriflux files have incosistent names for the
    # files so we will check and read two potential filename patterns
    if frequency == "monthly":
        csvs = glob(f"{RAW_PATH}/*FULLSET_MM*.csv")
        csvs.extend(glob(f"{RAW_PATH}/*FLUXMET_MM*.csv"))
    if frequency == "daily":
        csvs = glob(f"{RAW_PATH}/*FULLSET_DD*.csv")
        csvs.extend(glob(f"{RAW_PATH}/*FLUXMET_DD*.csv"))
    if frequency == "30min":
        csvs = glob(f"{RAW_PATH}/*FULLSET_HH*.csv")
        csvs.extend(glob(f"{RAW_PATH}/*FLUXMET_HH*.csv"))
    
    df = pd.DataFrame() 
    for csv in tqdm(csvs, desc="Concatenate csv's"):
        site = (csv.split("/")[-1]).split("_")[1]
        print(f"reading {csv}")
        dfs = pd.read_csv(csv, na_values=-9999)
        # check if all the REQUIRED_VARIABLES exist, else add and initialze
        # with nan
        for var in REQUIRED_VARIABLES:
            if var not in dfs.columns:
                dfs[var] = np.nan
         
        dfs["site"] = site
        if frequency in ["monthly", "daily"]:
            var_subset = REQUIRED_VARIABLES + ["TIMESTAMP", "site"]
            dfs = dfs[var_subset]
            dfs = dfs.set_index(["TIMESTAMP", "site"])
        if frequency == "30min":
            var_subset = REQUIRED_VARIABLES + ["TIMESTAMP_START", "site"]
            dfs = dfs[var_subset]
            dfs = dfs.set_index(["TIMESTAMP_START", "site"])
        df = pd.concat([df,dfs])
    #df = pd.concat(df)
    
    # As per https://ameriflux.org/data/ameriflux2015-dataset/variables-quick-start-guide/ we
    # will use the difference in partitioning methods as uncertainty for carbon variables.
    for var in ["GPP", "RECO"]:
        df[f"{var}_VUT_MEAN"] = 0.5 * (df[f"{var}_DT_VUT_REF"] + df[f"{var}_NT_VUT_REF"])
        df[f"{var}_VUT_UNCERT"] = 0.5 * np.abs(
            (df[f"{var}_DT_VUT_REF"] - df[f"{var}_NT_VUT_REF"])
        )
    
    # Convert the dataframe to a dataset and cleanup
    dfv = get_ameriflux_variable_conversions()
    dfi = get_site_information()
    df = df[dfv["ameriflux"]]
    ds = df.to_xarray()
    for _, row in dfv.iterrows():
        ds[row["ameriflux"]].attrs = {
            "standard_name": row["standard_name"],
            "units": row["units"],
        }
    ds = ds.rename({ameriflux: cmor for ameriflux, cmor in zip(dfv["ameriflux"], dfv["cmor"])})
    if frequency in ["monthly", "daily"]:
        ds = ds.rename({"TIMESTAMP": "time"})
    if frequency == "30min":
        ds = ds.rename({"TIMESTAMP_START": "time"})
    if frequency == "monthly":
        ds["time"] = [
            cf.DatetimeGregorian(int(str(t.values)[:4]), int(str(t.values)[-2:]), 15)
            for t in ds["time"]
        ]
    if frequency == "daily":
        ds["time"] = [
            cf.DatetimeGregorian(int(str(t.values)[:4]), int(str(t.values)[-4:-2]), int(str(t.values)[-2:]))
            for t in ds["time"]
        ]
    if frequency == "30min":
        ds["time"] = [
            cf.DatetimeGregorian(int(str(t.values)[:4]), int(str(t.values)[4:6]), int(str(t.values)[6:8]), int(str(t.values)[8:10]), int(str(t.values)[-2:]))
            for t in ds["time"]
        ]
    
    
    ds["site"].attrs["name"] = "AmeriFlux site id"
    ds = ds.assign_coords(
        dict(
            lat=("site", dfi.loc[ds["site"], "LOCATION_LAT"]),
            lon=("site", dfi.loc[ds["site"], "LOCATION_LONG"]),
        )
    )
   
    with open("dataset_to_check.pkl", 'wb') as f:
        pickle.dump(ds, f, protocol=-1)

    # Standardize the dimensions
    if frequency == "monthly":
        ds = ild.set_time_attrs(ds, bounds_frequency="M")
    if frequency == "daily":
        ds = ild.set_time_attrs(ds, bounds_frequency="D")
    if frequency == "30min":
        ds = ild.set_time_attrs(ds, bounds_frequency="30min")
    ds = ild.set_lat_attrs(ds)
    ds = ild.set_lon_attrs(ds)
    
    # Define the global attributes
    download_stamp = time.strftime(
    #   "%Y-%m-%d", time.localtime(Path(f"{RAW_PATH}/*.xlsx").stat().st_ctime)
        "%Y-%m-%d", time.localtime(os.stat(badm_file).st_ctime)
    )
    generate_stamp = time.strftime("%Y-%m-%d")
    # default obs4MIP tracking_id may not be applicable to all data
    # tracking_id = ild.gen_trackingid()
    # leave tracking_id field empty
    tracking_id = ""
    
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
    
        if frequency == "monthly":
            flabel="mon"
        if frequency == "daily":
            flabel="day"
        if frequency == "30min":
            flabel="30min"
        has_stderr = "gpp" in varname or "reco" in varname
        out_ds = ild.set_ods26_global_attrs(
            out_ds,
            aux_uncertainty_id="stderr" if has_stderr else "",
            comment=f"{varname}=(DT+NT)/2 and {varname}_stderr = |DT-NT|/2"
            if has_stderr
            else "",
            contact="AmeriFlux Support Team (ameriflux-support@lbl.gov)",
            creation_date=generate_stamp,
            dataset_contributor="Jitendra Kumar",
            doi="N/A",
            frequency=flabel,
            grid="site",
            grid_label="site",
            has_aux_unc="TRUE" if has_stderr else "FALSE",
            history=f"""
    {download_stamp}: downloaded using https://ameriflux.org/data/download-data/;
    {generate_stamp}: converted to obs4MIP format""",
            activity_id="ILAMB-Data",
            institution="The AmeriFlux Community",
            institution_id="AmeriFlux",
            license="Data in this file produced by ILAMB is licensed under a Creative Commons Attribution - 4.0 International (CC BY 4.0) License (https://creativecommons.org/licenses/).",
            nominal_resolution="site",
            processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/AmeriFlux-FLUXNET/convert.py",
            product="site-observations",
            realm="land",
            references="Pastorello, Gilberto and Trotta, Carlo and Canfora, Eleonora and Chu, et al., The FLUXNET2015 dataset and the ONEFlux processing pipeline for eddy covariance data, Scientific Data, 10.1038/s41597-020-0534-3.",
            region="global_land",
            source="Eddy covariance flux tower measurements",
            source_id="AmeriFlux-FLUXNET",
            source_data_retrieval_date=download_stamp,
            source_data_url="https://ameriflux.org/data/download-data/",
            source_label="AmeriFlux",
            source_type="insitu",
            source_version_number="20251010",
            title=f"AmeriFlux FLUXNET {varname}",
            tracking_id=tracking_id,
            variable_id=varname,
            variant_label="ILAMB",
            variant_info="CMORized product prepared by ILAMB",
            version=f"v{generate_stamp.replace('-', '')}",
        )
    
        out_path = ild.create_output_filename(out_ds.attrs)
        out_ds.to_netcdf(out_path)

if __name__ == "__main__":
    # Parse the arguments
    parser = argparse.ArgumentParser()
    # Boolean flags (Store True if provided, else False)
    parser.add_argument("--download", action="store_true", help="Download the raw data")
    parser.add_argument("--extract", action="store_true", help="Extract files from archives")
    parser.add_argument("--process", action="store_true", help="Process the data")
    parser.add_argument(
        "--frequency", 
        required=True, 
        choices=["monthly", "daily", "30min"],
        help="Set the data frequency (Required: monthly, daily, or 30min)"
    )
    args = parser.parse_args() 

    if args.download == True:
        # Download Ameriflux 
        download_data()
    if args.extract == True:
        # Unzip data files 
        unzip_data(frequency=args.frequency)
    if args.process == True:
        # process data
        process_data(frequency=args.frequency)

