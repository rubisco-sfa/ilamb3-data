import datetime
import os
import sqlite3
import subprocess
import time
import warnings

import cftime as cf
import numpy as np
import pandas as pd
import rioxarray as rxr
import xarray as xr
from dask.distributed import Client, LocalCluster
from osgeo import gdal

from ilamb3_data import biblatex_builder as bb
from ilamb3_data import (
    get_cmip6_variable_info,
    set_lat_attrs,
    set_lon_attrs,
    set_ods_global_attributes,
    set_time_attrs,
)

#####################################################
# set the parameters for this particular dataset
#####################################################

# main parameters
chunksize = 3000
var = "cSoil"
long_name = "carbon mass in soil pool"
layers = ["D1", "D2", "D3", "D4", "D5", "D6", "D7"]
pools = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
sdate = datetime.datetime(1960, 1, 1)
edate = datetime.datetime(2022, 1, 1)

# dask parameters -- adjust these to fit your computer's capabilities
# chatgpt can optimize n_workers, n_threads, and mem_limit if you provide your computer specs!
n_workers = 20
n_threads = 1
mem_limit = "3.5GB"

# paths to files
remote_rast = (
    "https://s3.eu-west-1.amazonaws.com/data.gaezdev.aws.fao.org/HWSD/HWSD2_RASTER.zip"
)
local_rast = "HWSD2_RASTER/HWSD2.bil"
remote_data = "https://www.isric.org/sites/default/files/HWSD2.sqlite"
local_data = "HWSD2.sqlite"
github_path = "https://github.com/rubisco-sfa/ILAMB-Data/blob/master/HWSD2/convert.py"

# suppress specific warnings
warnings.filterwarnings("ignore", message="invalid value encountered in cast")
gdal.DontUseExceptions()

#####################################################
# functions in the order that they are used in main()
#####################################################


# 1. download raster and sql database to connect to raster
def download_data(remote_rast, remote_data):
    # check for raster directory
    rast_dir = os.path.splitext(os.path.basename(remote_rast))[0]
    if not os.path.isdir(rast_dir) or not any(
        fname.endswith(".bil") for fname in os.listdir(rast_dir)
    ):
        subprocess.run(["mkdir", rast_dir])
        subprocess.run(["curl", "-L", remote_rast, "-o", os.path.basename(remote_rast)])
        subprocess.run(["unzip", os.path.basename(remote_rast), "-d", rast_dir])
    # check for database
    sql_database = os.path.basename(remote_data)
    if not os.path.isfile(sql_database):
        subprocess.run(["curl", "-L", remote_data, "-o", sql_database])
    else:
        print(
            f"Raster {rast_dir} and Database {sql_database} are already downloaded to current directory."
        )


# 2. initialize the dask multiprocessing client; the link can be used to track worker progress
def initialize_client(n_workers, n_threads, mem_limit):
    cluster = LocalCluster(
        n_workers=n_workers, threads_per_worker=n_threads, memory_limit=mem_limit
    )
    client = Client(cluster)
    print(f"Dask dashboard link: {client.dashboard_link}")
    return client


# 3. load the raster we use to connect with HWSDv2 data
def load_raster(path, chunksize):
    rast = rxr.open_rasterio(
        path,
        band_as_variable=True,
        mask_and_scale=True,
        chunks={"x": chunksize, "y": chunksize},
    )
    rast = (
        rast.astype("int16").drop_vars("spatial_ref").rename_vars(band_1="HWSD2_SMU_ID")
    )
    return rast


# 4. load the table with data from the sqlite database
def load_layer_table(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    layer_df = pd.read_sql_query(query, conn)
    conn.close()
    return layer_df


# 5(a). function to calculate carbon stock
def calculate_stock(df, depth, bulk_density_g_cm3, cf, organic_carbon):
    df["stock"] = (
        df[bulk_density_g_cm3]
        * (1 - df[cf] / 100)
        * df[depth]
        * 0.01
        * df[organic_carbon]
    )
    return df["stock"]


# 5(b). function to calculate weighted mean
def weighted_mean(values, weights):
    return (values * weights).sum() / weights.sum()


# 5. process each soil layer by selecting the layer & pools of interest,
# removing erroneous negative values, calculating C stock, and getting
# the weighted mean of the pools
def process_layers(layer_df, layers, pools, var):
    dfs = []
    for layer in layers:
        sel = layer_df[
            [
                "HWSD2_SMU_ID",
                "LAYER",
                "SEQUENCE",
                "ORG_CARBON",
                "BULK",
                "BOTDEP",
                "TOPDEP",
                "COARSE",
                "SHARE",
            ]
        ]
        df = sel[sel["LAYER"] == layer].drop(columns=["LAYER"])
        df = df[df["SEQUENCE"].isin(pools)]
        for attr in ["ORG_CARBON", "BULK", "SHARE"]:
            df[attr] = df[attr].where(df[attr] > 0, np.nan)
        df[var] = calculate_stock(df, "BOTDEP", "BULK", "COARSE", "ORG_CARBON")
        grouped = (
            df.groupby("HWSD2_SMU_ID")
            .apply(
                lambda x: pd.Series({var: weighted_mean(x["ORG_CARBON"], x["SHARE"])}),
                include_groups=False,
            )
            .reset_index()
        )
        dfs.append(grouped)
    return dfs


# 6. combine all the layers by summing, and set the data types
def combine_and_summarize(dfs, var):
    total_df = pd.concat(dfs)
    total_df = total_df.groupby("HWSD2_SMU_ID")[var].agg("sum").reset_index(drop=False)
    total_df["HWSD2_SMU_ID"] = total_df["HWSD2_SMU_ID"].astype("int16")
    total_df[var] = total_df[var].astype("float32")
    return total_df


# 7(a). function to map the soil unit ID to the cSoil variable
def map_uid_to_var(uid, uid_to_var):
    return uid_to_var.get(uid, float("nan"))


# 7. create a variable in the rast dataset containing cSoil data
def apply_mapping(rast, total_df, var):
    uid_to_var = total_df.set_index("HWSD2_SMU_ID")[var].to_dict()
    mapped_orgc = xr.apply_ufunc(
        map_uid_to_var,
        rast["HWSD2_SMU_ID"],
        input_core_dims=[[]],
        vectorize=True,
        dask="parallelized",
        output_dtypes=["float32"],
        kwargs={"uid_to_var": uid_to_var},
    )
    rast = rast.assign({var: mapped_orgc})
    return rast


# 8. save the rast dataset as a tif
def save_raster(rast, var, layers, pools):
    output_path = f"hwsd2_{var}_{layers[0]}-{layers[-1]}_seq{pools[0]}-{pools[-1]}.tif"
    rast[[var]].rio.to_raster(output_path)
    return output_path


# 9. resample the 250m resolution to 0.5deg resolution
def resample_raster(input_path, output_path, xres, yres, interp, nan):
    gdal.SetConfigOption("GDAL_CACHEMAX", "500")
    ds = gdal.Warp(
        output_path,
        input_path,
        xRes=xres,
        yRes=yres,
        resampleAlg=interp,
        outputType=gdal.GDT_Float32,
        dstNodata=nan,
        outputBounds=(-180.0, -90.0, 180.0, 90.0),
    )
    del ds


# 10. create a netcdf of the 0.5deg resolution raster
def create_netcdf(input_path, output_path, var, sdate, edate, long_name):
    # open the .tif file
    csoil = rxr.open_rasterio(input_path, band_as_variable=True, mask_and_scale=True)

    # rename the bands
    csoil = csoil.rename({"x": "lon", "y": "lat", "band_1": var})

    # create time dimension
    tb_arr = np.asarray(
        [
            [cf.DatetimeNoLeap(sdate.year, sdate.month, sdate.day)],
            [cf.DatetimeNoLeap(edate.year, edate.month, edate.day)],
        ]
    ).T
    tb_da = xr.DataArray(tb_arr, dims=("time", "nv"))
    csoil = csoil.expand_dims(time=tb_da.mean(dim="nv"))
    csoil["time_bounds"] = tb_da

    # set dimension attributes
    csoil = set_time_attrs(csoil)
    csoil = set_lat_attrs(csoil)
    csoil = set_lon_attrs(csoil)

    # populate information from CMIP6 variable information
    info = get_cmip6_variable_info(var)
    csoil[var].attrs["long_name"] = info["variable_long_name"]
    csoil[var].attrs["units"] = "kg m-2"

    # create the global attributes
    generate_stamp = time.strftime(
        "%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(local_data))
    )

    # set up citation
    citation = bb.generate_biblatex_techreport(
        cite_key="Nachtergaele2023",
        author=[
            "Nachtergaele, Freddy",
            "van Velthuizen, Harrij",
            "Verelst, Luc",
            "Wiberg, Dave",
            "Henry, Matieu",
            "Chiozza, Federica",
            "Yigini, Yusuf",
            "Aksoy, Ece",
            "Batjes, Niels",
            "Boateng, Enoch",
            "Fischer, Günther",
            "Jones, Arwyn",
            "Montanarella, Luca",
            "Shi, Xuezheng",
            "Tramberend, Sylvia",
        ],
        title="Harmonized World Soil Database",
        institution="Food and Agriculture Organization of the United Nations and International Institute for Applied Systems Analysis, Rome and Laxenburg",
        year="2023",
        number="2.0",
    )

    csoil = set_ods_global_attributes(
        ds=csoil,
        activity_id="obs4MIPs",
        contact="Matieu Henry (Matieu.Henry@fao.org)",
        conventions="CF-1.12 ODS-2.5",
        creation_date=generate_stamp,
        dataset_contributor="Morgan Steckler (stecklermr@ornl.gov)",
        data_specs_version="2.5",
        external_variables="",
        frequency="fx",
        grid="0.5x0.5 degree",
        grid_label="gn",
        history=f"""
{generate_stamp}: downloaded source from {remote_data}
{generate_stamp}: filtered data to soil dominance sequence(s) {pools}; where 1 is the dominant soil type
{generate_stamp}: masked invalid negative organic_carbon_pct_wt and bulk_density_g_cm3 with np.nan
{generate_stamp}: calculated cSoilLevels in kg m-2 for each level {layers}: bulk_density_g_cm3 * 10 * (1 - coarse_fragment_pct_vol / 100) * bottom_depth_cm / 10 * organic_carbon_pct_wt / 100)
{generate_stamp}: calculated {var} by getting the weighted mean of all pools in a level and summing {layers} cSoilLevels
{generate_stamp}: resampled to 0.5 degree resolution using mean
{generate_stamp}: created CF-compliant metadata
{generate_stamp}: exact details on this process can be found at {github_path}""",
        institution="Food and Agriculture Organization of the United Nations",
        institution_id="FAO",
        license="CC BY-NC-SA 3.0",
        nominal_resolution="0.5x0.5 degree",
        processing_code_location=github_path,
        product="observations",
        realm="land",
        references=citation,
        source="HWSD",
        source_id="HWSD",
        source_data_notes="",
        source_data_retrieval_date=generate_stamp,
        source_data_url=f"{remote_rast} {remote_data}",
        source_label="",
        source_type="insitu_gridded",
        source_version_number="2.0",
        title="Harmonized World Soil Database version 2.0",
        tracking_id="",
        variable_id="cSoil",
        variant_label="BE",
        variant_info="",
    )

    # clean up the dataset
    csoil["lat"] = csoil["lat"].astype("float32")
    csoil["lon"] = csoil["lon"].astype("float32")
    csoil = csoil.drop_vars("spatial_ref")
    csoil = csoil.reindex(lat=list(reversed(csoil.lat)))

    # export as netcdf
    csoil.to_netcdf(output_path, format="NETCDF4", engine="netcdf4")


# use all nine steps above to convert the data into a netcdf
def main():
    download_data(remote_rast, remote_data)

    client = initialize_client(n_workers, n_threads, mem_limit)

    rast = load_raster(local_rast, chunksize)

    layer_df = load_layer_table(local_data, "HWSD2_LAYERS")

    dfs = process_layers(layer_df, layers, pools, var)

    total_df = combine_and_summarize(dfs, var)

    rast = apply_mapping(rast, total_df, var)

    output_path = save_raster(rast, var, layers, pools)

    resample_raster(
        output_path,
        f"hwsd2_{var}_{layers[0]}-{layers[-1]}_seq{pools[0]}-{pools[-1]}_resamp.tif",
        0.5,
        0.5,
        "average",
        0,
    )

    create_netcdf(
        f"hwsd2_{var}_{layers[0]}-{layers[-1]}_seq{pools[0]}-{pools[-1]}_resamp.tif",
        f"hwsd2_{var}.nc",
        var,
        sdate,
        edate,
        long_name,
    )

    client.close()


if __name__ == "__main__":
    main()
