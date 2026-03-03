# Contribute Data to ILAMB

- [What data does ILAMB software ingest?](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#what-data-does-ilamb-software-ingest)
- [What does an ILAMB reference dataset NetCDF look like?](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#what-does-an-ilamb-reference-dataset-netcdf-look-like)
- [How to format an ILAMB-ready dataset](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#how-to-format-an-ilamb-ready-dataset)
    - [Create a GitHub Issue](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#1-create-a-github-issue-in-the-ilamb3-data-repository)
    - [Fork and clone the repository](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#2-fork-and-clone-the-ilamb3-data-repository-to-your-local-machine)
    - [Set up the working environment](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#3-set-up-the-ilamb3-data-working-environment)
    - [Write the formatting script](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#4-write-the-formatting-script)
        - [Import libraries and functions](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#41-import-libraries-and-functions)
        - [Download the data](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#42-download-the-data)
        - [Convert variables names to CMIP6 standard names](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#43-convert-variables-names-to-cmip6-standard-names-when-possible)
        - [Set up temporal coordinates and bounds](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#44-set-up-the-temporal-coordinates-and-bounds)
        - [Set up spatial coordinates and bounds](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#45-set-up-the-spatial-coordinates-and-bounds)
        - [Set up variable attributes](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#46-set-up-the-variable-attributes)
        - [Set up global attributes and export](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#47-set-up-the-global-attributes-and-export-an-ilamb-ready-netcdf)
- [How to add formatted data to ilamb3-data](https://github.com/rubisco-sfa/ilamb3-data/blob/add-data-doc/docs/how-to-contribute-data.md#how-to-add-formatted-data-to-ilamb3-data)

---

ILAMB only ingests data stored in a particular file format. This means that the software may fail to run if the benchmarking data are not formatted in a way that ILAMB can understand. So, ILAMB comes pre-equipped with data that are already formatted and usable by the software. 

But what if you want to benchmark using a reference dataset that we haven’t prepared, and you don’t want to spend hours trying to format it to work in ILAMB? We have created tools for you to quickly and properly format an ILAMB-ready dataset. Once a dataset is ILAMB-ready, it can be added to the ILAMB data registry for any user to access, and you will have contributed to the development of ILAMB software!
<br>
<br>

## What data does ILAMB software ingest?

ILAMB-ready data are stored as [NetCDFs](https://www.unidata.ucar.edu/software/netcdf/), and they adhere to community formatting conventions. Each NetCDF contains one data variable (with an optional uncertainty variable), relevant dimensions (like latitude, longitude, time, depth, etc.), global attributes (metadata), and is gridded or site-level.
<br>
<br>

## What does an ILAMB reference dataset NetCDF look like?

Let’s walk through the structure of a gridded dataset that is both ILAMB-ready and adheres to community-defined formatting standards. In this example, we have a precipitation variable from the Conserving Land-Atmosphere Synthesis Suite (CLASS) version 1.1 that is ILAMB-ready. You can [download the NetCDF](https://www.ilamb.org/ilamb3-data/CLASS-1-1/obs4MIPs_UNSW_CLASS-1-1_mon_pr_gn_v20260302.nc) onto your machine and use the command `ncdump` to display the header of the file:

```ncdump -h obs4MIPs_UNSW_CLASS-1-1_mon_pr_gn_v20260302.nc```

You should get something like this:

```bash
dimensions:
        time = 84 ;
        lat = 360 ;
        lon = 720 ;
        bnds = 2 ;
```

At the top of the header, we see 4 dimensions: time, lat, lon, and bnds (bounds). Dimensions define the shape and order of the data variables. For example, precipitation (pr) has 84 time steps, and the grid is 360 by 720 pixels large (i.e., 1 degree latitude x longitude). The time, lat, and lon values are in sequential order, 

Bounds (bnds) is a dimension, but it has no pre-defined values (i.e., coordinates) associated with it. You can think of coordinates as data values. The longitude dimension will have values like -179.75, -179.25, -178.75, etc., and latitude will have values like -89.75, -89.25, -88.75, etc. You can see that bnds has no coordinates by running this command:

```ncdump -c obs4MIPs_UNSW_CLASS-1-1_mon_pr_gn_v20260302.nc```

You will see something like this:


```bash
data:
 lon = -179.75, -179.25, -178.75, -178.25, -177.75, -177.25, -176.75, …
 lat = -89.75, -89.25, -88.75, -88.25, -87.75, -87.25, -86.75, -86.25, …
 time = 15.5, 45, 74.5, 105, 135.5, 166, 196.5, 227.5, 258, 288.5, 319, …
```

See, bnds is not displayed because it has no defined coordinate values. This is because bounds (bnds) are often used for variables that are also dimensions, such as time, latitude, longitude, and depth. For example, a latitude coordinate of 35.75 has bnds of (35.5, 36). This helps us understand where the edges (bnds) of the 35.75 latitude centroid (coordinate) are located. You can see that the bnds values would be different for latitude and longitude. So, we can’t have specified coordinates for the bnds dimension; it only defines the shape, which is a tuple of two bounding values.

After you run `ncdump -h obs4MIPs_UNSW_CLASS-1-1_mon_pr_gn_v20260302.nc`, you will also see this:

```bash
variables:
        float pr(time, lat, lon) ;
                pr:_FillValue = 1.e+20f ;
                pr:long_name = "Precipitation" ;
                pr:standard_name = "precipitation_flux" ;
                pr:units = "kg m-2 s-1" ;
                pr:ancillary_variables = "pr_sd" ;
        float pr_sd(time, lat, lon) ;
                pr_sd:_FillValue = 1.e+20f ;
                pr_sd:standard_name = "pr standard_deviation" ;
                pr_sd:units = "kg m-2 s-1" ;
        double lon(lon) ;
                lon:axis = "X" ;
                lon:units = "degrees_east" ;
                lon:standard_name = "longitude" ;
                lon:long_name = "Longitude" ;
                lon:bounds = "lon_bnds" ;
        double lat(lat) ;
                lat:axis = "Y" ;
                lat:units = "degrees_north" ;
                lat:standard_name = "latitude" ;
                lat:long_name = "Latitude" ;
                lat:bounds = "lat_bnds" ;
        double time_bnds(time, bnds) ;
        double time(time) ;
                time:axis = "T" ;
                time:standard_name = "time" ;
                time:long_name = "time" ;
                time:bounds = "time_bnds" ;
                time:units = "days since 2003-01-01 00:00:00" ;
                time:calendar = "gregorian" ;
        double lat_bnds(lat, bnds) ;
        double lon_bnds(lon, bnds) ;
```

Now, we can dive into the data itself. We have a precipitation variable, pr, which has attributes like `_FillValue`, `long_name`, `standard_name`, `units`, and `ancillary_variables`. These attributes describe the variable itself, as well as any directly related variables. This formatting structure adheres to [CF Conventions](https://cfconventions.org/) that “define metadata that provide a definitive description of what the data in each variable represents, and the spatial and temporal properties of the data.” The data also adhere to [obs4MIPs Data Specifications](https://doi.org/10.5281/zenodo.11500473) (ODS) so that reference data can be published as part of the obs4MIPs project in the Earth System Grid Federation (ESGF), which archives petabytes of data related to the Coupled Model Intercomparison Projects (CMIPs).

While adhering data to community standards is great, you can imagine that formatting your data to look like this would usually be painstaking, extremely detail-oriented, and prone to error. That’s why we have built all the tools you need to format your data quickly, without having to know all of the community conventions.
<br>
<br>

## How to format an ILAMB-ready dataset
### 1. Create a GitHub Issue in the ilamb3-data repository

First, let us know that you're working on formatting a dataset by creating a GitHub Issue in the ilamb3-data repository. This way, we can keep track of the datasets that are being worked on, and we can also help you if you have any questions along the way. To create an issue, visit https://github.com/rubisco-sfa/ilamb3-data/issues. Click on “New Issue,” and then fill out the template with the name of the dataset you are working on, a brief description of the dataset, and any relevant links or other information. This will help us understand what you are working on so we can properly assist you. You can look at current and closed Issues to see how we have been writing them.

If you run into any bugs, or you have an idea for a new function that would make formatting easier, please also create an Issue to let us know. We are always looking for ways to improve ilamb3-data, and we appreciate any feedback you have.
<br>
<br>

### 2. Fork and clone the ilamb3-data repository to your local machine

Follow this tutorial if you would like to format a dataset to be ILAMB-ready. We have all the built-in tools you need to make an ILAMB-legible reference dataset that also conveniently adheres to community standards. First, you will need to create a GitHub account, and then you will need to fork our repository. To do so,visit https://github.com/rubisco-sfa/ilamb3-data and click “Fork,” or optionally install the [GitHub command line tool](https://cli.github.com/), and then run the following in your terminal:

```bash
gh auth login
cd Desktop  # navigate to a directory where you want to clone the repo
gh repo fork rubisco-sfa/ilamb3-data --clone
```

Now, you have a copy of the repo stored on your GitHub account. Whenever you do work on a fork, it is good practice to regularly pull down the latest commits from the repository. We update the repository often, so you’ll want to fetch repo updates every time you work on your fork:

```bash
cd ilamb3-data  # navigate to the directory where you cloned the repo
git remote add upstream https://github.com/rubisco-sfa/ilamb3-data.git
git remote -v
git fetch upstream
```
<br>
<br>

### 3. Set up the ilamb3-data working environment

Now that you have forked and created a clone of the ilamb3-data, you should set up your Python working environment. To manage working environments, we recommend using Astral’s library called `uv`. You can follow [these instructions](https://docs.astral.sh/uv/getting-started/installation/) to install it on your machine. It is open-source and lightning fast, but unlike [Anaconda](https://www.anaconda.com/download), it can only manage Python environments and cannot configure non-Python libraries. You can use any environment manager you prefer. Here is how you can quickly set up your working environment with `uv`:

```bash
cd ilamb3-data  # navigate to the directory where you cloned the repo
uv sync
```

That’s it! Now you have a Python version and other libraries that work with our built-in functions. To run a Python script or Jupyter notebook, you need to activate the environment. Navigate to the outer ilamb3-data directory where the `uv.sync` file is located:

```bash
cd ilamb3-data
source .venv/bin/activate
```

Now you can use our Python functions to format your data. To deactivate the environment, just type `deactivate` in your terminal. Next, you will need to create a new data directory. This is where you will store the raw (pre-formatted) data, your formatting python script, and any other files you need while you work. For example, some people like to code in blocks using Jupyter. You’ll just need to temporarily install Jupyter into the environment using `uv pip install jupyterlab`. Here is where you should create your new working directory:

```bash
cd ilamb3-data/data
mkdir EXAMPLE-1-0  # we prefer to name these folders after the source and version of the dataset
vi convert.py  # this creates the Python formatting script; you can call it whatever you want, e.g., format.py, process.py, etc.
:wq  # this writes the file and quits the vi editor
```

Now that you’ve created the Python script, you can use your favorite code editor to write in it (e.g., VSCode, JupyterLab, etc.). You might also create a `convert.ipynb` if you prefer to draft code in blocks before transferring it to the final convert.py.
<br>
<br>

### 4. Write the formatting script

We store these formatting scripts on the ilamb3-data GitHub repository so that we have a record of how we formatted the data. This helps us debug any errors that we find, and two years from now, we can know exactly what we were thinking when we formatted a dataset. Further, if we make any big changes to how we format ILAMB data, we can now just re-run all the scripts to get new ILAMB versions of the data.

A formatting script does only 3 things:

- Downloads the data (or provides instructions on how to access it)
- Formats the data
- Exports a NetCDF that is ready to use in ILAMB
<br>
<br>

#### 4.1. Import libraries and functions

Let’s walk through the [convert.py](https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/CLASS-1-1/convert.py) formatting script for CLASS-1-1, step-by-step. First, we import the necessary libraries and functions:

```python
import time
from pathlib import Path

import cftime as cf  # cftime is great for working with time values in NetCDFs, especially when they have non-standard calendars
import numpy as np
import xarray as xr  # xarray is great for working with multi-dimensional tifs/tables/NetCDFs

import ilamb3_data as ild  # this is the library we built with all the formatting tools you need
```
<br>
<br>

#### 4.2. Download the data

Then, we download the data. First, we create a directory called `_raw`, which is where we will store the raw (pre-formatted) data. We like to use the `Path` library because it is simpler and more robust than `os` or `shutil`. If the directory already exists, that's fine. If it doesn't yet exist, it will be created.

```python
RAW_PATH = Path("_raw")
if not RAW_PATH.is_dir():
    RAW_PATH.mkdir()
```

This particular dataset is stored on a thredds server, which you can access directly using htmls. We have a built-in function called `download_from_html()` that is handy here. You just need to provide the URL and a path to the file you want to save the data to. The function will check if the file already exists, and if it doesn't, it will download the data and save it to the specified path. We have other built-in downloading functions, such as `download_from_zenodo()`. We have also installed [earthaccess](https://earthaccess.readthedocs.io/en/latest/) into the Python environment, so you can easily search and download data from NASA Earthdata servers. All of our built-in functions are stored in `ilamb3_data/ilamb3_data/`, so you can check there to see if we have a function that works for your data source. If not, you can write your own function and add it to the library for others to use!

```python
# define sources
remote_sources = [
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2003.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2004.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2005.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2006.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2007.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2008.nc",
    "https://thredds.nci.org.au/thredds/fileServer/ks32/ARCCSS_Data/CLASS/v1-1/CLASS_v1-1_2009.nc",
]
local_sources = [str(RAW_PATH / Path(s).name) for s in remote_sources]  # create local paths for the data to be saved to

# ensure we have downloaded the data
for remote_source, local_source in zip(remote_sources, local_sources):
    if not Path(local_source).is_file():  # don't download the data if we already have it
        ild.download_from_html(remote_source, local_source)  # you can find our built-in functions at ilamb3-data/ilamb3_data
    download_stamp = ild.gen_utc_timestamp(Path(local_source).stat().st_mtime)  # save when the data were downloaded; this will go in our global attrs later
```
<br>
<br>

#### 4.3. Convert variables names to CMIP6 standard names when possible

Now that we have the data, we can format it. This is where you will use the built-in functions in `ilamb3_data` to quickly and easily format your data. You can also write your own functions if you need to do something that our built-in functions don't cover. The goal here is to get your data into a format that looks like the example NetCDF we showed earlier. You can use the `xarray` library to work with your data as you format it. We have found that it is the best library for working with multi-dimensional data, and it has a lot of built-in functions that make formatting easier.

First, we open the raw data files as an xarray object. There are several kinds of data objects to choose from in xarray. We are most often working with xarray Datasets and DataArrays. Read more about them [here](https://docs.xarray.dev/en/latest/user-guide/data-structures.html). Here, we load the data files as an xarray Dataset, which is a collection of DataArrays. Each DataArray has its own name, dimensions, coordinates, and attributes. The Dataset itself also has its own attributes.

```python
ds = xr.open_mfdataset(local_sources)
ds  # If you're working in a Jupyter notebook, you can run this line to see the structure of the Dataset and the variables it contains
```

The first step is to understand the structure of the raw data files. This is an important step because it helps you understand what you are working with before you start formatting. Once you understand the structure of the raw data files, you can start using xarray functions to manipulate the data and get it into the format that ILAMB expects. For example, you might need to rename some variables, change the dimensions, add attributes, etc. 

After inspecting the raw files, we can see that, first, we need to rename some variables to match CMIP6 standard names. Not all data variables are in the [CMIP6 Variable Controlled Vocabulary Table](https://airtable.com/appYNLuWqAgzLbhSq/shrgcENhJZU1y3ye0/tbleXPCaJeYeIzAhR) (CV) or the [MIP CV Table](https://clipc-services.ceda.ac.uk/dreq/mipVars.html), but if they are, it is best practice to use them. CVs change over time, so you should use the most recent version when deciding how to name your variables. If you are unsure about how to name your variables, you can check the CVs, or you can ask us in the Issue you created on the ilamb3-data repository. 

In this case, we can see that the raw data files have variables called `hfds` and `rs`. We know that these variables represent surface downwelling heat flux (hfds) and surface downwelling shortwave radiation (rs). So, we can search these descriptions on the [CMIP6 Variable CV Table](https://airtable.com/appYNLuWqAgzLbhSq/shrgcENhJZU1y3ye0/tbleXPCaJeYeIzAhR), which shows us that the standard name for `hfds` is actually `hfdsl` and the standard name for `rs` is actually `rss`. So, we can use the `rename()` function in xarray to rename these variables to match the CMIP6 standard names. This is important because it helps ensure that our data are consistent with other datasets and that they can be easily compared and used in ILAMB.

```python
ds = ds.rename({"hfds": "hfdsl", "hfds_sd": "hfdsl_sd", "rs": "rss", "rs_sd": "rss_sd"})  # rename variables/uncertainty to match CMIP6 standard names
```

We noticed that the variables also come with standard deviation as uncertainty (_sd). To properly name an uncertainty variable, we refer to the [obs4MIPs ODS 2.6](https://zenodo.org/records/17789550) Appendix 5, which defines 7 uncertainty suffixes to append to a standard variable name:

- nobs: the number of discrete observations or measurements from which a data value has been derived
- ustd: the per-datum standard uncertainty is a combination of independent, structured and common effects and is equal to the positive square root of the sum of the components. The standard uncertainty is provided at one standard deviation. Replacement for ‘standard error’ due to updated terminology
- uind: the per-datum component of uncertainty that is associated with uncorrelated effects, between observations, provided at one standard deviation
- ustr: the per-datum component of uncertainty that is structured and correlated over a defined space/time scale, provided at one standard deviation. This correlation length scale in space and time must be provided in the variable metadata.
- ucom: the per-datum component of uncertainty that is common to all observations of a given type (specific instrument, resolution, variable type) at one standard deviation. The definition of the type of measurement for which this uncertainty is common should be provided in the variable metadata
- lbnd: alternative to stderr and ustd for observations with asymmetrical uncertainty distribution, lower bound of the uncertainty interval
- ubnd: alternative to stderr and ustd for observations with asymmetrical uncertainty distribution, upper bound of the uncertainty interval

There are also two legacy suffixes that can be used:

- stderr: legacy term used for backward compatibility. Definition is identical to that of ustd, the standard uncertainty
- sd: the standard deviation on the mean, legacy term for backwards compatibility

If you find that your uncertainty data does not fit into any of these categories, you can create your own suffix, but you must define it in the variable metadata. In this case, we have standard deviation on the mean, so we can use the legacy suffix of `_sd`.
<br>
<br>

#### 4.4. Set up the temporal coordinates and bounds

Now that the variables are named properly, we can fix up the coordinates: time, latitude, and longitude, as well as the bnds dimension. You should always inspect the time coordinates before manipulating them. Commonly, the time coordinates are relative to a reference date, e.g., in "days since YYY-MM-DD". If that's the case, you can parse the "days since YYY-MM-DD" integers into `cf.datetime` objects using the `cftime` library we imported earlier. 

```python
ds["time"].dtype  # determine the data type of the time coordinates
```

You should see that the time coordiantes are `dtype('<M8[ns]')`, which means that they are in nanosecond datetime format. Xarray has a `.dt` accessor that allows you to easily manipulate datetime objects like this. So, let's turn out datetime coordinates into `cf.datetime` objects so we can use our built-in functions: 

```python
# parse the time coordinates into cf.datetime objects; Gregorian is a common calendar that many prefer to use
ds["time"] = [cf.DatetimeGregorian(t.dt.year, t.dt.month, t.dt.day) for t in ds["time"]]
# then, use our set_time_attrs() function to add the necessary time attributes to the Dataset
ds = ild.set_time_attrs(
    ds, bounds_frequency="M", ref_date=cf.DatetimeGregorian(2003, 1, 1)  # choose a reference date that makes sense for your data
)
```

The function `set_time_attrs()` can do several things. It always ingests an xr.Dataset, and you must specify the `bounds_frequency` (e.g., "M" for monthly, "D" for daily, etc.). Optionally, you can provide a cf.datetime `ref_date` to set as the time units in "days since ref_date", set `create_new_time=True` if you want to build the time dimension from scratch given a cf.datetime `sdate` and `edate`, or we can even create a climatological calendar by setting `climatology=True` and defining the cf.datetime `clim_sdate` and `clim_edate`. This function will automatically generate the necessary time attributes and encoding, such as `axis`, `long_name`, `standard_name`, `units`, `calendar`, and `bounds`.

[!HINT] Sometimes, data creators set the time units to something like "months since YYY-MM-DD." This violates CF Conventions and is not legible by xarray. When you try to read a dataset like this into xarray, it will fail. See how we handled a case like this in the WOA-23 heat content [formatting script](/Users/6ru/Desktop/ilamb3-data/data/WOA-23/heat_content/convert.py).
<br>
<br>

#### 4.5. Set up the spatial coordinates and bounds

Next, we can set up the latitude and longitude coordinates and bounds. We have a built-in function called `set_lat_attrs()`, `set_lon_attrs()`, and `set_depth_attrs()` that can do this for you. These functions will automatically add the necessary attributes and encoding, such as `axis`, `long_name`, and `standard_name`. Then, we have `set_coord_bounds()` to add bounds to any coordinate that needs a bnds dimension. Usually this will be used on latitude, longitude, and/or depth.

```python
ds = ild.set_lat_attrs(ds)
ds = ild.set_lon_attrs(ds)
ds = ild.set_coord_bounds(ds, "lat")
ds = ild.set_coord_bounds(ds, "lon")
```
<br>
<br>

#### 4.6. Set up the variable attributes

Next, we can set up the variable attributes. This is where you will use the `set_var_attrs()` function to add the necessary attributes to your data variables:

```python
generate_stamp = time.strftime("%Y%m%d")  # today's date; this will go in our global attrs
tracking_id = ild.gen_trackingid()  # a unique identifier for this dataset; this will go in our global attrs
variables = [v for v in ds if ("_sd" not in v and "_bnds" not in v)]  # get the variable names, excluding uncertainty and bounds variables
for var in variables:
    uncert = f"{var}_sd"
    out = ds.drop_vars(
        [d for d in ds if d not in [var, uncert] and not d.endswith("_bnds")]
    )  # drop any variables that are not the main variable, its uncertainty, or bounds variables

    # get standard/long name info and manage when it's not in CMIP6 CV
    if "_sd" not in var:
        try:
            # get a dict of CMIP6 standard name, long name, and units for the variable
            var_info = ild.get_cmip6_variable_info(var, var)
        except Exception:
            var_info = {
                "cf_standard_name": out[var].attrs.get("standard_name", var),
                "variable_long_name": out[var].attrs.get(
                    "long_name", var.replace("_", " ").title()
                ),
                "variable_units": out[var].attrs.get("units", ""),
            }
            var_info["variable_long_name"] = (
                var_info["variable_long_name"].replace("_", " ").title()
            )

    # format the variable attributes
    out = ild.set_var_attrs(
        out,
        var=var,
        cmip6_units=var_info["variable_units"],
        cmip6_standard_name=var_info["cf_standard_name"],
        cmip6_long_name=var_info["variable_long_name"],
        ancillary_variables=uncert,
        target_dtype=np.float32,
        convert=False,
    )
```

We have a handy function called `get_cmip6_variable_info()` that uses a library called `intake-esgf` to see if the variable is in the CMIP6 CV. If it is, the function will return the standard name, long name, and units for that variable. If it is not, the function will throw an error, and you can catch that error to assign your own variable info. This is a great way to ensure that your variable names are consistent with the CMIP6 CV when possible, but also gives you the flexibility to define your own variable info when necessary.

Then, you can input that information into `set_var_attrs()`, so you can correctly set the variable attributes. This function also allows you to set the data type and convert data values to different units. In this case, we do not convert the data values to CMIP6 standard units. Sometimes, converting variables between units results in data loss due to precision issues, so you should be mindful when converting data values. You can also use this function to set ancillary variables like uncertainty, and you can also add `cell_methods` information if your variable was integrated over a particular dimension and how. E.g., if your variable is a monthly average, you can set `cell_methods="time: mean"`.

After we set up the variable attributes, we drop some straggling unnecessary information and then set the uncertainty variable's attributes. Refer to the ODS 2.6 Appendix 5 for how to name and format uncertainty variables. In this case, we have standard deviation on the mean, so we use the legacy suffix of `_sd`, and we set the standard name to be the main variable's standard name with "standard_deviation" appended to it.

```python
    # drop some straggling attrs
    out[var].attrs.pop("ALMA_short_name", None)

    # format the ancillary var attrs
    out[uncert].attrs = {
        "standard_name": f"{var} standard_deviation",
        "units": out[var].attrs["units"],
    }
    out[uncert].encoding["_FillValue"] = np.float32(1.0e20)  # this is the standard _FillValue for float32 variables
```
<br>
<br>

#### 4.7. Set up the global attributes and export an ILAMB-ready NetCDF

Finally, we can add global attributes (metadata) to the Dataset. Global attributes are important because they provide context and information about the dataset as a whole. They can include things like the title of the dataset, the source of the data, the history of how the data were processed, etc. We have a built-in function called `set_ods26_global_attrs()` that sets the global attributes to adhere to obs4MIPs Data Specifications (ODS) 2.6. The function will display warnings when required attributes are missing or incorrect. We begin by looping through each variable so that we end up with one variable per NetCDF, which is the format that ILAMB can interpret. An uncertainty variable is allowed, but any other variables that are not the main variable or its uncertainty should be removed from the Dataset before exporting.

```python
    out = ild.set_ods26_global_attrs(
        out,
        aux_uncertainty_id="sd",
        contact="Sanaa Hobeichi (s.hobeichi@unsw.edu.au)",
        creation_date=generate_stamp,
        dataset_contributor="Nathan Collier",
        doi="https://doi.org/10.25914/5c872258dc183",
        frequency="mon",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="TRUE",
        history=f"""
{download_stamp}: downloaded using https://geonetwork.nci.org.au/geonetwork/srv/eng/catalog.search#/metadata/f4854_2536_6084_5147;
{generate_stamp}: converted to obs4MIP format""",
        institution="University of New South Wales, Sydney, New South Wales, AUS",
        institution_id="UNSW",
        license="https://creativecommons.org/licenses/by-nc-sa/4.0/",
        nominal_resolution="0.5 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/CLASS-1-1/convert.py",
        product="derived",
        realm="land",
        references="Hobeichi, Sanaa and Abramowitz, Gab and Evans, Jason, Conserving Land-Atmosphere Synthesis Suite (CLASS), Journal of Climate, 33(5), 2020, 10.1175/JCLI-D-19-0036.1.",
        region="global_land",
        source="Ground Heat Flux (GLDAS, MERRALND, MERRAFLX, NCEP_DOII, NCEP_NCAR), Sensible Heat Flux(GLDAS, MERRALND, MERRAFLX, NCEP_DOII, NCEP_NCAR, MPIBGC, Princeton), Latent Heat Flux(DOLCE1.0), Net Radiation (GLDAS, MERRALND, NCEP_DOII, NCEP_NCAR, ERAI, EBAF4.0), Precipitation(REGEN1.1), Runoff(LORA1.0), Change in Water storage(GRACE(GFZ, JPL, CSR))",
        source_id="CLASS-1-1",
        source_data_retrieval_date=download_stamp,
        source_data_url=",".join(remote_sources),
        source_label="CLASS",
        source_type="gridded_insitu",
        source_version_number="1.1",
        title=f"Conserving Land-Atmosphere Synthesis Suite ({var})",
        tracking_id=tracking_id,
        variable_id=var,
        variant_label="ILAMB",
        variant_info="CMORized product prepared by ILAMB",
        version=f"v{generate_stamp}",
    )
    out_path = ild.create_output_filename(out.attrs)
    out.to_netcdf(out_path)
```
<br>
<br>

##### ODS2.6 Global Attributes Cheatsheet

- `ds` (xr.Dataset): The xarray dataset to which global attributes will be added
- `activity_id` (str, optional): Name of the MIP activity the dataset is part of, default is "obs4MIPs"; this is the project name used in the ESGF archive
- `aux_uncertainty_id` (str, optional): Suffix appended to the variable that indicates the type of auxiliary uncertainty; must be one listed in the [obs4MIPs_aux_uncertainty_id table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_aux_uncertainty_id.json)
- `comment` (str, optional): Miscellaneous information about the data that doesn't fit into any other attribute
- `contact` (str): Contact information for the dataset; written as `First Last (email)`
- `Conventions` (str, optional): The conventions followed by the dataset; default of `CF-1.12 ODS-2.6`
- `creation_date` (str): Date the dataset was formatted; written as `YYYY-MM-DDThh:mm:ssZ`
- `dataset_contributor` (str, optional): Name of the individual or organization that generated the raw data; written as `First Last` or `Organization Name`
- `data_specs_version` (str, optional): Version of the obs4MIPs data specifications followed, default of `2.6`
- `doi` (str, optional): Digital Object Identifier for the dataset that includes the `https://doi.org/` prefix
- `frequency` (str, optional): Temporal frequency of the data; must be one listed in the [obs4MIPs_frequency table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_frequency.json)
- `grid` (str, optional): Description of the horizontal grid and/or regridding procedure with no standard format. When necessary, provide a brief description of native grid and resolution, and if the data have been re-gridded, note the re-gridding procedure and description of target grid
- `grid_label` (str, optional): Label identifying the grid; must be one listed in the [obs4MIPs_grid_label table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_grid_label.json)
- `has_aux_unc` (str, optional): Indicates if auxiliary uncertainty data is included, must be "TRUE" or "FALSE"; default is "FALSE"
- `history` (str, optional): List of applications/dates that have modified the raw data and how they were modified
- `institution` (str, optional): Where the original data were produced; should be one listed in the [obs4MIPs_institution_id table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json). If it is not in the JSON, create one like `Institution Name, City, State, Country`
- `institution_id` (str, optional): Identifier for the institution producing the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json. If not in the JSON already, create one as an abbreviation of the institution(s) name(s). E.g., "NASA-JPL" or "EmoryU".
- `license` (str, optional): License under which the data is shared, preferably with a URL to the license terms, e.g., `https://creativecommons.org/licenses/by-nc-sa/4.0/`
- `nominal_resolution` (str, optional): Nominal spatial resolution of the data; should be one listed in the [obs4MIPs_nominal_resolution table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_nominal_resolution.json), but this won't always happen, so you can provide your own description of the nominal resolution, e.g., `0.5 degree`
- `processing_code_location` (str, optional): URL or DOI linking to the script that was used to process the data
- `product` (str, optional): Type of product; must be one in the [obs4MIPs_product table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_product.json)
- `realm` (str, optional): Earth system domain of the data; must be one in the [obs4MIPs_realm table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_realm.json)
- `references` (str, optional): Reference(s) describing the data or methods used to produce the data, preferably with the DOI included; no strict format, but a common approach is `Last, FirstInitial., Title. Journal, Volume(Issue). Year. DOI.`
- `region` (str, optional): Geographic region covered by the data; must be one in the [obs4MIPs_region table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_region.json)
- `site_id` (str, optional): If the data are site observations, this is the identifier for the observation site (if one site); see the [obs4MIPs_site_id table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_site_id.json) to guide formatting. If multiple sites, set as `collection`
- `site_location` (str, optional): Description of the observation site location. If multiple sites, set as `collection`
- `source` (str, optional): A description of the methods used to produce the raw data
- `source_data_retrieval_date` (str, optional): Date the original source data were retrieved in the format `YYYY-MM-DD`
- `source_data_url` (str, optional): URL where the original source data can be requested or downloaded
- `source_id` (str, optional): Identifier for the source of the data; use one that already exists in the [obs4MIPs_source_id table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json), or create one as an abbreviation of the source's name + the version, e.g., "CLASS-1-1" or "Fluxnet-2015"
- `source_label` (str, optional): Short label for the source of the data; use one that already exists in the [obs4MIPs_source_id table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json), or create one as an abbreviation of the source's name, e.g., "CLASS" or "Fluxnet"
- `source_type` (str, optional): Type of source that produced the data; must be one in the [obs4MIPs_source_type table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_type.json)
- `source_version_number` (str, optional): Version number of the source that produced the data, e.g., `ver1.2`, `v1`, `5.0`, etc.
- `table_id` (str, optional): Identifier for the CMOR table the data would fit into; must be one in the [obs4MIPs_table_id table](https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_table_id.json)
- `title` (str, optional): A short description of the data source and contents, usually including what variable is stored, e.g., `Harmonized World Soil Database v2.0 soil carbon`
- `tracking_id` (str, optional): Unique identifier for the dataset instance, can be generated using our function, `gen_trackingid()`
- `variable_id` (str, optional): Identifier for the variable stored in the file; when possible, use a name in that is part of the [CMIP6 CV](https://airtable.com/appYNLuWqAgzLbhSq/shrKcLEdssxb8Yvcp/tblL7dJkC3vl5zQLb) or the [MIP CV](https://clipc-services.ceda.ac.uk/dreq/mipVars.html), but if not, create one that makes sense to you, e.g., `evapfrac`
- `variant_label` (str, optional): Label indicating the party that prepared the obs4MIPs data, if different from the source_id, e.g., `ILAMB`
- `variant_info` (str, optional): Description of the party that prepared the obs4MIPs data, if variant_label is not the source_id. e.g., `CMORized product prepared by ILAMB`
- `version` (str, optional): Version of the dataset instance created by the data contributor; for ILAMB, we format it as "vYYYYMMDD", where the date is when the formatting script was executed and the NetCDFs were created

After the global attributes are set, we create an output filename using the `create_output_filename()` function, which uses the global attributes to create a standardized filename in the format: `variable_id_source_id_variant_label_table_id_version.nc`. Finally, we export the Dataset as a NetCDF to the `output/` directory. This NetCDF is now ILAMB-ready and can be used as a reference dataset in ILAMB.
<br>
<br>

## How to add formatted data to ilamb3-data

Once you have produced your NetCDF, you can check the formatting using `ncdump` or display the dataset in xarray. You should also plot your data and visually inspect it to make sure it looks correct; we recommend doing this with xarray `.plot()` or using `ncview` in the command line. If everything looks good, you can push your changes to your fork and then create a pull request to merge your changes into the main ilamb3-data repository. At this point, we will review your formatting script and the resultant NetCDF. We will request changes if they need to be made, but if everything looks good, we can merge it into the main repository so that others can use it as a reference dataset in ILAMB.

Congratulations! You are now part of the ILAMB Development Collective and can contribute to the ilamb3-data repository whenever you find a dataset that you think would be useful to have in ILAMB. We are always happy to help if you have any questions or need any guidance, so please don't hesitate to reach out to us in the Issues on GitHub or via email. We also welcome any feedback on how we can make this process easier for you and others in the future.
