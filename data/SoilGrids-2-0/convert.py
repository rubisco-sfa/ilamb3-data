import cftime as cf
import xarray as xr

import ilamb3_data as ild

# Variables of interest
var = "cSoil"
uncert = "q05_q95_ratio"

# Read the native 250m remote VRTs and average directly onto a 0.5 degree EPSG:4326 grid
urls = {
    var: "https://files.isric.org/soilgrids/latest/data/ocs/ocs_0-30cm_mean.vrt",
    uncert: "https://files.isric.org/soilgrids/latest/data/ocs/ocs_0-30cm_uncertainty.vrt",
}
netcdfs = {
    name: ild.download.from_raster(
        source,
        0.5,
        dtype="float32",
        resampling="average",
    )
    for name, source in urls.items()
}
download_stamp = ild.output.utc_timestamp(
    max(path.stat().st_mtime for path in netcdfs.values())
)

# Read the reprojected rasters and split the shared band name into data variables
ds = xr.open_mfdataset(
    list(netcdfs.values()),
    engine="rasterio",
    combine="nested",  # Stack along a new dimension
    concat_dim="variable",  # Call the temporary dimension "variable"
)
ds = (
    ds["band_data"]
    .assign_coords(variable=list(netcdfs))
    .squeeze("band", drop=True)
    .rename({"x": "lon", "y": "lat"})
    .drop_vars("spatial_ref", errors="ignore")  # Drop straggling coordinate
    .to_dataset(dim="variable")
    .load()  # Prevent lazy evaluation from keeping the source files open
)

# Add units and decode the SoilGrids uncertainty ratio
ds[var].attrs = {"units": "t ha-1"}
ds[uncert] *= 0.1  # Source stores 10 * (Q0.95 - Q0.05) / Q0.50
ds[uncert].attrs = {"units": "1"}

# Decode source scaling and create the fixed time axis
ds = ds.expand_dims(time=[0])  # Associate time dimension with each variable
ds = ild.time.create_time_axis(  # Fill in the actual time coordinates and bounds
    ds,
    cf.DatetimeNoLeap(1905, 3, 31),
    cf.DatetimeNoLeap(2016, 7, 4),
    "fx",
    calendar="noleap",
)

# Standardize dimensions and coordinates
ds = ds.sortby("lat")
ds = ild.lat.standardize(ds)
ds = ild.lon.standardize(ds)
ds = ild.bounds.add_rectilinear_bounds(ds)

# Look up and standardize variable metadata
var_info = ild.variable.lookup_cmip6(var, var)
compression = {
    "zlib": True,
    "complevel": 4,
    "shuffle": True,
    "chunksizes": (1, 180, 360),
}
ds = ild.variable.standardize(
    ds,
    var,
    units=var_info["variable_units"],
    standard_name=var_info["cf_standard_name"],
    long_name=var_info["variable_long_name"],
    ancillary_variables=uncert,
    cell_methods="area: mean where land",  # download.from_raster(resampling="average")
    target_dtype="float32",
    convert=True,  # t ha-1 to kg m-2
    compression=compression,
)
ds = ild.variable.standardize(
    ds,
    uncert,
    units="1",
    standard_name=f"{var_info['cf_standard_name']} {uncert}",
    long_name="Relative 90 Percent Prediction Interval Width of Soil Carbon Stock",
    cell_methods="area: mean where land",
    target_dtype="float32",
    extra_attrs={"comment": "Defined by SoilGrids as (Q0.95 - Q0.05) / Q0.50"},
    compression=compression,
)

# Set global metadata
ds = ild.global_attrs.set_ods26(
    ds,
    activity_id="ILAMB",
    aux_uncertainty_id=uncert,
    comment=(
        "The ancillary field is the spatial mean of pixel-level relative uncertainty, "
        "not the formal uncertainty of the aggregated 0.5 degree carbon stock"
    ),
    contact="Nathan Collier (nathaniel.collier@gmail.com)",
    creation_date=ild.output.utc_timestamp(),
    dataset_contributor="Nathan Collier",
    doi="https://doi.org/10.5194/soil-7-217-2021",
    frequency="fx",
    grid=(
        "Native 250 m Interrupted Goode Homolosine data reprojected by area-weighted averaged to 0.5x0.5 degree latitude x longitude"
    ),
    grid_label="gr",
    has_aux_unc="TRUE",
    history=(
        f"{download_stamp}: read {', '.join(urls.values())}, excluded each VRT's "
        "native nodata value, and reprojected and area-weighted average-resampled the "
        "native 250 m pixels directly to 0.5 degree with rasterio; "
        f"{ild.output.utc_timestamp()}: converted carbon stock from t ha-1 to kg m-2 "
        "and decoded the SoilGrids uncertainty ratio by multiplying times 0.1."
    ),
    institution="ISRIC - World Soil Information, Wageningen, Netherlands",
    institution_id="ISRIC",
    license="Creative Commons Attribution 4.0 International License (CC BY 4.0)",
    nominal_resolution="0.5 degree",
    processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/SoilGrids-2-0/convert.py",
    product="derived",
    realm="land",
    references=(
        "Poggio, L., de Sousa, L.M., Batjes, N.H., Heuvelink, G.B.M., Kempen, "
        "B., Ribeiro, E., and Rossiter, D. (2021) SoilGrids 2.0: producing soil "
        "information for the globe with quantified spatial uncertainty, SOIL, 7, "
        "217-240, https://doi.org/10.5194/soil-7-217-2021."
    ),
    region="global_land",
    source="Soil organic carbon stock ML Forest Model predictions derived from WoSIS soil profile observations and climate, land cover, and terrain morphology covariates",
    source_data_retrieval_date=download_stamp,
    source_data_url="https://docs.isric.org/globaldata/soilgrids/",
    source_id="SoilGrids-2-0",
    source_label="SoilGrids",
    source_type="AI_upscaling",
    source_version_number="2.0",
    title="SoilGrids 2.0 Soil Organic Carbon Stock for 0-30 cm",
    tracking_id=ild.output.new_tracking_id(),
    variable_id=var,
    variant_label="ILAMB",
    variant_info="CF-compliant and ODS-aligned product prepared by ILAMB",
    version=f"v{ild.output.utc_timestamp()[:10].replace('-', '')}",
)

# Write output
ds = ild.output.order_dimensions(ds)
out_path = ild.output.filename_from_attrs(ds.attrs)
ds.to_netcdf(out_path)
