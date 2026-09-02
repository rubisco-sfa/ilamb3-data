from pathlib import Path

import cftime as cf
import xarray as xr

import ilamb3_data as ild

# Variables of interest
var = "cSoil"
uncert = "coefficient_of_variation"

# Download final 0.5 degree rasters using server-side WCS averaging
coverages = {
    var: "ocs_0-30cm_mean",
    uncert: "ocs_0-30cm_uncertainty",
}
local_sources = {
    name: ild.download.from_wcs(
        "https://maps.isric.org/mapserv",
        coverage,
        0.5,
        Path(__file__).parent
        / "_raw"
        / f"{coverage}_wcs_average_0p5deg.tif",
        map_file="/map/ocs.map",
        interpolation="AVERAGE",
    )
    for name, coverage in coverages.items()
}
download_stamp = ild.output.utc_timestamp(
    max(path.stat().st_mtime for path in local_sources.values())
)
creation_stamp = ild.output.utc_timestamp()

# Read the WCS rasters and mask only exact fill sentinels
data_vars = {}
for name, path in local_sources.items():
    with xr.open_dataset(path, engine="rasterio") as source:
        da = source["band_data"].squeeze("band", drop=True)
        da = da.where(~da.isin([-32768, -1, 32767])).load()
        data_vars[name] = da.rename({"x": "lon", "y": "lat"}).drop_vars(
            "spatial_ref", errors="ignore"
        )

# Decode source scaling and create the fixed time axis
ds = xr.Dataset(data_vars).expand_dims(time=[0])
ds[var].attrs = {"units": "t ha-1"}
ds[uncert] *= 0.1  # Source stores 10 * (Q0.95 - Q0.05) / Q0.50
ds[uncert].attrs = {"units": "1"}
ds = ild.time.create_time_axis(
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
    cell_methods="area: mean where land",
    target_dtype="float32",
    convert=True,
    compression=compression,
)
ds = ild.variable.standardize(
    ds,
    uncert,
    units="1",
    standard_name=None,
    long_name="Relative 90 Percent Prediction Interval Width of Soil Carbon Stock",
    cell_methods="area: mean where land",
    target_dtype="float32",
    extra_attrs={"comment": "Defined by SoilGrids as (Q0.95 - Q0.05) / Q0.50."},
    compression=compression,
)

# Set global metadata for this experimental WCS comparison
ds = ild.global_attrs.set_ods26(
    ds,
    activity_id="ILAMB",
    aux_uncertainty_id=uncert,
    comment=(
        "Experimental WCS-derived comparison product. The ISRIC WCS may include "
        "nodata in server-side averages near coastlines; only exact fill sentinels "
        "were masked so mixed coastal artifacts remain visible. The ancillary field "
        "is the spatial mean of pixel-level relative uncertainty, not the formal "
        "uncertainty of the aggregated carbon stock."
    ),
    contact="Nathan Collier (nathaniel.collier@gmail.com)",
    creation_date=creation_stamp,
    dataset_contributor="Nathan Collier",
    doi="https://doi.org/10.5194/soil-7-217-2021",
    frequency="fx",
    grid=(
        "Native 250 m Interrupted Goode Homolosine data reprojected and averaged "
        "by the ISRIC WCS directly to a 0.5x0.5 degree EPSG:4326 grid"
    ),
    grid_label="gn",
    has_aux_unc="TRUE",
    history=(
        f"{download_stamp}: downloaded {', '.join(coverages.values())} from the "
        "ISRIC WCS as 0.5 degree EPSG:4326 rasters using AVERAGE interpolation; "
        f"{creation_stamp}: masked exact fill sentinels, converted carbon stock from "
        "t ha-1 to kg m-2, and decoded the SoilGrids uncertainty ratio."
    ),
    institution="ISRIC - World Soil Information, Wageningen, Netherlands",
    institution_id="ISRIC",
    license="Creative Commons Attribution 4.0 International License (CC BY 4.0)",
    nominal_resolution="0.5 degree",
    processing_code_location=(
        "https://github.com/rubisco-sfa/ilamb3-data/blob/main/"
        "data/SoilGrids-2-0/convert_wcs.py"
    ),
    product="derived",
    realm="land",
    references=(
        "Poggio, L., de Sousa, L. M., Batjes, N. H., Heuvelink, G. B. M., Kempen, "
        "B., Ribeiro, E., and Rossiter, D. (2021): SoilGrids 2.0: producing soil "
        "information for the globe with quantified spatial uncertainty, SOIL, 7, "
        "217-240, https://doi.org/10.5194/soil-7-217-2021."
    ),
    region="global_land",
    source=(
        "Soil organic carbon stock predictions derived with Quantile Random Forests "
        "from soil profiles and environmental covariates"
    ),
    source_data_retrieval_date=download_stamp,
    source_data_url="https://maps.isric.org/mapserv?map=/map/ocs.map",
    source_id="SoilGrids-2-0",
    source_label="SoilGrids",
    source_type="AI_upscaling",
    source_version_number="2.0",
    title="SoilGrids 2.0 Soil Organic Carbon Stock for 0-30 cm (WCS comparison)",
    tracking_id=ild.output.new_tracking_id(),
    variable_id=var,
    variant_label="ILAMB",
    variant_info="Experimental WCS-average comparison prepared by ILAMB",
    version=f"v{creation_stamp[:10].replace('-', '')}",
)

# Write a distinct comparison file without replacing the primary product
ds = ild.output.order_dimensions(ds)
out_path = Path(ild.output.filename_from_attrs(ds.attrs))
out_path = out_path.with_stem(f"{out_path.stem}_wcs")
ds.to_netcdf(out_path)
