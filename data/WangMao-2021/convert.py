import xarray as xr

import ilamb3_data as ild

# Variable of interest
var = "mrsol"

# Download
urls = [
    "https://drive.google.com/file/d/1-UZ7FEHbqoAXHq6Wa3b5zme8vPeX02kf/view?usp=drive_link",
    "https://drive.google.com/file/d/1-0Ze2XfDWyOqgxpetQqf5HeVxDxX_y7_/view?usp=drive_link",
]
input_netcdfs = ild.download.from_google_drive(urls)
download_stamp = ild.output.utc_timestamp(input_netcdfs[0].stat().st_mtime)

# Describe some attrs unique to each netcdf
per_netcdf_attrs = {
    "olc_ors": {
        "source": "optimal linear combination technique of (Hobeichi, 2018)",
        "uncert_long_name": "Estimated standard deviation of optimal-linear-combination volumetric soil moisture",
        "uncert_comment": "Calculated from the adjusted source soil moisture datasets and adjusted optimal-linear-combination weights",
    },
    "ec_ors": {
        "source": "emergent constraint technique of (Mystakidis, 2016)",
        "uncert_long_name": "Estimated standard deviation of emergent-constraint volumetric soil moisture",
        "uncert_comment": "Regression prediction standard deviation where a significant emergent-constraint relationship exists; otherwise, the standard deviation across source soil moisture datasets.",
    },
}

# Open dataset and rename variable to match CMIP5/6 naming conventions
for path in input_netcdfs:
    netcdf = next(path.rglob("*.nc"))
    method = netcdf.stem  # Indicates OLC method or EC method
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(netcdf, decode_times=time_coder)
    ds = ds.rename({"sm": var, "std": f"{var}_sd"})

    # Convert volumetric soil moisture and its standard deviation to kg m-2
    thickness = ds["depth_bnds"].diff(dim="bnds").squeeze("bnds", drop=True)
    conversion = thickness * 998.0  # Multiply by water density
    for name in (var, f"{var}_sd"):
        ds[name] = ds[name] * conversion
        ds[name].attrs["units"] = "kg m-2"

    # Standardize (CF) the time, lat, and lon coordinates, and add bounds dimension
    ds = ild.time.standardize(ds, bounds_frequency="MS")
    ds = ild.lat.standardize(ds)
    ds = ild.lon.standardize(ds)
    ds = ild.depth.build_from_bounds(
        ds,
        bounds=ds["depth_bnds"].values,
        units="m",
        positive="down",
        long_name="depth of soil",
    )
    ds = ild.bounds.add_rectilinear_bounds(ds, overwrite=True)

    # Look up the variable in the CMIP6 variable table
    var_info = ild.variable.lookup_cmip6(var, var)

    # Set compression levels for variables
    compression = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "chunksizes": (6, 1, 180, 360),
    }

    # Format the variable attributes
    ds = ild.variable.standardize(
        ds,
        var,
        units=var_info["variable_units"],
        standard_name=var_info["cf_standard_name"],
        long_name=var_info["variable_long_name"],
        ancillary_variables=f"{var}_sd",
        target_dtype="float32",
        convert=False,  # Ignores what is set as `units=` and retains original
        compression=compression,
    )

    # Format the ancillary var attrs
    ds = ild.variable.standardize(
        ds,
        f"{var}_sd",
        units=ds[f"{var}_sd"].attrs["units"],
        standard_name=f"{ds[var].attrs['standard_name']} standard_deviation",
        long_name=per_netcdf_attrs[method]["uncert_long_name"],
        target_dtype="float32",
        convert=False,  # Ignores what is set as `units=` and retains original
        extra_attrs={
            "comment": per_netcdf_attrs[method]["uncert_comment"],
        },
        compression=compression,
    )

    # Set the global attrs
    out = ild.global_attrs.set_ods26(
        ds,
        activity_id="ILAMB",
        aux_uncertainty_id="sd",
        contact="Yaoping Wang (wang7@ornl.gov)",
        creation_date=ild.output.utc_timestamp(),
        dataset_contributor="Morgan Steckler",
        doi="https://doi.org/10.6084/m9.figshare.13661312.v1",
        frequency="mon",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="TRUE",
        history=f"Downloaded from {urls[0]} and {urls[1]} on {download_stamp}. Converted to CF-compliant and ODS-aligned product by ILAMB on {ild.output.utc_timestamp()}. Converted units from volumetric soil moisture to kg m-2 by multiplying by the thickness of the soil layer and the density of water.",
        institution="Oak Ridge National Laboratory, TN, USA",  # this is mispelled as "Oakridge" on obs4mip-cmor-tables
        institution_id="ORNL",
        license="Creative Commons Attribution 4.0 International License (CC BY 4.0)",
        nominal_resolution="0.5 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/tree/main/data/WangMao-2021/convert.py",
        product="derived",
        realm="land",
        references="Wang, Y., Mao, J., Jin, M., Hoffman, F. M., Shi, X., Wullschleger, S. D., & Dai, Y. (2021) Development of observation-based global multilayer soil moisture products for 1970 to 2016, Earth Syst. Sci. Data, 13, 4385-4405, https://doi.org/10.5194/essd-13-4385-2021.",
        region="global_land",
        source=f"{ds[var].attrs['long_name']} estimated using the {per_netcdf_attrs[method]['source']} on ESACCIv4.5, GLEAMv3.3a, CERA20C, ERA20C, ERA-Interim, ERA5, coupled ESM-simulated datasets, and offline LSM-simulated datasets",
        source_id="WangMao-2021",
        source_data_retrieval_date=download_stamp,
        source_data_url=",".join(urls),
        source_label="WangMao",
        source_type="AI_upscaling",
        source_version_number="2021",
        title=f"Wang & Mao (2021) Observation-Based Global Multilayer Soil Moisture Product ({method.replace('_', '-').upper()} {var})",
        tracking_id=ild.output.new_tracking_id(),
        variable_id=var,
        variant_label=method.replace("_", "-").upper(),
        variant_info=f"Version of {var} estimated using the {per_netcdf_attrs[method]['source']}",
        version=f"v{ild.output.utc_timestamp()[:10].replace('-', '')}",
    )
    out = ild.output.order_dimensions(out)
    out_path = ild.output.filename_from_attrs(out.attrs)
    out.to_netcdf(out_path)
