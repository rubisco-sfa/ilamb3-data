import xarray as xr

import ilamb3_data as ild

# Download the data
url = "https://www.ilamb.org/ILAMB-Data/DATA/nbp/HOFFMAN/nbp_1850-2010.nc"
netcdf = ild.download.from_html(url)
download_stamp = ild.output.utc_timestamp(netcdf.stat().st_mtime)  # type: ignore

# Load the dataset
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_dataset(netcdf, decode_times=time_coder)  # type: ignore

# Standardize the dimensions (data are a global mean, so there isn't lat or lon)
ds = ild.time.standardize(ds, bounds_frequency="YS")

# Describe variable-specific metadata in one place
bounds = {"lbnd": "Lower", "ubnd": "Upper"}
variable_specs = {
    "nbp": {
        "realm": "land",
        "uncertainty_comment": "Uncertainty in land uptake calculated by carbon-budget mass balance, using the uncertainty envelope of the adjusted ocean-uptake trajectory",
        "source_description": "derived as a global carbon-budget residual using historical anthropogenic emissions, atmospheric CO₂ observations from ice cores, Mauna Loa, and NOAA, and an fgco2 ocean-uptake estimate",
    },
    "fgco2": {
        "realm": "ocean",
        "uncertainty_comment": "Uncertainty envelope of the Khatiwala et al. ocean-uptake trajectory after scaling it to the 2010 ocean-inventory estimate and adjusting it to an 1850 baseline",
        "source_description": "derived from in situ ship-based ocean carbon, tracer, and salinity measurements using a maximum-entropy-constrained Green's-function ocean-transport inversion",
    },
}

# Create one NetCDF per variable
for var, spec in variable_specs.items():
    source_bounds_name = f"{var}_bnds"
    bound_names = [f"{var}{suffix}" for suffix in bounds]
    out = ds[[var, source_bounds_name, "time_bnds"]].copy()

    # Extract the lower and upper uncertainty bounds
    source_bounds = out[source_bounds_name]
    out = out.assign(
        {
            name: source_bounds.isel(bnds=index, drop=True)
            for index, name in enumerate(bound_names)
        }
    ).drop_vars(source_bounds_name)

    # Get CMIP6 variable information and standardize the primary variable
    var_info = ild.variable.lookup_cmip6(var, var)
    out = ild.variable.standardize(
        out,
        var,
        units=var_info["variable_units"],
        standard_name=var_info["cf_standard_name"],
        long_name=var_info["variable_long_name"],
        ancillary_variables=" ".join(bound_names),
        target_dtype="float32",
    )
    print(var_info["variable_units"])

    # Standardize the uncertainty variables
    for suffix, label in bounds.items():
        out = ild.variable.standardize(
            out,
            f"{var}{suffix}",
            units=var_info["variable_units"],
            standard_name=f"{out[var].attrs['standard_name']} {label.lower()}_bound",
            long_name=f"{label} Bound of Predicted {var_info['variable_long_name']}",
            target_dtype="float32",
            extra_attrs={"comment": spec["uncertainty_comment"]},
        )

    # Clean up straggling attributes
    out[var].encoding.pop("missing_value", None)
    out[var].attrs.pop("bounds", None)

    # Set global attributes
    out = ild.global_attrs.set_ods26(
        out,
        activity_id="ILAMB",
        aux_uncertainty_id=", ".join(bounds),
        contact="Forrest Hoffman (forrest@climatemodeling.org)",
        creation_date=ild.output.utc_timestamp(),
        dataset_contributor="Morgan Steckler",
        frequency="yr",
        grid_label="gm",
        has_aux_unc="TRUE",
        history=f"Downloaded on {download_stamp}. Converted to CF-compliant and ODS-aligned product by ILAMB on {ild.output.utc_timestamp()}.",
        institution="University of California at Irvine, Irvine, CA, USA; Oak Ridge National Laboratory, Oak Ridge, TN, USA",
        institution_id="UCI-ORNL",
        license="Creative Commons Attribution 4.0 International License (CC BY 4.0)",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/Hoffman-1-0/convert.py",
        product="derived",
        realm=spec["realm"],
        references="Hoffman, F.M., et al. (2014): Causes and Implications of Persistent Atmospheric Carbon Dioxide Biases in Earth System Models. J. Geophys. Res. Biogeosci., 119(2):141-162. https://doi.org/10.1002/2013JG002381.",
        region=f"global_{spec['realm']}",
        source=f"{out[var].attrs['long_name']} {spec['source_description']}",
        source_id="Hoffman-1-0",
        source_data_retrieval_date=download_stamp,
        source_data_url=url,
        source_label="Hoffman",
        source_type="insitu",
        source_version_number="1.0",
        title=f"{spec['realm'].capitalize()} Anthropogenic Carbon Flux Estimates",
        tracking_id=ild.output.new_tracking_id(),
        variable_id=var,
        variant_label="ILAMB",
        variant_info="Formatted product prepared by ILAMB",
        version=f"v{ild.output.utc_timestamp()[:10].replace('-', '')}",
    )

    # Write output
    out = ild.output.order_dimensions(out)
    out_path = ild.output.filename_from_attrs(out.attrs)
    out.to_netcdf(out_path)
