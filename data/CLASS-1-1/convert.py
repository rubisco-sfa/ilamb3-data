import xarray as xr

import ilamb3_data as ild

# Download
url = (
    "https://thredds.nci.org.au/thredds/catalog/ks32/ARCCSS_Data/CLASS/v1-1/catalog.xml"
)
input_netcdfs = ild.download.from_thredds(url, pattern="CLASS_v1-1_*.nc")
download_stamp = ild.output.utc_timestamp(input_netcdfs[0].stat().st_mtime)

# Open dataset and rename some variables to match CMIP5/6 naming conventions
time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
ds = xr.open_mfdataset(input_netcdfs, decode_times=time_coder)
ds = ds.rename(
    {v: str(v).replace("hfds", "hfdsl").replace("rs", "rns") for v in ds.data_vars}
)


# Standardize (CF) the time, lat, and lon coordinates, and add bounds dimension
ds = ild.time.standardize(ds, bounds_frequency="MS")
ds = ild.lat.standardize(ds)
ds = ild.lon.standardize(ds)
ds = ild.bounds.add_rectilinear_bounds(ds, overwrite=True)

# Describe some attrs unique to each variable
per_var_attrs = {
    "dw": {
        "uncert_basis": "GRACE-propagated",
        "source_desc": "Monthly storage change derived from JPL GRACE RL06 mascon anomalies",
    },
    "pr": {
        "uncert_basis": "Kriging-based",
        "source_desc": "Monthly REGEN gauge-based precipitation with kriging uncertainty",
    },
    "rns": {
        "uncert_basis": "Observation-constrained",
        "source_desc": "Synthesis of five net-radiation products constrained by flux and radiation sites",
    },
    "hfdsl": {
        "uncert_basis": "Flux-tower-constrained",
        "source_desc": "Synthesis of five ground-heat products constrained by flux towers",
    },
    "hfss": {
        "uncert_basis": "Flux-tower-constrained",
        "source_desc": "Synthesis of six sensible-heat products constrained by flux towers",
    },
    "mrro": {
        "uncert_basis": "Streamflow-constrained",
        "source_desc": "LORA synthesis of 11 modeled runoff estimates constrained by streamflow",
    },
    "hfls": {
        "uncert_basis": "Flux-tower-constrained",
        "source_desc": "DOLCE synthesis of six evapotranspiration products constrained by flux towers",
    },
}

# Use source metadata for variables absent from the CMIP6 lookup
source_name_attr = {"dw": "long_name", "rns": "standard_name"}

# Set compression levels for variables
compression = {
    "zlib": True,
    "complevel": 4,
    "shuffle": True,
    "chunksizes": (1, 360, 720),
}

# Write a NetCDF for each variable and its uncertainty
for var in (str(v) for v in ds.data_vars if not str(v).endswith(("_sd", "_bnds"))):
    # Select variable and its uncertainty, and any bounds variables
    uncert = f"{var}_sd"
    out = ds[[var, uncert, *[v for v in ds if str(v).endswith("_bnds")]]]

    # Clean up special cases that don't have a CMIP6 standard name or long name
    if var in source_name_attr:
        long_name = out[var].attrs[source_name_attr[var]].replace("_", " ").title()
        var_info = {
            "cf_standard_name": out[var].attrs.get("standard_name")
            or long_name.replace(" ", "_").lower(),
            "variable_long_name": long_name,
        }
    # Otherwise, look up the variable in the CMIP6 variable table
    else:
        var_info = ild.variable.lookup_cmip6(var, var)
    var_info["variable_units"] = out[var].attrs.get("units", "")

    # Format the variable attributes
    out = ild.variable.standardize(
        out,
        var,
        units=var_info["variable_units"],
        standard_name=var_info["cf_standard_name"],
        long_name=var_info["variable_long_name"],
        ancillary_variables=uncert,
        target_dtype="float32",
        convert=False,  # Ignores what is set as `units=` and retains original
        compression=compression,
    )
    out[var].attrs.pop("ALMA_short_name", None)  # Drop straggling attribute

    # Format the ancillary var attrs
    out = ild.variable.standardize(
        out,
        uncert,
        units=out[uncert].attrs.get("units", ""),
        standard_name=f"{out[var].attrs['standard_name']} standard_deviation",
        long_name=f"{per_var_attrs[var]['uncert_basis']} standard deviation of {out[var].attrs['long_name'].lower()}",
        target_dtype="float32",
        convert=False,  # Ignores what is set as `units=` and retains original
        compression=compression,
    )
    out[uncert].attrs.pop("cell_methods", None)  # Drop incorrect attribute

    # Set the global attrs
    out = ild.global_attrs.set_ods26(
        out,
        activity_id="ILAMB",
        aux_uncertainty_id="sd",
        contact="Sanaa Hobeichi (s.hobeichi@unsw.edu.au)",
        creation_date=ild.output.utc_timestamp(),
        dataset_contributor="Nathan Collier",
        doi="https://doi.org/10.25914/5c872258dc183",
        frequency="mon",
        grid="0.5x0.5 degree latitude x longitude",
        grid_label="gn",
        has_aux_unc="TRUE",
        history=f"""Downloaded from {url} on {download_stamp}. Converted to CF-compliant and ODS-aligned product by ILAMB on {ild.output.utc_timestamp()}.""",
        institution="University of New South Wales, Sydney, New South Wales, AUS",
        institution_id="UNSW",
        license="Creative Commons Attribution-ShareAlike 4.0 International License (CC BY-SA 4.0)",
        nominal_resolution="0.5 degree",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/CLASS-1-1/convert.py",
        product="derived",
        realm="land",
        references="Hobeichi, S., Abramowitz, G., & Evans, J. (2020) Conserving Land-Atmosphere Synthesis Suite (CLASS), Journal of Climate, 33(5), 10.1175/JCLI-D-19-0036.1.",
        region="global_land",
        source=f"{per_var_attrs[var]['source_desc']}",
        source_id="CLASS-1-1",
        source_data_retrieval_date=download_stamp,
        source_data_url=url,
        source_label="CLASS",
        source_type="AI_upscaling",
        source_version_number="1.1",
        title=f"Conserving Land-Atmosphere Synthesis Suite ({var})",
        tracking_id=ild.output.new_tracking_id(),
        variable_id=var,
        variant_label="ILAMB",
        variant_info="CF-compliant and ODS-aligned product prepared by ILAMB",
        version=f"v{ild.output.utc_timestamp()[:10].replace('-', '')}",
    )
    out = ild.output.order_dimensions(out)
    out_path = ild.output.filename_from_attrs(out.attrs)
    out.to_netcdf(out_path)
