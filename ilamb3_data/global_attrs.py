import json
import urllib.request
import warnings

import xarray as xr


def _load_json_from_url(url):
    with urllib.request.urlopen(url) as response:
        return json.load(response)


def set_cf(
    ds: xr.Dataset,
    *,  # keyword only for the following args
    title: str,
    institution: str,
    source: str,
    history: str,
    references: str,
    comment: str,
    conventions: str,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12.

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        title (str): Short description of the file contents.
        institution (str): Where the original data was produced.
        source (str): Method of production of the original data.
        history (str): List of applications that have modified the original data.
        references (str): References describing the data or methods used to produce it.
        comment (str): Miscellaneous information about the data or methods used.
        conventions (str): The name of the conventions followed by the dataset.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If a required global attribute is missing.
    """

    # Build and validate attributes
    attrs = {
        "title": title,
        "institution": institution,
        "source": source,
        "history": history,
        "references": references,
        "comment": comment,
        "Conventions": conventions,
    }

    # Ensure all values are explicitly set (None not allowed)
    missing = [k for k, v in attrs.items() if v is None]
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    ds.attrs.update(attrs)
    return ds


def set_ods26(
    ds: xr.Dataset,
    *,
    activity_id: str = "obs4MIPs",
    aux_uncertainty_id: str = "N/A",
    comment: str | None = None,
    contact: str = "N/A",  # First Last (email)
    Conventions: str = "CF-1.12 ODS-2.6",
    creation_date: str = "N/A",
    dataset_contributor: str | None = None,
    data_specs_version: str = "2.6",
    doi: str | None = None,
    frequency: str = "N/A",
    grid: str = "N/A",
    grid_label: str = "N/A",
    has_aux_unc: str = "FALSE",  # must be TRUE or FALSE
    history: str | None = None,
    institution: str = "N/A",
    institution_id: str = "N/A",
    license: str = "N/A",
    nominal_resolution: str = "N/A",
    processing_code_location: str = "N/A",
    product: str = "N/A",
    realm: str = "N/A",
    references: str = "N/A",
    region: str = "N/A",
    site_id: str = "N/A",
    site_location: str = "N/A",
    source: str = "N/A",
    source_data_retrieval_date: str | None = None,
    source_data_url: str = "N/A",
    source_id: str = "N/A",
    source_label: str = "N/A",
    source_type: str = "N/A",
    source_version_number: str = "N/A",
    table_id: str = "N/A",
    title: str | None = None,
    tracking_id: str = "N/A",
    variable_id: str = "N/A",
    variant_label: str = "N/A",
    variant_info: str | None = None,
    version: str | None = None,
) -> xr.Dataset:
    """
    Set required NetCDF global attributes according to CF-Conventions 1.12 and ODS-2.6.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to None.


    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        activity_id (str, optional): Name of the MIP activity the dataset is part of, default of "obs4MIPs".
        aux_uncertainty_id (str, optional): Identifier for auxiliary uncertainty data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_aux_uncertainty_id.json.
        comment (str, optional): Miscellaneous information about the data or methods used.
        contact (str): Contact information for the dataset written as <First Last (email)>.
        Conventions (str, optional): Name of the conventions followed by the dataset, default of "CF-1.12 ODS-2.6".
        creation_date (str): Date the dataset was created in the format <YYYY-MM-DDThh:mm:ssZ>.
        dataset_contributor (str, optional): Name of the individual or organization contributing the dataset.
        data_specs_version (str, optional): Version of the data specifications followed, default of "2.6".
        doi (str, optional): Digital Object Identifier for the dataset including the "https://doi.org/" prefix.
        frequency (str, optional): Temporal frequency of the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_frequency.json.
        grid (str, optional): Description of the horizontal grid and/or regridding procedure with no standard format. When necessay, provide brief description of native grid and resolution, and if data have been re-gridded, the re-gridding procedure and description of target grid.
        grid_label (str, optional): Label identifying the grid, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_grid_label.json.
        has_aux_unc (str, optional): Indicates if auxiliary uncertainty data is included, must be "TRUE" or "FALSE", default of "FALSE".
        history (str, optional): List of applications/dates that have modified the original data and how.
        institution (str, optional): Where the original data were produced, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json. If not in the JSON already, create one like <Institution Name, City, State, Country>.
        institution_id (str, optional): Identifier for the institution producing the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_institution_id.json. If not in the JSON already, create one as an abbreviation of the institution(s) name(s). E.g., "NASA-JPL" or "EmoryU".
        license (str, optional): License under which the data is shared, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_license.json.
        nominal_resolution (str, optional): Nominal spatial resolution (similar spatial resolution in km) of the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_nominal_resolution.json.
        processing_code_location (str, optional): URL or DOI pointing to the script used to process the data.
        product (str, optional): Type of product, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_product.json.
        realm (str, optional): Earth system domain of the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_realm.json.
        references (str, optional): Reference(s) describing the data or methods used to produce the data.
        region (str, optional): Geographic region covered by the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_region.json.
        site_id (str, optional): Identifier for the observation site, if one site, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_site_id.json to guide formatting. If multiple sites, set as "collection".
        site_location (str, optional): Description of the observation site location(s). If multiple sites, set as "collection".
        source (str, optional): Description of how the data were produced; could just be an expansion of abbreviations in source_id.
        source_data_retrieval_date (str, optional): Date the original source data were retrieved in the format <YYYY-MM-DD>.
        source_data_url (str, optional): URL where the original source data can be found.
        source_id (str, optional): Identifier for the source of the data, formatted as an abbreviation + version (e.g., GFED-5-0). See https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json.
        source_label (str, optional): Short label for the source of the data, formatted as an abbreviation (e.g., GFED). See "source_label" inside https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_id.json.
        source_type (str, optional): Type of source that produced the data, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_source_type.json.
        source_version_number (str, optional): Version number of the source that produced the data (e.g., "5", "ver1.2", "v1", etc).
        table_id (str, optional): Identifier for the CMOR table the data would fit into, see https://github.com/PCMDI/obs4MIPs-cmor-tables/blob/master/obs4MIPs_table_id.json.
        title (str, optional): A short description of the file contents, usually including what variable is stored (e.g., "Harmonized World Soil Database v2.0 soil carbon").
        tracking_id (str, optional): Unique identifier for the dataset instance, can be generated using output.new_tracking_id().
        variable_id (str, optional): Identifier for the variable stored in the file, see https://clipc-services.ceda.ac.uk/dreq/index/CMORvar.html or https://clipc-services.ceda.ac.uk/dreq/mipVars.html.
        variant_label (str, optional): Label indicating the party that prepared the obs4MIPs data, if different from the source_id.
        variant_info (str, optional): Description of the party that prepared the obs4MIPs data, if variant_label is not the source_id.
        version (str, optional): Version of the dataset instance created by the data contributor, formatted as "vYYYYMMDD".

    Returns:
        xr.Dataset: The dataset with updated global attributes.
    """

    base_url = "https://raw.githubusercontent.com/PCMDI/obs4MIPs-cmor-tables/master/"

    # Load controlled vocabularies from online JSON files
    # auxuncid_cv = _load_json_from_url(base_url + "obs4MIPs_aux_uncertainty_id.json")
    freq_cv = _load_json_from_url(base_url + "obs4MIPs_frequency.json")
    gridlabel_cv = _load_json_from_url(base_url + "obs4MIPs_grid_label.json")
    # hasauxunc_cv = _load_json_from_url(base_url + "obs4MIPs_has_aux_unc.json")
    instid_cv = _load_json_from_url(base_url + "obs4MIPs_institution_id.json")
    nomres_cv = _load_json_from_url(base_url + "obs4MIPs_nominal_resolution.json")
    product_cv = _load_json_from_url(base_url + "obs4MIPs_product.json")
    realm_cv = _load_json_from_url(base_url + "obs4MIPs_realm.json")
    region_cv = _load_json_from_url(base_url + "obs4MIPs_region.json")
    reqattrs_cv = _load_json_from_url(
        base_url + "obs4MIPs_required_global_attributes.json"
    )
    siteid_cv = _load_json_from_url(base_url + "obs4MIPs_site_id.json")
    sourceid_cv = _load_json_from_url(base_url + "obs4MIPs_source_id.json")
    sourcetype_cv = _load_json_from_url(base_url + "obs4MIPs_source_type.json")
    # tableid_cv = _load_json_from_url(base_url + "obs4MIPs_table_id.json")

    # Fill in required global attributes
    attrs = {
        "activity_id": activity_id,
        "aux_uncertainty_id": aux_uncertainty_id,
        "contact": contact,
        "Conventions": Conventions,
        "creation_date": creation_date,
        "data_specs_version": data_specs_version,
        "frequency": frequency,
        "grid": grid,
        "grid_label": grid_label,
        "has_aux_unc": has_aux_unc,
        "institution": institution,
        "institution_id": institution_id,
        "license": license,
        "nominal_resolution": nominal_resolution,
        "processing_code_location": processing_code_location,
        "product": product,
        "realm": realm,
        "references": references,
        "region": region,
        "site_id": site_id,
        "site_location": site_location,
        "source": source,
        "source_data_url": source_data_url,
        "source_id": source_id,
        "source_label": source_label,
        "source_type": source_type,
        "source_version_number": source_version_number,
        "table_id": table_id,
        "tracking_id": tracking_id,
        "variable_id": variable_id,
        "variant_label": variant_label,
    }

    # Add optional attributes if provided
    optional_attrs = {
        "comment": comment,
        "dataset_contributor": dataset_contributor,
        "doi": doi,
        "history": history,
        "source_data_retrieval_date": source_data_retrieval_date,
        "title": title,
        "variant_info": variant_info,
        "version": version,
    }
    for key, value in optional_attrs.items():
        if value is not None:
            attrs[key] = value

    # Sort the keys alphabetically
    attrs = dict(sorted(attrs.items()))

    # Validate required attributes
    missing = []
    for attr in reqattrs_cv["required_global_attributes"]:
        if attr not in attrs or attrs[attr] is None:
            missing.append(attr)
    if missing:
        raise ValueError(f"Missing required global attributes: {', '.join(missing)}")

    # Validate controlled vocabularies
    cv_checks = {
        # "aux_uncertainty_id": auxuncid_cv["aux_uncertainty_id"],
        "frequency": freq_cv["frequency"],
        "grid_label": gridlabel_cv["grid_label"],
        # "has_aux_unc": hasauxunc_cv["has_aux_unc"],
        "institution_id": instid_cv["institution_id"],
        "nominal_resolution": nomres_cv["nominal_resolution"],
        "product": product_cv["product"],
        "realm": realm_cv["realm"],
        "region": region_cv["region"],
        "site_id": siteid_cv["site_id"],
        "source_id": sourceid_cv["source_id"],
        "source_type": sourcetype_cv["source_type"],
    }
    for attr, cv_obj in cv_checks.items():
        if attrs[attr] not in cv_obj:
            # Warn but don't fail
            warnings.warn(
                f"Attribute '{attr}' has value '{attrs[attr]}' "
                f"which is not in the controlled vocabulary."
            )
            continue

    # set global attributes
    ds.attrs = attrs
    return ds


def set_regions(
    ds: xr.Dataset,
    creation_date: str,
    dataset_contributor: str,
    frequency: str,
    grid: str,
    grid_label: str,
    history: str,
    institution: str,
    institution_id: str,
    license: str,
    nominal_resolution: str,
    references: str,
    region: str,
    source: str,
    source_data_retrieval_date: str,
    source_id: str,
    title: str,
    variable_id: str,
    version: str,
    *,
    activity_id: str = "ILAMB",
    comment: str | None = None,
    conventions: str = "CF-1.12",
    contact: str | None = None,
    processing_code_location: str | None = None,
    source_data_url: str | None = None,
    source_version_number: str | None = None,
    tracking_id: str | None = None,
    variant_label: str | None = None,
) -> xr.Dataset:
    """
    Set NetCDF global attributes according to a hybrid of CF-Conventions 1.12. This also
    blends some useful elements from ODS-2.6 standards and adds some additional custom
    attributes specific to the ILAMB project.

    This function validates that all required attributes are provided and assigns them
    to the global attributes of the input xarray dataset. Optional fields may be set to
    None.

    Args:
        ds (xr.Dataset): The xarray dataset to which global attributes will be added.
        activity_id (str): An ID assigned to the activity dictating the creation of the
            dataset.
        comment (str): Any comments related to the dataset.
        contact (str): The First, Last name and (email) of the person or organization
            that generated the dataset.
        conventions (str): The conventions & versions that the NetCDF are aligned to.
        creation_date (str): The date when the dataset was created.
        dataset_contributor (str): The name of the person or organization that prepared
            the NetCDF.
        frequency (str): The temporal frequency of the dataset.
        grid (str): A description of the spatial grid of the dataset, including any
            transformations.
        grid_label (str): A label identifying the grid used in the dataset.
        history (str): A record of the NetCDF processing history.
        institution (str): The name of the institution responsible for generating the
            original dataset.
        institution_id (str): The identifier of the institution responsible for
            generating the original dataset.
        license (str): The license under which the dataset is distributed.
        nominal_resolution (str): The nominal spatial resolution of the dataset.
        processing_code_location (str): The URL or path to the code used for processing
            the dataset.
        references (str): References or citations related to the dataset.
        region (str): The geographical region covered by the dataset.
        source (str): A description of how the dataset was generated.
        source_data_retrieval_date (str): The date when the source data was downloaded.
        source_data_url (str): The URL from which the source data was obtained.
        source_id (str): The identifier of the source dataset, e.g., GFED-1-0.
        source_version_number (str): The version number of the source dataset.
        title (str): The title of the dataset, usually what is displayed on plots.
        tracking_id (str): A unique identifier for this NetCDF.
        variable_id (str): The identifier of the variable within the NetCDF.
        variant_label (str): The label identifying the variant of the dataset, usually
            named after the project or organization generating the dataset.
        version (str): The version of the NetCDF produced.

    Returns:
        xr.Dataset: The dataset with updated global attributes.

    Raises:
        ValueError: If any required attribute is missing or not valid.
    """

    # Fill in global attributes
    attrs = {
        "activity_id": activity_id,
        "comment": comment,
        "contact": contact,
        "Conventions": conventions,
        "creation_date": creation_date,
        "dataset_contributor": dataset_contributor,
        "frequency": frequency,
        "grid": grid,
        "grid_label": grid_label,
        "history": history,
        "institution": institution,
        "institution_id": institution_id,
        "license": license,
        "nominal_resolution": nominal_resolution,
        "processing_code_location": processing_code_location,
        "references": references,
        "region": region,
        "source": source,
        "source_data_retrieval_date": source_data_retrieval_date,
        "source_data_url": source_data_url,
        "source_id": source_id,
        "source_version_number": source_version_number,
        "title": title,
        "tracking_id": tracking_id,
        "variable_id": variable_id,
        "variant_label": variant_label,
        "version": version,
    }

    # Set the attrs
    ds.attrs = attrs
    return ds


# To-Do: We may want to create an ILAMB global attributes function that is very similar
# to ODS but is more relaxed or has different standards
# --------------------------------------------------------------------------------------
#    - variant_label : I think these should be more descriptive than ODS defines; e.g.,
#        WangMao-2021 uses two different methods to derive the same variable, and just
#        calling it essentially "variant 1", "variant 2", etc. isn't very helpful when
#        browsing by file name
#            - This should also definitely be optional and doesn't need to be provided
#    - variant_info : Obviously we would want to be descriptive about what the variant
#        is, which this already does
#    - aux_uncertainty_id : I think these should be more descriptive than ODS defines;
#        we could accept specific measures like "standard_error", "confidence_interval",
#        etc., but maintain some CVs of these
#    - aux_uncertainty_info : Include a description of how uncertainty is calculated
#    - input_sources : Maybe a list of all the source_ids that were used to create
#        the dataset, especially if they are in ILAMB registries; useful for being
#        careful about interpreting benchmarks
#    - drop tracking_id if we don't need it
#    - drop redundant data_specs_version (Conventions carries that, and it's less
#        ambiguous)
#    - activity_id should have activity_info or something to describe the activity
