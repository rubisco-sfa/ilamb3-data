from datetime import datetime
from pathlib import Path

import xarray as xr

from ilamb3_data import (
    create_output_filename,
    download_file,
    gen_trackingid,
    gen_utc_timestamp,
    get_cmip6_variable_info,
    set_ods_coords,
    set_ods_global_attrs,
    set_time_attrs,
    set_var_attrs,
)

# Download the data
remote_source = "https://www.ilamb.org/ILAMB-Data/DATA/nbp/HOFFMAN/nbp_1850-2010.nc"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / Path(remote_source).name
if not local_source.is_file():
    download_file(remote_source, str(local_source))

# Set timestamps and tracking id
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
creation_stamp = gen_utc_timestamp()
today_stamp = datetime.now().strftime("%Y%m%d")
tracking_id = gen_trackingid()

# Load the dataset for adjustments
ds = xr.open_dataset(local_source).load()
ds = set_time_attrs(ds)

# Get attribute info for fgco2 and nbp
fgco2_info = get_cmip6_variable_info("fgco2")
nbp_info = get_cmip6_variable_info("nbp")

# Set correct attribute information for the vars
ds = set_var_attrs(
    ds,
    var="fgco2",
    cmip6_units=fgco2_info["variable_units"],
    cmip6_standard_name=fgco2_info["cf_standard_name"],
    cmip6_long_name=fgco2_info["variable_long_name"],
    convert=False,
)
ds = set_var_attrs(
    ds,
    var="nbp",
    cmip6_units=nbp_info["variable_units"],
    cmip6_standard_name=nbp_info["cf_standard_name"],
    cmip6_long_name=nbp_info["variable_long_name"],
    convert=False,
)

# Create {var}lbnd and {var}ubnd ancillary variables from {var}_bnds variables
ds.fgco2.attrs["ancillary_variables"] = "fgco2lbnd fgco2ubnd"
ds["fgco2lbnd"] = ds.fgco2_bnds[:, 0]
ds["fgco2lbnd"].attrs.pop("units", None)
ds["fgco2ubnd"] = ds.fgco2_bnds[:, 1]
ds["fgco2ubnd"].attrs.pop("units", None)

ds.nbp.attrs["ancillary_variables"] = "nbplbnd nbpubnd"
ds["nbplbnd"] = ds.nbp_bnds[:, 0]
ds["nbplbnd"].attrs.pop("units", None)
ds["nbpubnd"] = ds.nbp_bnds[:, 1]
ds["nbpubnd"].attrs.pop("units", None)

# Remove the old {var}_bnds variables and clean up attrs
ds = ds.drop_vars(["fgco2_bnds", "nbp_bnds"])
for var in ds.variables:
    ds[var].encoding.pop("missing_value", None)
ds["nbp"].attrs.pop("bounds", None)
ds["fgco2"].attrs.pop("bounds", None)

# Extract time info for file naming purposes
time_range = f"{ds['time'].min().dt.year:d}{ds['time'].min().dt.month:02d}"
time_range += f"-{ds['time'].max().dt.year:d}{ds['time'].max().dt.month:02d}"

# Set global attributes and export
for var in ["nbp", "fgco2"]:
    # Create one ds per variable
    out_ds = ds.drop_vars([v for v in ds if (var not in v and "time" not in v)])

    # Define varibable-dependant attributes
    dynamic_attrs = {
        "realm": "land" if var == "nbp" else "ocean",
        "region": f"global_{'land' if var == 'nbp' else 'ocean'}",
        "title": f"{'Land' if var == 'nbp' else 'Ocean'} anthropogenic carbon flux estimates",
        "variable_id": var,
    }

    # Set global attributes
    out_ds = set_ods_global_attrs(
        out_ds,
        activity_id="obs4MIPs",
        aux_variable_id=f"{var}lbnd {var}ubnd",
        comment="Not yet obs4MIPs compliant: 'version' attribute is temporary; source_id not in obs4MIPs yet; The Boundary variables 'time_bnds' should not have the attributes: '['units']'",
        contact="Forrest Hoffman (forrest@climatemodeling.org)",
        conventions="CF-1.12 ODS-2.5",
        creation_date=creation_stamp,
        dataset_contributor="Nathan Collier",
        data_specs_version="2.5",
        doi="N/A",
        external_variables="N/A",
        frequency="yr",
        grid="global mean data",
        grid_label="gm",
        has_auxdata="True",
        history=f"""
{download_stamp}: downloaded {remote_source};
{creation_stamp}: converted to obs4MIP format""",
        institution="University of California at Irvine and Oak Ridge National Laboratory",
        institution_id="UCI-ORNL",
        license="N/A",
        nominal_resolution="N/A",
        processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/Hoffman/convert.py",
        product="derived",
        realm=dynamic_attrs["realm"],
        references="Hoffman, Forrest M., James T. Randerson, Vivek K. Arora, Qing Bao, Patricia Cadule, Duoying Ji, Chris D. Jones, Michio Kawamiya, Samar Khatiwala, Keith Lindsay, Atsushi Obata, Elena Shevliakova, Katharina D. Six, Jerry F. Tjiputra, Evgeny M. Volodin, and Tongwen Wu, (2014) Causes and Implications of Persistent Atmospheric Carbon Dioxide Biases in Earth System Models. J. Geophys. Res. Biogeosci., 119(2):141-162. doi:10.1002/2013JG002381.",
        region=dynamic_attrs["region"],
        source="N/A",
        source_id="Hoffman",
        source_data_retrieval_date=download_stamp,
        source_data_url=remote_source,
        source_label="Hoffman",
        source_type="stastical-estimate",
        source_version_number="1",
        title=dynamic_attrs["title"],
        tracking_id=tracking_id,
        variable_id=var,
        variant_label="REF",
        variant_info="CMORized product prepared by ILAMB and CMIP IPO",
        version=today_stamp,
    )

    out_ds = set_ods_coords(out_ds)
    out_path = create_output_filename(out_ds.attrs, time_range)
    out_ds.to_netcdf(out_path)

    print(f"Exported to {out_path}")
