from pathlib import Path

import xarray as xr

from ilamb3_data import download_file, fix_time, gen_utc_timestamp

# Download source
remote_source = "https://www.ilamb.org/ILAMB-Data/DATA/nbp/HOFFMAN/nbp_1850-2010.nc"
local_source = Path("_raw")
local_source.mkdir(parents=True, exist_ok=True)
local_source = local_source / Path(remote_source).name
if not local_source.is_file():
    download_file(remote_source, str(local_source))
download_stamp = gen_utc_timestamp(local_source.stat().st_mtime)
generate_stamp = gen_utc_timestamp()

# Load the dataset for adjustments
ds = xr.open_dataset(local_source).load()
ds = ds.assign_coords({v: ds[v] for v in ds if v.endswith("_bnds")})
ds["time"] = fix_time(ds)
ds["time"].attrs["bounds"] = "time_bnds"
ds["time_bnds"].attrs["long_name"] = "time_bounds"

# Fix up the dimensions
time_range = f"{ds["time"].min().dt.year:d}{ds["time"].min().dt.month:02d}"
time_range += f"-{ds["time"].max().dt.year:d}{ds["time"].max().dt.month:02d}"

# Populate attributes
attrs = {
    "activity_id": "obs4MIPs",
    "contact": "Forrest Hoffman (forrest@climatemodeling.org)",
    "Conventions": "CF-1.12 ODS-2.5",
    "creation_data": generate_stamp,
    "dataset_contributor": "Nathan Collier",
    "data_specs_version": "2.5",
    "doi": "N/A",
    "frequency": "yr",
    "grid": "global mean data",
    "grid_label": "gm",
    "history": f"""
{download_stamp}: downloaded {remote_source};
{generate_stamp}: converted to obs4MIP format""",
    "institution": "University of California at Irvine and Oak Ridge National Laboratory",
    "institution_id": "UCI-ORNL",
    "license": "N/A",
    "nominal_resolution": "N/A",
    "processing_code_location": "https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/Hoffman/convert.py",
    "product": "derived",
    "references": "Hoffman, Forrest M., James T. Randerson, Vivek K. Arora, Qing Bao, Patricia Cadule, Duoying Ji, Chris D. Jones, Michio Kawamiya, Samar Khatiwala, Keith Lindsay, Atsushi Obata, Elena Shevliakova, Katharina D. Six, Jerry F. Tjiputra, Evgeny M. Volodin, and Tongwen Wu, (2014) Causes and Implications of Persistent Atmospheric Carbon Dioxide Biases in Earth System Models. J. Geophys. Res. Biogeosci., 119(2):141-162. doi:10.1002/2013JG002381.",
    "source": "N/A",
    "source_id": "Hoffman",
    "source_data_retrieval_date": download_stamp,
    "source_data_url": remote_source,
    "source_type": "stastical-estimate",
    "source_version_number": "1",
    "variant_label": "BE",
}

# Write out files
for var in ["nbp", "fgco2"]:
    out = ds.drop_vars([v for v in ds if (var not in v and "time" not in v)])
    out.attrs = attrs | {
        "realm": "land" if var == "nbp" else "ocean",
        "region": f"global_{"land" if var == "nbp" else "ocean"}",
        "title": f"{"Land" if var == "nbp" else "Ocean"} anthropogenic carbon flux estimates",
        "variable_id": var,
    }
    out.to_netcdf(
        "{variable_id}_{frequency}_{source_id}_{variant_label}_{grid_label}_{time_mark}.nc".format(
            **out.attrs, time_mark=time_range
        ),
    )
