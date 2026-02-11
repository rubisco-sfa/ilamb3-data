#!/usr/bin/env python3
"""
Convert AmeriFlux / FLUXNET FULLSET daily (_DD_) CSV files to ILAMB3-ready
NetCDF files (site-based), WITHOUT requiring internet access.

Why offline?
------------
ilamb3_data.set_ods26_global_attrs() fetches obs4MIPs controlled vocabulary JSON
over HTTPS. On some HPC systems this fails due to SSL certificate verification.
This script sets the required global attributes locally so the ILAMB3 validator
can run without network.

Key behaviors
-------------
- reads one CSV (--csv) or many (--csv-dir + --pattern)
- constructs ds(time, site) and assigns lat/lon as site coordinates
- converts TA_F (degC) -> tas (K)
- standardizes time + time bounds + lat/lon attrs using ilamb3_data helpers
- writes one file per variable to:
    <repo>/data/Ameriflux/_output_offline/
  (anchored to the script location to avoid duplicate "data/Ameriflux/data/Ameriflux")
"""

import time
from glob import glob
from pathlib import Path

import cftime as cf
import numpy as np
import pandas as pd
from tqdm import tqdm

import ilamb3_data as ild

# -------------------------------------------------------------------
# Tower metadata (expand as needed)
# -------------------------------------------------------------------
FLUX_TOWERS = {
    "US-ICh": {"lat": 68.6068, "lon": -149.2958},
    "US-ICt": {"lat": 68.6058, "lon": -149.3110},
    "US-ICs": {"lat": 68.6063, "lon": -149.3041},
}

DEFAULT_MISSING = -9999


def get_variable_table() -> pd.DataFrame:
    """
    Mapping of CMOR-like variable_id -> AmeriFlux column, plus CF standard_name/units.
    """
    return pd.DataFrame(
        [
            {
                "standard_name": "gross_primary_productivity",
                "variable_id": "gpp",
                "amf_col": "GPP_DT_VUT_REF",
                "units": "gC m-2 d-1",
            },
            {
                "standard_name": "precipitation",
                "variable_id": "pr",
                "amf_col": "P_F",
                "units": "mm d-1",
            },
            {
                # AmeriFlux TA_F is typically degC; we convert to K below
                "standard_name": "surface_air_temperature",
                "variable_id": "tas",
                "amf_col": "TA_F",
                "units": "K",
            },
        ]
    )


def infer_site_from_filename(csv_path: Path) -> str | None:
    # Typical: AMF_US-ICt_FLUXNET_FULLSET_DD_2007-2023_4-6.csv -> US-ICt
    name = csv_path.name
    for token in name.replace(".", "_").split("_"):
        if token.startswith("US-"):
            return token
    return None


def parse_time_col(df: pd.DataFrame, csv_name: str) -> pd.Series:
    """
    AmeriFlux daily FULLSET commonly uses TIMESTAMP or TIMESTAMP_START with YYYYMMDD.
    Returns pandas datetime64 series.
    """
    if "TIMESTAMP" in df.columns:
        t = pd.to_datetime(df["TIMESTAMP"], format="%Y%m%d", errors="coerce")
    elif "TIMESTAMP_START" in df.columns:
        t = pd.to_datetime(df["TIMESTAMP_START"], format="%Y%m%d", errors="coerce")
    else:
        raise ValueError(
            f"No recognizable timestamp column in {csv_name}. "
            "Expected TIMESTAMP or TIMESTAMP_START."
        )
    if t.isna().any():
        bad = df.loc[t.isna()].head(5)
        raise ValueError(f"Some timestamps could not be parsed in {csv_name}. Examples:\n{bad}")
    return t


def pandas_time_to_cftime_daily(t: pd.Series) -> list:
    """
    Convert pandas timestamps to cftime.DatetimeGregorian at daily resolution.
    """
    out = []
    for ts in pd.DatetimeIndex(t):
        out.append(cf.DatetimeGregorian(int(ts.year), int(ts.month), int(ts.day)))
    return out


def maybe_convert_units(varname: str, values: np.ndarray) -> np.ndarray:
    """
    Apply unit conversions for specific variables if needed.
    Currently:
      - tas: Celsius -> Kelvin
    """
    if varname == "tas":
        return values + 273.15
    return values


def set_required_global_attrs_offline(
    ds,
    *,
    source_id: str,
    source_label: str,
    variable_id: str,
    contact: str,
    frequency: str,
    realm: str,
    region: str,
    download_stamp: str,
    generate_stamp: str,
    tracking_id: str,
    source_version_number: str,
    processing_code_location: str,
):
    """
    Provide enough global attributes for the ILAMB3 validator to accept the dataset
    without needing ild.set_ods26_global_attrs().
    """
    ds.attrs.update(
        dict(
            # Required by validator (at least these)
            Conventions="CF-1.8",
            source_version_number=source_version_number,
            # Identity
            title=f"{source_label} {variable_id}",
            source_id=source_id,
            source_label=source_label,
            variable_id=variable_id,
            # Provenance
            contact=contact,
            institution="AmeriFlux / FLUXNET Community",
            institution_id="AmeriFlux",
            source="Eddy covariance flux tower measurements",
            source_type="insitu",
            source_data_url="https://ameriflux.lbl.gov/",
            source_data_retrieval_date=download_stamp,
            creation_date=generate_stamp,
            processing_code_location=processing_code_location,
            history=f"{download_stamp}: source CSV file(s) prepared locally; {generate_stamp}: converted to ILAMB3 format",
            # Classification
            realm=realm,
            region=region,
            frequency=frequency,
            nominal_resolution="site",
            grid="site",
            grid_label="site",
            product="site-observations",
            # Versioning
            tracking_id=tracking_id,
            variant_label="ILAMB",
            variant_info="CMORized product prepared by ILAMB (offline attrs; CV not fetched)",
            version=f"v{generate_stamp.replace('-', '')}",
            # Optional but helpful
            license="N/A (check AmeriFlux/FLUXNET terms for the specific product you downloaded).",
            references="N/A",
            doi="N/A",
        )
    )
    return ds


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(
        description="Convert AmeriFlux daily FULLSET CSV(s) to ILAMB3 netcdf (offline-safe)."
    )
    p.add_argument("--csv", default=None, help="Path to a single CSV file")
    p.add_argument("--csv-dir", default=None, help="Directory containing CSV files")
    p.add_argument("--pattern", default="*_DD*.csv", help="Glob pattern under --csv-dir (default: *_DD*.csv)")
    p.add_argument(
        "--var",
        default=None,
        help="Which variable_id to export (gpp, pr, tas). If omitted, export all available.",
    )
    p.add_argument("--amf-col", default=None, help="Override AmeriFlux column name (only used if --var is set)")
    p.add_argument("--units", default=None, help="Override units (only used if --var is set)")
    p.add_argument("--site", default=None, help="Override site id (otherwise inferred from filename)")

    p.add_argument("--source-id", default="Ameriflux", help="source_id for global attrs")
    p.add_argument("--source-label", default="AmeriFlux", help="Short label for global attrs")
    p.add_argument("--source-version-number", default="N/A", help="Required global attr: source_version_number")
    p.add_argument("--contact", default="N/A", help="Contact info for global attrs")

    p.add_argument("--bounds-frequency", default="D", help="Time bounds frequency for ild.set_time_attrs (daily: 'D')")
    p.add_argument("--frequency", default="day", help="Global attr frequency (daily: day)")
    p.add_argument("--realm", default="land", help="Global attr realm")
    p.add_argument("--region", default="global_land", help="Global attr region")
    p.add_argument("--missing", type=float, default=DEFAULT_MISSING, help="Missing value flag in CSV (default -9999)")

    p.add_argument(
        "--outdir",
        default=None,
        help="Output directory. Default is <repo>/data/Ameriflux/_output_offline (anchored to script path).",
    )
    args = p.parse_args()

    # Anchor paths to script location to avoid duplicate directories
    script_dir = Path(__file__).resolve().parent        # .../ilamb3-data/data/Ameriflux
    repo_root = script_dir.parent.parent                # .../ilamb3-data
    default_outdir = script_dir / "_output_offline"     # .../ilamb3-data/data/Ameriflux/_output_offline
    outdir = Path(args.outdir).expanduser().resolve() if args.outdir else default_outdir
    outdir.mkdir(parents=True, exist_ok=True)

    # Resolve input CSV list
    if args.csv:
        csvs = [str(Path(args.csv).expanduser().resolve())]
    elif args.csv_dir:
        d = Path(args.csv_dir).expanduser().resolve()
        csvs = sorted(glob(str(d / args.pattern)))
    else:
        raise ValueError("Provide either --csv <file> or --csv-dir <dir>")

    if len(csvs) == 0:
        raise ValueError("No CSV files found with the provided inputs.")

    dfv = get_variable_table()

    # Filter to one variable (and apply overrides)
    if args.var is not None:
        dfv = dfv[dfv["variable_id"] == args.var].copy()
        if dfv.empty:
            raise ValueError(
                f"--var {args.var} not recognized. Known: {get_variable_table()['variable_id'].tolist()}"
            )
        if args.amf_col is not None:
            dfv.loc[:, "amf_col"] = args.amf_col
        if args.units is not None:
            dfv.loc[:, "units"] = args.units

    # Read CSV(s) -> dataframe indexed by (time, site)
    frames = []
    for csv in tqdm(csvs, desc="Reading CSV(s)"):
        csv_path = Path(csv)
        site = args.site or infer_site_from_filename(csv_path)
        if site is None:
            raise ValueError(f"Could not infer site from filename: {csv_path.name}. Provide --site.")

        if site not in FLUX_TOWERS:
            raise ValueError(f"Site {site} not in FLUX_TOWERS. Add it to the dict first.")

        df = pd.read_csv(csv_path, na_values=[args.missing, int(args.missing)])
        t = parse_time_col(df, csv_path.name)

        keep_cols = [row["amf_col"] for _, row in dfv.iterrows() if row["amf_col"] in df.columns]
        if len(keep_cols) == 0:
            raise ValueError(
                f"None of requested columns found in {csv_path.name}. Requested: {dfv['amf_col'].tolist()}"
            )

        dfs = df.loc[:, keep_cols].copy()
        dfs["time"] = t
        dfs["site"] = site
        frames.append(dfs.set_index(["time", "site"]).sort_index())

    df_all = pd.concat(frames).sort_index()

    # DataFrame -> xarray Dataset
    ds = df_all.to_xarray()

    # Rename AMF columns -> variable_ids
    rename_map = {}
    for _, row in dfv.iterrows():
        if row["amf_col"] in ds.data_vars:
            rename_map[row["amf_col"]] = row["variable_id"]
    ds = ds.rename(rename_map)

    # Unit conversion + variable attrs
    for _, row in dfv.iterrows():
        vid = row["variable_id"]
        if vid in ds:
            # Apply unit conversion if needed (e.g., tas: degC -> K)
            vals = ds[vid].values
            vals = maybe_convert_units(vid, vals)
            ds[vid].values[:] = vals

            ds[vid].attrs = {"standard_name": row["standard_name"], "units": row["units"]}

    # Convert time coord to cftime daily objects
    ds = ds.sortby("time")
    ds["time"] = pandas_time_to_cftime_daily(pd.Series(pd.DatetimeIndex(ds["time"].values)))

    # Assign site coordinates lat/lon
    sites = [str(s) for s in ds["site"].values]
    ds["site"].attrs["name"] = "AmeriFlux site id"
    ds = ds.assign_coords(
        lat=("site", [FLUX_TOWERS[s]["lat"] for s in sites]),
        lon=("site", [FLUX_TOWERS[s]["lon"] for s in sites]),
    )

    # Standardize coordinate attrs and create time bounds
    ds = ild.set_time_attrs(ds, bounds_frequency=args.bounds_frequency)
    ds = ild.set_lat_attrs(ds)
    ds = ild.set_lon_attrs(ds)

    # Stamps
    generate_stamp = time.strftime("%Y-%m-%d")
    tracking_id = ild.gen_trackingid()
    newest_mtime = max(Path(f).stat().st_mtime for f in csvs)
    download_stamp = time.strftime("%Y-%m-%d", time.localtime(newest_mtime))

    # For traceability in attrs
    processing_code_location = f"{repo_root}/data/Ameriflux/convert.py"

    # Write one file per variable
    for varname in tqdm(list(ds.data_vars), desc="Writing netcdf files"):
        if varname.endswith("_bnds") or "_bnds" in varname:
            continue
        if args.var is not None and varname != args.var:
            continue

        to_drop = [v for v in ds.data_vars if (v != varname) and (not v.endswith("_bnds"))]
        out_ds = ds.drop_vars(to_drop)

        out_ds = set_required_global_attrs_offline(
            out_ds,
            source_id=args.source_id,
            source_label=args.source_label,
            variable_id=varname,
            contact=args.contact,
            frequency=args.frequency,
            realm=args.realm,
            region=args.region,
            download_stamp=download_stamp,
            generate_stamp=generate_stamp,
            tracking_id=tracking_id,
            source_version_number=args.source_version_number,
            processing_code_location=processing_code_location,
        )

        # Offline deterministic filename (no ODS26 CV lookup)
        site_tag = "-".join(sites)
        out_path = outdir / f"{args.source_id}_{varname}_{site_tag}_{args.frequency}_v{generate_stamp.replace('-','')}.nc"
        out_ds.to_netcdf(out_path)
        print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

