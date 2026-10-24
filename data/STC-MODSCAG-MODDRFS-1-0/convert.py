"""
convert.py — STC-MODSCAG-MODDRFS-1-0 obs4MIPs daily snow, Western US.

Pipeline
--------
1. Download the original STC-MODSCAG/MODDRFS MODIS-Sinusoidal 500 m daily tiles
   (5 WUS tiles) from the public anonymous FTP source into _raw/ next to this script.
2. Areal bin-and-mean regrid to a 0.125 deg lat/lon WUS grid. This reproduces the
   reference MATLAB regrid.
3. Format to obs4MIPs (CF-1.12 ODS-2.6) with the ilamb3_data (`ild`) helpers and
   write one file per variable.

Orientation
-----------
Latitude is written ascending (South to North) with the data in the same order,
so the grid is correct by construction.

Variables (source name, output name)
------------------------------------
  snow_fraction                    snc        (CMIP6:  %, surface_snow_area_fraction)
  albedo_dirty_terrain_corrected   snowalb    (custom: %, surface_snow_albedo, snow-masked)
  grain_size                       snowgrain  (custom: micrometer, snow_grain_size, snow-masked)
snc is averaged over all valid pixels; albedo/grain over snow pixels (snow_fraction > 0),
matching the reference processing.

Time window
-----------
The full record on the server: 1 Oct 2000 to 30 Sep 2023 (water years 2001-2023).
Set CONVERT_START / CONVERT_END (YYYY-MM-DD) to process a shorter window, e.g. a test
over one water year: CONVERT_START=2004-10-01 CONVERT_END=2005-09-30.

Raw data
--------
Downloaded from the public anonymous FTP source (see SRC); also archived at NSIDC
(doi:10.7265/f6j3-f387).
"""

import os
import re
from datetime import date, timedelta
from ftplib import FTP, error_perm
from pathlib import Path

import cftime as cf
import netCDF4 as nc
import numpy as np
import xarray as xr

import ilamb3_data as ild

START = date(2000, 10, 1)  # first day on the server
END = date(2023, 9, 30)  # last day on the server
if os.environ.get("CONVERT_START"):
    START = date.fromisoformat(os.environ["CONVERT_START"])
if os.environ.get("CONVERT_END"):
    END = date.fromisoformat(os.environ["CONVERT_END"])

TILES = ["h08v04", "h08v05", "h09v04", "h09v05", "h10v04"]
RAW = Path(__file__).resolve().parent / "_raw"  # RAW/<tile>/<year>/*.nc

R = 6371007.181  # MODIS sinusoidal sphere radius, used for the tile projection


def target_grid(res=0.125, lat=(32.0625, 49.9375), lon=(-124.9375, -104.0625)):
    """Build the regular lat/lon target grid.

    Returns (lat_centres, lon_centres, lat_edges, lon_edges); latitude ascending
    South to North, longitude in -180..180. Defaults to the 0.125 deg WUS grid
    (144 x 168 cells).
    """
    lat_c = np.arange(lat[0], lat[1] + res / 2, res)
    lon_c = np.arange(lon[0], lon[1] + res / 2, res)
    lat_e = np.arange(lat_c[0] - res / 2, lat_c[-1] + res, res)
    lon_e = np.arange(lon_c[0] - res / 2, lon_c[-1] + res, res)
    return lat_c, lon_c, lat_e, lon_e


LAT_C, LON_C, LAT_E, LON_E = target_grid()

# Public source: anonymous FTP at NSIDC (sidads), which holds the 5 WUS tiles.
SRC = {
    "label": "STC",
    "ftp": {
        "host": "sidads.colorado.edu",
        "base": "/pub/DATASETS/STC_MODSCGDRF_HIST_v1",
    },
    "attrs": {
        "contact": "Karl Rittger (karl.rittger@colorado.edu)",
        "doi": "https://doi.org/10.7265/f6j3-f387",
        "institution": "Institute of Arctic and Alpine Research, University of Colorado Boulder, CO, USA",
        "institution_id": "INSTAAR",
        "license": "Creative Commons Attribution 4.0 International License (CC BY 4.0)",
        "references": (
            "Rittger, K., Lenard, S. J. P., Palomaki, R. T., Brodzik, M. J., Stillinger, T., "
            "Bair, E. H., Dozier, J., & Painter, T. H. (2025). Historical MODIS/Terra L3 Global "
            "Daily 500m SIN Grid Snow Cover, Snow Albedo, and Snow Physical Properties, Version 1. "
            "Boulder, Colorado USA. NSIDC. doi:10.7265/f6j3-f387. Rittger, K., Raleigh, M. S., "
            "Dozier, J., et al. (2020), Water Resources Research 56, doi:10.1029/2019WR024914."
        ),
        "source": "MOD09GA Terra Collection 6 (v006; v061 from 28 Dec 2021), STC-MODSCAG/MODDRFS",
        "source_id": "STC-MODSCAG-MODDRFS-1-0",
        "source_label": "STC-MODSCAG-MODDRFS",
        "source_type": "satellite_retrieval",
        "source_version_number": "1.0",
        "title_stem": "STC-MODSCAG/MODDRFS Western US daily snow",
    },
}

# per-variable output settings, keyed by source variable name
VARS = {
    "snow_fraction": {"out": "snc", "nodata": 255, "snow_only": False, "cmip6": "snc"},
    "albedo_dirty_terrain_corrected": {
        "out": "snowalb",
        "nodata": 255,
        "snow_only": True,
        "units": "%",
        "standard_name": "surface_snow_albedo",
        "long_name": "Snow albedo",
    },
    "grain_size": {
        "out": "snowgrain",
        "nodata": 65535,
        "snow_only": True,
        "units": "micrometer",
        "standard_name": "snow_grain_size",
        "long_name": "Snow grain size (effective radius)",
    },
}


def download_tiles(ftp_cfg, start, end, tiles, out):
    """Download the WUS tiles for days start..end into out/<tile>/<year>/ over anonymous FTP.

    The source is the dataset's public anonymous FTP share (no credentials required).
    Files that already exist locally (non-empty) are skipped, so the download is resumable.
    """
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    ftp = FTP(ftp_cfg["host"], timeout=120)
    ftp.login("anonymous", "")
    n_total = 0
    try:
        for t in tiles:
            for year in range(start.year, end.year + 1):
                try:
                    ftp.cwd(f"{ftp_cfg['base']}/{t}/{year}")
                except error_perm:
                    print(f"  {t}/{year}: not on server, skipping")
                    continue
                names = []
                ftp.retrlines("NLST", names.append)
                names = [Path(n).name for n in names if n.endswith(".nc")]
                names = [n for n in names if start <= _file_date(n) <= end]
                dest = out / t / str(year)
                dest.mkdir(parents=True, exist_ok=True)
                got = 0
                for name in names:
                    lp = dest / name
                    if lp.is_file() and lp.stat().st_size > 0:
                        continue
                    with open(lp, "wb") as fh:
                        ftp.retrbinary(f"RETR {name}", fh.write)
                    got += 1
                    n_total += 1
                print(f"  {t}/{year}: {got} downloaded ({len(names)} on server)")
    finally:
        ftp.quit()
    print(f"download complete: {n_total} new files in {out}")


def _file_date(name):
    """Date encoded as YYYYMMDD in a tile filename."""
    stamp = re.search(r"_(\d{8})_", name).group(1)
    return date(int(stamp[:4]), int(stamp[4:6]), int(stamp[6:]))


def _tile_latlon(x, y):
    """MODIS sinusoidal (sphere R) inverse giving per-pixel lat/lon (deg)."""
    phi = y / R
    lat = np.degrees(phi)
    lon = np.degrees(x[None, :] / (R * np.cos(phi)[:, None]))
    return np.repeat(lat[:, None], x.size, axis=1), lon


def _binmean(lat, lon, vals):
    """Areal mean of point values into the 0.125 deg grid; row 0 = South."""
    m = np.isfinite(vals)
    s, _, _ = np.histogram2d(lat[m], lon[m], bins=[LAT_E, LON_E], weights=vals[m])
    c, _, _ = np.histogram2d(lat[m], lon[m], bins=[LAT_E, LON_E])
    out = np.full_like(s, np.nan)
    nz = c > 0
    out[nz] = s[nz] / c[nz]
    return out


def regrid_day(day):
    """Return {out_var: (144,168) grid} for one day (South-up), or NaN grids if none."""
    cols = {"lat": [], "lon": [], **{sv: [] for sv in VARS}}
    for t in TILES:
        # glob so the lookup is robust to small variations in tile filenames
        matches = list((RAW / t / str(day.year)).glob(f"*{t}*{day:%Y%m%d}*.nc"))
        if not matches:
            continue
        ds = nc.Dataset(matches[0])
        x = ds.variables["x"][:].astype("f8")
        y = ds.variables["y"][:].astype("f8")
        lat2d, lon2d = _tile_latlon(x, y)
        raw = {}
        for sv, spec in VARS.items():
            v = ds.variables[sv]
            v.set_auto_maskandscale(False)  # avoid wrong valid_range masking
            a = np.squeeze(v[:]).astype("f4")
            a[a == spec["nodata"]] = np.nan
            raw[sv] = a
        ds.close()
        valid = np.isfinite(raw["snow_fraction"])
        snow = valid & (raw["snow_fraction"] > 0)
        cols["lat"].append(lat2d[valid])
        cols["lon"].append(lon2d[valid])
        for sv, spec in VARS.items():
            a = np.where(snow, raw[sv], np.nan) if spec["snow_only"] else raw[sv]
            cols[sv].append(a[valid])
    if not cols["lat"]:
        print(f"  {day}: no tiles found, writing missing values")
        return {
            spec["out"]: np.full((LAT_C.size, LON_C.size), np.nan)
            for spec in VARS.values()
        }
    lat = np.concatenate(cols["lat"])
    lon = np.concatenate(cols["lon"])
    return {
        spec["out"]: _binmean(lat, lon, np.concatenate(cols[sv]))
        for sv, spec in VARS.items()
    }


def build_dataset():
    """Regrid START..END into an xr.Dataset (time, lat, lon), lat ascending."""
    cubes = {spec["out"]: [] for spec in VARS.values()}
    times = []
    day = START
    while day <= END:
        for out, g in regrid_day(day).items():
            cubes[out].append(g.astype("f4"))
        times.append(cf.DatetimeGregorian(day.year, day.month, day.day))
        if day.month == 12 and day.day == 31 or day == END:
            print(f"  {SRC['label']} {day.year} regridded ({len(times)} days total)")
        day += timedelta(days=1)
    data = {out: (("time", "lat", "lon"), np.stack(c)) for out, c in cubes.items()}
    return xr.Dataset(data, coords={"time": times, "lat": LAT_C, "lon": LON_C})


# Download
ga = SRC["attrs"]
source_url = f"ftp://{SRC['ftp']['host']}{SRC['ftp']['base']}"
download_tiles(SRC["ftp"], START, END, TILES, RAW)
download_stamp = ild.output.utc_timestamp()

# Regrid, then standardize (CF) the time, lat, and lon coordinates and add bounds
ds = build_dataset()
ds = ild.time.standardize(
    ds, bounds_frequency="D", ref_date=cf.DatetimeGregorian(START.year, 1, 1)
)
ds = ild.lat.standardize(ds)
ds = ild.lon.standardize(ds)
ds = ild.bounds.add_rectilinear_bounds(ds, overwrite=True)

# Set compression levels for variables
compression = {
    "zlib": True,
    "complevel": 4,
    "shuffle": True,
    "chunksizes": (1, LAT_C.size, LON_C.size),
}

# Write a NetCDF for each variable
for spec in VARS.values():
    var = spec["out"]
    out = ds.drop_vars([s["out"] for s in VARS.values() if s["out"] != var])

    # Use the CMIP6 table where the variable exists, otherwise our own names
    if "cmip6" in spec:
        info = ild.variable.lookup_cmip6(spec["cmip6"], spec["cmip6"])
        units, standard_name, long_name = (
            info["variable_units"],
            info["cf_standard_name"],
            info["variable_long_name"],
        )
    else:
        units, standard_name, long_name = (
            spec["units"],
            spec["standard_name"],
            spec["long_name"],
        )

    # Format the variable attributes
    out = ild.variable.standardize(
        out,
        var,
        units=units,
        standard_name=standard_name,
        long_name=long_name,
        target_dtype="float32",
        convert=False,
        compression=compression,
    )

    # Set the global attrs
    out = ild.global_attrs.set_ods26(
        out,
        activity_id="ILAMB",
        contact=ga["contact"],
        creation_date=ild.output.utc_timestamp(),
        dataset_contributor="Loc Luong",
        doi=ga["doi"],
        frequency="day",
        grid=(
            "0.125x0.125 degree latitude x longitude; regridded from MODIS Sinusoidal "
            "500 m tiles (h08v04, h08v05, h09v04, h09v05, h10v04) by areal bin-and-mean"
        ),
        grid_label="gn",
        history=(
            f"Downloaded from {source_url} on {download_stamp}. Regridded from MODIS "
            "Sinusoidal 500 m tiles to 0.125 degree by areal bin-and-mean"
            + (" over snow pixels only" if spec["snow_only"] else "")
            + ". Converted to CF-compliant and ODS-aligned product by ILAMB on "
            f"{ild.output.utc_timestamp()}."
        ),
        institution=ga["institution"],
        institution_id=ga["institution_id"],
        license=ga["license"],
        nominal_resolution="10 km",
        processing_code_location=f"https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/{ga['source_id']}/convert.py",
        product="observations",
        realm="land",
        references=ga["references"],
        region="western_united_states",
        source=ga["source"],
        source_data_retrieval_date=download_stamp,
        source_data_url=source_url,
        source_id=ga["source_id"],
        source_label=ga["source_label"],
        source_type=ga["source_type"],
        source_version_number=ga["source_version_number"],
        title=f"{ga['title_stem']} ({var})",
        tracking_id=ild.output.new_tracking_id(),
        variable_id=var,
        variant_label="ILAMB",
        variant_info="Regridded and CMORized product prepared by ILAMB",
        version=f"v{ild.output.utc_timestamp()[:10].replace('-', '')}",
    )
    out = ild.output.order_dimensions(out)
    out_path = ild.output.filename_from_attrs(out.attrs)
    out.to_netcdf(out_path)
    print("wrote", out_path)
