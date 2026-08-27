"""Convert the CARDAMOM Carbon-Water-Energy Reanalysis (ORNL DAAC ds 2492) into
CF-compliant, per-variable netCDF4 files for ILAMB benchmarking.

See README for detailed information and benchmarking caveats.
"""

from __future__ import annotations

import time
from pathlib import Path

import cftime as cf
import earthaccess
import numpy as np
import xarray as xr

import ilamb3_data as ild

# --------------------------------------------------------------------------------------
# configuration
# --------------------------------------------------------------------------------------
DOI = "10.3334/ORNLDAAC/2492"
SOURCE_FILE = "_raw/CARDAMOM_satellite_constrained_terrestrial_biosphere_reanalysis.nc4"
INSTITUTION = "Jet Propulsion Laboratory, California Institute of Technology"
OUT_SOURCE = "CARDAMOM"  # sub-directory label inside DATA/<var>/

# --------------------------------------------------------------------------------------
# Quality control: fixed physical ceilings on *source* variables
# --------------------------------------------------------------------------------------
QC_BOUNDS = [
    # base, bound, reduce, domain, rationale
    (
        "C_som",
        1.0e5,
        "any",
        "carbon",
        "soil C > 100 kgC m-2; exceeded only by deep peat",
    ),
    ("C_cwd", 2.0e4, "any", "carbon", "coarse woody debris > 20 kgC m-2"),
    (
        "rh_co2",
        20.0,
        "any",
        "carbon",
        "Rh > 20 gC m-2 d-1 is not sustained by any ecosystem",
    ),
    ("H2O_LY3", 1.0e4, "any", "water", "soil layer 3 > 10 m water column"),
    (
        "H2O_SWE",
        3.0e3,
        "mean",
        "water",
        "mean SWE > 3 m implies a glacier, which DALEC-CWE lacks",
    ),
    ("runoff", 30.0, "any", "water", "runoff > 30 kg m-2 d-1"),
    (
        "D_TEMP_LY1",
        323.15,
        "any",
        "temp",
        "soil T > 50 C exceeds the hottest skin temperature in the forcing (41.8 C)",
    ),
]

# Source quantities that may legitimately be negative (net exchanges)
SIGNED_BASES = {"NBE", "NEP"}

# --------------------------------------------------------------------------------------
# The CF `comment` attribute. Caveats a user needs before benchmarking against
# this dataset; see README.md for the full discussion.
# --------------------------------------------------------------------------------------
COMMENT_BASE = (
    "Grid cells whose state is non-physical have been masked; see the history "
    "attribute for the exact bounds. These arise where CARDAMOM's per-pixel "
    "initial conditions are poorly constrained (producing a decaying "
    "disequilibrium transient over 2001-2003) or over ice caps, which DALEC-CWE "
    "does not represent. The QC bounds are evaluated on the median member only; "
    "the 25th/75th ancillary variables are the raw ensemble interquartile range "
    "and may exceed them at retained cells. IMPORTANT: CARDAMOM assimilates "
    "GRACE/GRACE-FO "
    "water storage, mean runoff (GRUN), satellite GPP and LAI, above- and "
    "below-ground biomass, harmonised soil organic carbon, MODIS snow-covered "
    "fraction, MOPITT-CO-derived fire emissions and CMS-Flux NBE, and is forced "
    "by ERA5 meteorology, atmospheric CO2 and prescribed burned area. "
    "Comparisons against models are therefore NOT independent of the "
    "corresponding ILAMB reference datasets; see README.md for the mapping and "
    "for which variables remain informative."
)
COMMENT = {
    "mrso": COMMENT_BASE + " Additionally, DALEC-CWE's three soil water layers "
    "are per-pixel calibrated stores whose thicknesses are not published, so "
    "although summing them is the correct construction for CMIP mrso, absolute "
    "magnitudes are not comparable to a model with a fixed soil column depth "
    "(the upper tail exceeds any CMIP column capacity). Use tws, which ILAMB's "
    "ConfTWSA de-means before scoring, for a like-for-like comparison.",
    "tws": COMMENT_BASE + " ILAMB's ConfTWSA subtracts the temporal mean from "
    "both reference and model, so the unpublished layer depths do not affect "
    "this comparison. Note that GRACE/GRACE-FO water storage anomalies were "
    "assimilated by CARDAMOM, so this is not an independent check against GRACE.",
    "tsl": COMMENT_BASE + " The soil-temperature ceiling is set from the model's "
    "own ERA5 forcing, whose hottest monthly-mean skin temperature anywhere is "
    "41.8 C; a subsurface layer cannot exceed its own surface forcing as a "
    "monthly mean. Additionally, this is the temperature of DALEC-CWE's first "
    "energy state, whose depth is not published. No depth coordinate is "
    "provided and depth-sensitive analyses (e.g. ILAMB's ConfPermafrost, which "
    "uses dmax=3.5 m) are not supported by this dataset.",
    "nee": COMMENT_BASE + " Additionally, nee is defined here as -NEP, taken "
    "directly from the source. This is not identical to reco - gpp as published "
    "in this collection (they differ by up to 1.7 gC m-2 d-1, median 0.006), and "
    "ILAMB derives the model-side nee as ra + rh - gpp, so a small definitional "
    "offset between reference and model is expected.",
}

# --------------------------------------------------------------------------------------
# ILAMB output variable definitions.
#   terms : list of (source_base_name, coefficient) summed together
#   domain: which QC mask applies -- "carbon", "water", or None for neither
VARDEFS = {
    # --- carbon fluxes (g m-2 d-1 -> kg m-2 s-1) --------------------------------------
    "gpp": {
        "terms": [("gpp", 1.0)],
        "domain": "carbon",
    },
    "ra": {
        "terms": [("resp_auto", 1.0)],
        "domain": "carbon",
    },
    "rh": {
        "terms": [("rh_co2", 1.0)],
        "domain": "carbon",
    },
    # reco is not a cmip6 variable so shouldn't have a standard_name
    # long_name is made up but mirrors rh and ra
    "reco": {
        "terms": [("resp_auto", 1.0), ("rh_co2", 1.0)],
        "domain": "carbon",
        "units": "kg m-2 s-1",
        "standard_name": "",
        "long_name": "Carbon mass flux per unit area into atmosphere due to total ecosystem respiration on land (ra + rh)",
    },
    "npp": {
        "terms": [("gpp", 1.0), ("resp_auto", -1.0)],
        "domain": "carbon",
    },
    "nbp": {
        "terms": [("NBE", -1.0)],
        "domain": "carbon",
    },
    # nee is not a cmip6 variable so shouldn't have a standard_name
    # long_name is made up but mirrors the definition of nee as ra + rh - gpp
    "nee": {
        "terms": [("NEP", -1.0)],
        "domain": "carbon",
        "units": "kg m-2 s-1",
        "standard_name": "",
        "long_name": "Carbon mass flux per unit area into atmosphere due to net ecosystem exchange on land (ra + rh - gpp)",
    },
    "fFire": {
        "terms": [("f_total", 1.0)],
        "domain": "carbon",
    },
    # --- carbon pools (g m-2 -> kg m-2) -----------------------------------------------
    "cVeg": {
        "terms": [("C_fol", 1.0), ("C_lab", 1.0), ("C_roo", 1.0), ("C_woo", 1.0)],
        "domain": "carbon",
    },
    "cSoil": {
        "terms": [("C_som", 1.0)],
        "domain": "carbon",
    },
    "cLitter": {
        "terms": [("C_lit", 1.0), ("C_cwd", 1.0)],
        "domain": "carbon",
    },
    # --- water fluxes (kg m-2 d-1 -> kg m-2 s-1) --------------------------------------
    "et": {
        "terms": [("ets", 1.0)],
        "domain": "water",
    },
    "tran": {
        "terms": [("transp", 1.0)],
        "domain": "water",
    },
    "mrro": {
        "terms": [("runoff", 1.0)],
        "domain": "water",
    },
    "mrros": {
        "terms": [("q_surf", 1.0)],
        "domain": "water",
    },
    # --- water / snow pools (kg m-2) --------------------------------------------------
    # see README about mrso caveats
    "mrso": {
        "terms": [("H2O_LY1", 1.0), ("H2O_LY2", 1.0), ("H2O_LY3", 1.0)],
        "domain": "water",
    },
    # see README about tws caveats
    "tws": {
        "terms": [
            ("H2O_LY1", 1.0),
            ("H2O_LY2", 1.0),
            ("H2O_LY3", 1.0),
            ("H2O_SWE", 1.0),
        ],
        "domain": "water",
    },
    "snw": {
        "terms": [("H2O_SWE", 1.0)],
        "domain": "water",
    },
    # --- diagnostics ------------------------------------------------------------------
    "lai": {
        "terms": [("D_LAI", 1.0)],
        "domain": None,
    },
    # See README about tsl caveats
    "tsl": {
        "terms": [("D_TEMP_LY1", 1.0)],
        "domain": "temp",
    },
}

# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def ensemble_member(ds: xr.Dataset, base: str, statistic: str) -> xr.DataArray:
    """Read one of the source's fixed ensemble-statistic variables."""
    da = ds[f"{base}_{statistic}"]
    if base not in SIGNED_BASES:
        da = da.clip(min=0)
    return da


def combine_terms(
    ds: xr.Dataset, terms: list[tuple[str, float]], statistic: str
) -> xr.DataArray:
    """Combine like-unit source terms and retain their units for conversion."""
    sources = [ensemble_member(ds, base, statistic) for base, _ in terms]
    units = sources[0].attrs["units"]
    arrays = [coefficient * source for source, (_, coefficient) in zip(sources, terms)]
    combined = sum(arrays)
    combined.attrs = {"units": units}
    return combined


def add_variable_metadata() -> None:
    """Fill CMIP metadata not supplied explicitly in ``VARDEFS``."""
    for name, spec in VARDEFS.items():
        if "units" in spec:
            continue
        info = ild.variable.lookup_cmip6(name, variable_id=name)
        spec.update(
            units=info["variable_units"],
            long_name=info["variable_long_name"],
            standard_name=info["cf_standard_name"],
        )


def build_qc_masks(ds: xr.Dataset) -> dict[str, xr.DataArray]:
    """Return ``{domain: 2-D bool mask}`` of cells breaching a QC_BOUNDS ceiling.

    True marks a cell to be removed. Masks are per-domain so that the water
    bounds do not remove otherwise sound carbon cells, and vice versa; see the
    module docstring.
    """
    template = xr.zeros_like(ds["C_som_median"].isel(time=0), dtype=bool)
    masks = {domain: template.copy() for *_, domain, _ in QC_BOUNDS}
    detail = {}  # (lat, lon) -> list of reasons, for the run log

    for base, bound, how, domain, why in QC_BOUNDS:
        da = ensemble_member(ds, base, "median")
        stat = da.mean("time") if how == "mean" else da.max("time")
        bad = (stat > bound).fillna(False)
        masks[domain] |= bad
        for i, j in zip(*np.where(bad.values)):
            key = (float(ds["lat"].values[i]), float(ds["lon"].values[j]))
            detail.setdefault(key, []).append(f"{base} {how}>{bound:g}")

    print("  QC: masking non-physical cells (fixed physical bounds)")
    for (la, lo), reasons in sorted(detail.items()):
        print(f"      lat={la:6.1f} lon={lo:7.1f}  {', '.join(reasons)}")
    nland = int(ensemble_member(ds, "C_som", "median").isel(time=0).count())
    for dom, m in masks.items():
        print(
            f"      {dom:6s} mask: {int(m.sum()):3d} cells removed of {nland} land cells"
        )
    return masks


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------
def main(source: str = SOURCE_FILE):

    # Download the data if it doesn't exist locally
    source_path = Path(source)
    if not source_path.is_file():
        earthaccess.login()
        granules = earthaccess.search_data(doi=DOI, granule_name="*.nc4")
        source_path = Path(earthaccess.download(granules, "_raw")[0])

    # Generate timestamps for download and creation times
    download_stamp = ild.output.utc_timestamp(source_path.stat().st_mtime)
    creation_stamp = ild.output.utc_timestamp()

    # Open the dataset with CFDatetimeCoder to handle time decoding correctly
    time_coder = xr.coders.CFDatetimeCoder(use_cftime=True)
    ds = xr.open_dataset(source_path, decode_times=time_coder)

    # Add CMIP6 variable metadata to VARDEFS
    add_variable_metadata()

    masks = build_qc_masks(ds)
    qc_rule = "; ".join(
        f"{base} {how}>{bound:g} ({why})" for base, bound, how, _, why in QC_BOUNDS
    )

    # --- build output datasets for each variable --------------------------------------

    for varname, spec in VARDEFS.items():
        arr = combine_terms(ds, spec["terms"], "median")

        def finish(da: xr.DataArray) -> xr.DataArray:
            da = da.astype("float32").transpose("time", "lat", "lon")
            if spec["domain"]:
                da = da.where(~masks[spec["domain"]])
            return da

        out = finish(arr).to_dataset(name=varname)

        # --- percentile ancillary variables -------------------------------------------

        # Create percentile variables if applicable
        has_percentiles = len(spec["terms"]) == 1
        if has_percentiles:
            src_base, coefficient = spec["terms"][0]
            # Flip 25th and 75th percentiles based on the coefficient sign
            source_statistics = (
                ("25th", "75th") if coefficient > 0 else ("75th", "25th")
            )
            # Create both percentile variables before declaring them ancillary.
            for percentile, statistic in zip((25, 75), source_statistics):
                ancillary = f"{varname}_p{percentile}"
                out[ancillary] = finish(
                    combine_terms(ds, [(src_base, coefficient)], statistic)
                )

        # --- main variable attributes -------------------------------------------------

        # Set the main variable attributes
        out = ild.variable.standardize(
            out,
            varname,
            units=spec["units"],
            standard_name=spec["standard_name"],
            long_name=spec["long_name"],
            ancillary_variables=(
                f"{varname}_p25 {varname}_p75" if has_percentiles else None
            ),
            target_dtype="float32",
            convert=True,
            compression={"zlib": True},
        )
        # Not all vars have standard_name, so remove it if not defined
        if not spec["standard_name"]:
            out[varname].attrs.pop("standard_name")

        # --- ancillary variable attributes --------------------------------------------

        # Set ancillary attributes after the main variable attrs are set
        if has_percentiles:
            for percentile in (25, 75):
                ancillary = f"{varname}_p{percentile}"
                standard_name = (
                    f"{spec['standard_name']} {percentile}th_percentile"
                    if spec["standard_name"]
                    else ""
                )
                out = ild.variable.standardize(
                    out,
                    ancillary,
                    units=spec["units"],
                    standard_name=standard_name,
                    long_name=(
                        f"{spec['long_name']} CARDAMOM posterior ensemble "
                        f"{percentile}th Percentile"
                    ),
                    target_dtype="float32",
                    convert=True,
                    compression={"zlib": True},
                )
                if not standard_name:
                    out[ancillary].attrs.pop("standard_name")

        # Use built-in ILAMB funcs to build dimensions/attributes
        out = ild.time.standardize(
            out,
            bounds_frequency="M",
            ref_date=cf.DatetimeProlepticGregorian(1850, 1, 1),
        )
        out = ild.lat.standardize(out)
        out = ild.lon.standardize(out)
        out = ild.bounds.add_rectilinear_bounds(out)
        out = ild.output.order_dimensions(out)

        # --- set NetCDF global attributes ---------------------------------------------

        out = ild.global_attrs.set_ods26(
            out,
            activity_id="ILAMB",
            aux_uncertainty_id=f"{varname}_p25 {varname}_p75",
            comment=COMMENT.get(varname, COMMENT_BASE),
            contact="Renato Braghiere (renatob@caltech.edu)",
            creation_date=creation_stamp,
            dataset_contributor="Renato Braghiere",
            doi=DOI,
            frequency="mon",
            grid="4x5 degree latitude x longitude",
            grid_label="gn",
            has_aux_unc="TRUE",
            history=(
                f"{download_stamp}: downloaded {SOURCE_FILE} from earthdata (ORNL DAAC); "
                "only extracted the median ensemble member (25th/75th percentiles retained as CF bounds where the "
                f"variable derives from a single source term); masked grid cells breaching "
                f"fixed physical bounds [{qc_rule}], applied per domain "
                f"(carbon bounds to carbon variables, water bounds to water variables); "
                f"converted to CF-compliant netCDF for ILAMB"
            ),
            institution="NASA Jet Propulsion Laboratory, Pasadena, CA 91109, USA",
            institution_id="NASA-JPL",
            license="Creative Commons Attribution 4.0 International (CC BY 4.0)",
            nominal_resolution="4x5-degree",
            processing_code_location="https://github.com/rubisco-sfa/ilamb3-data/blob/main/data/CARDAMOM-1100-1/convert.py",
            product="reanalysis",
            realm="land",
            references=(
                "Bilir, T.E., Bloom, A.A., Parazoo, N.C., Liu, J., & Braghiere, R.K. "
                "(2026). CARDAMOM Carbon-Water-Energy Reanalysis v1100.1, 2001-2021. "
                f"ORNL DAAC. https://doi.org/{DOI}"
            ),
            region="global",
            source="CARbon DAta MOdel fraMework (CARDAMOM) uses Bayesian inference and Markov Chain Monte Carlo (MCMC) optimization to calibrate parameters and initial conditions of the Data Assimilation Linked Ecosystem carbon-water-energy model (DALEC CWE) at each grid cell, integrating multiple satellite and ancillary observational constraints",
            source_data_retrieval_date=download_stamp,
            source_data_url="https://www.earthdata.nasa.gov/data/catalog/ornl-cloud-cardamom-gridded-fluxes-2492-1",
            source_id="CARDAMOM-1100-1",
            source_label="CARDAMOM",
            source_type="reanalysis",
            source_version_number="v1100.1",
            title=f"CARDAMOM Carbon-Water-Energy Reanalysis v1100.1, 2001-2021: {varname}",
            tracking_id=ild.output.new_tracking_id(),
            variable_id=varname,
            version=f"v{time.strftime('%Y%m%d')}",
        )

        # Export
        filename = ild.output.filename_from_attrs(out.attrs)
        out.to_netcdf(filename)


if __name__ == "__main__":
    main()
