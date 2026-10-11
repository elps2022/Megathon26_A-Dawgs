"""Stage 2 feature engineering (ARCHITECTURE 2, 3.2-3.5): one row of features per home.

  - energy-model inputs with exact NEED parity (models.need_data.epc_features)
  - weather-normalised annual use (data.degree_day_fit): gas for gas-heated homes, electricity for electrically
    heated ones; other electricity stays as annualised, matching NEED (gas weather corrected, electricity not)
  - intensities: kWh/m2 per fuel and purchased (grid import + gas), from the weather-normalised figures
  - coverage: months observed per fuel and the share of a typical year they cover
Unavailable values stay NaN with a flag in feature_flags, never 0.

    uv run python -m greenhome.data.features              real book  -> data/processed/real/features.parquet
    uv run python -m greenhome.data.features --interim    synthetic  -> data/generated/features.csv

Bills are never energy-model inputs (no lagged consumption, 3.5); they feed normalisation, V1 calibration
and the "worse than similar homes" detector (3.4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from greenhome.data.annualise import ELEC_SEASONAL_PROFILE, GAS_SEASONAL_PROFILE
from greenhome.data.degree_day_fit import fit, heating_fuel
from greenhome.models.need_data import PARITY_FEATURES, epc_features

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "data" / "processed" / "real"
INTERIM = ROOT / "data" / "generated"
SUMMARY = ROOT / "eval" / "results" / "features_summary.json"


def load(interim: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    if interim:
        return pd.read_csv(INTERIM / "properties.csv"), pd.read_csv(INTERIM / "monthly_energy.csv")
    parts = sorted((REAL / "monthly_energy").glob("part-*.parquet"))
    return pd.read_parquet(REAL / "properties.parquet"), pd.concat(pd.read_parquet(f) for f in parts)


def coverage(monthly: pd.DataFrame) -> pd.DataFrame:
    """Months observed and typical-year share covered (distinct calendar months), per fuel."""
    m = monthly.assign(cal=pd.to_datetime(monthly.month).dt.month)
    out = {}
    for fuel, col, prof in (("elec", "elec_import_kwh", ELEC_SEASONAL_PROFILE), ("gas", "gas_kwh", GAS_SEASONAL_PROFILE)):
        obs = m[m[col].notna()]
        out[f"{fuel}_months"] = obs.groupby("property_id").size()
        share = pd.Series(prof)
        out[f"{fuel}_year_share"] = (obs.drop_duplicates(["property_id", "cal"]).assign(s=lambda d: d.cal.map(share))
                                     .groupby("property_id").s.sum().round(3))
    return pd.DataFrame(out)


def build(props: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    x = epc_features(props)
    x.insert(0, "property_id", props.property_id.to_numpy())
    x.insert(1, "heating_fuel", heating_fuel(props).to_numpy())   # from the home, so homes with no fit still count
    f = x.merge(fit(props, monthly).drop(columns="heating_fuel"), on="property_id", how="left").merge(
        coverage(monthly), left_on="property_id", right_index=True, how="left")
    f["fit_ok"] = f.fit_ok.fillna(False).astype(bool)
    f["fit_months"] = f.fit_months.fillna(0).astype(int)
    base = props.set_index("property_id")[["annual_elec_kwh", "annual_gas_kwh", "floor_area_m2", "data_flags"]]
    f = f.join(base, on="property_id")
    for c in ("elec_months", "gas_months"):
        f[c] = f[c].fillna(0).astype(int)
    flags = [[] for _ in range(len(f))]

    # Weather-normalised annual use: the heating fuel is normalised when its fit is usable; else annualised + flag
    wn = f.annual_heating_fuel_kwh_normalised
    f["annual_gas_kwh_wn"] = np.where((f.heating_fuel == "gas") & wn.notna(), wn, f.annual_gas_kwh)
    f["annual_elec_kwh_wn"] = np.where((f.heating_fuel == "elec") & wn.notna(), wn, f.annual_elec_kwh)
    for i in np.flatnonzero(f.heating_fuel.notna() & wn.isna()):
        flags[i].append("weather_normalisation_unavailable")
    for i in np.flatnonzero(f.data_flags.fillna("").str.contains("heating_fuel_unmetered")):
        flags[i].append("heating_fuel_unmetered")

    # Intensities (kWh/m2/yr); unavailable when floor area or a fuel is missing
    area = f.floor_area_m2.where(f.floor_area_m2 > 0)
    f["elec_kwh_per_m2"] = (f.annual_elec_kwh_wn / area).round(1)
    f["gas_kwh_per_m2"] = (f.annual_gas_kwh_wn / area).round(1)
    f["purchased_kwh_per_m2"] = ((f.annual_elec_kwh_wn + f.annual_gas_kwh_wn) / area).round(1)
    for i in np.flatnonzero(f.purchased_kwh_per_m2.isna()):
        flags[i].append("intensity_unavailable")
    f["feature_flags"] = [",".join(fl) for fl in flags]
    cols = (["property_id"] + PARITY_FEATURES +
            ["heating_fuel", "fit_months", "base_kwh_per_month", "slope_kwh_per_hdd", "fit_r2", "fit_ok",
             "annual_elec_kwh_wn", "annual_gas_kwh_wn", "elec_kwh_per_m2", "gas_kwh_per_m2", "purchased_kwh_per_m2",
             "elec_months", "gas_months", "elec_year_share", "gas_year_share", "feature_flags"])
    return f[cols]


def summarise(f: pd.DataFrame, props: pd.DataFrame) -> dict:
    by_fuel = {}
    for fuel, raw in (("gas", "annual_gas_kwh"), ("elec", "annual_elec_kwh")):
        h = f[f.heating_fuel == fuel]
        r = props.set_index("property_id").loc[h.property_id, raw].to_numpy()
        ratio = h[f"annual_{fuel}_kwh_wn"].to_numpy() / r
        by_fuel[fuel] = {"homes": len(h), "fit_ok_share": round(float(h.fit_ok.mean()), 4),
                         "median_r2": round(float(h.fit_r2.median()), 3),
                         "median_normalised_over_annualised": round(float(np.nanmedian(ratio)), 4)}
    return {"homes": len(f), "heating_fuel_fits": by_fuel,
            "parity_features_missing_share": {c: round(float(f[c].isna().mean()), 4) for c in PARITY_FEATURES},
            "median_purchased_kwh_per_m2": round(float(f.purchased_kwh_per_m2.median()), 1),
            "intensity_available_share": round(float(f.purchased_kwh_per_m2.notna().mean()), 4),
            "flags": pd.Series([x for fl in f.feature_flags for x in fl.split(",") if x]).value_counts().to_dict()}


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--interim", action="store_true", help="build for the interim synthetic dataset")
    interim = ap.parse_args(argv).interim
    props, monthly = load(interim)
    f = build(props, monthly)
    if interim:
        f.to_csv(INTERIM / "features.csv", index=False)
        print(f"wrote {INTERIM / 'features.csv'} ({len(f):,} homes)")
    else:
        f.to_parquet(REAL / "features.parquet", index=False)
        summary = summarise(f, props)
        SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
        print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
