"""Derive small calibration tables in config/ from the raw public data in data/raw/.

Only this script reads data/raw/. Its outputs are aggregate statistics (no addresses), so they are
committed, and teammates can run the synthetic generator without the raw downloads.

    uv run python -m greenhome.data.calibrate
"""

from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
CONFIG = ROOT / "config"

NEED_URL = "https://www.gov.uk/government/statistics/national-energy-efficiency-data-framework-need-anonymised-data-2025"
METOFFICE_URL = "https://www.metoffice.gov.uk/research/climate/maps-and-data/uk-and-regional-series"
PPD_URL = "https://www.gov.uk/government/statistical-data-sets/price-paid-data-downloads"
HPI_URL = "https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-july-2026"

# NEED_YEAR is the latest year in the 2025 release; post-2022 levels match current bills better than 2019.
NEED_YEAR = 2023
NEED_FACTORS = ["PROP_TYPE", "FLOOR_AREA_BAND", "PROP_AGE_BAND", "EPC"]

# Local authority -> (ONS region used as Property.region, Met Office areal series file)
LA_REGIONS = {
    "Leeds": ("Yorkshire and The Humber", "Tmean_England_E_and_NE.txt"),
    "Cornwall": ("South West", "Tmean_England_SW_and_S_Wales.txt"),
    "Southwark": ("London", "Tmean_England_SE_and_Central_S.txt"),
}
HDD_BASE_C = 15.5
PPD_COLS = ["tid", "price", "date", "postcode", "ptype", "new_build", "tenure", "paon", "saon",
            "street", "locality", "town", "district", "county", "ppd_cat", "status"]


def fit_loglinear(df: pd.DataFrame, target: str) -> dict:
    """OLS of log(target) on one-hot NEED bands. Returns intercept, per-level effects, residual sd."""
    x = pd.get_dummies(df[NEED_FACTORS].astype(str), drop_first=True, dtype=float)
    y = np.log(df[target].to_numpy())
    a = np.column_stack([np.ones(len(x)), x.to_numpy()])
    beta, *_ = np.linalg.lstsq(a, y, rcond=None)
    resid = y - a @ beta
    effects: dict[str, dict[str, float]] = {f: {} for f in NEED_FACTORS}
    for f in NEED_FACTORS:
        for lvl in sorted(df[f].astype(str).unique()):
            col = f"{f}_{lvl}"
            effects[f][lvl] = round(float(beta[1 + list(x.columns).index(col)]), 4) if col in x.columns else 0.0
    return {"intercept": round(float(beta[0]), 4), "effects": effects,
            "resid_sd": round(float(resid.std()), 4), "n": int(len(df))}


def need_retrofit_effects(d: pd.DataFrame) -> dict:
    """Difference-in-differences on NEED: change in use from 2 years before to 2 years after an install,
    relative to the same change in never-treated homes. Returns the median fractional change."""
    out = {}
    specs = {"loft_insulation": ("LI", "G"), "cavity_wall_insulation": ("CWI", "G"), "solar_pv": ("PV", "E")}
    never = d[(d.LI_FLAG == 0) & (d.CWI_FLAG == 0) & (d.PV_FLAG == 0)]
    for name, (flag, fuel) in specs.items():
        cons = "Gcons" if fuel == "G" else "Econs"
        ratios = []
        for year in range(2010, 2021):
            pre = [f"{cons}{y}" for y in (year - 2, year - 1)]
            post = [f"{cons}{y}" for y in (year + 1, year + 2)]
            treated = d[(d[f"{flag}_FLAG"] == 1) & (d[f"{flag}_DATE"] == year)]
            # Only gas-heated homes for gas effects, and no other intervention in the window.
            others = [o for o in ("LI", "CWI", "PV") if o != flag]
            for o in others:
                treated = treated[~treated[f"{o}_DATE"].between(year - 2, year + 2)]
            if fuel == "G":
                treated = treated[treated.MAIN_HEAT_FUEL == 1]
            t = (treated[post].mean(axis=1) / treated[pre].mean(axis=1)).dropna()
            c = (never[post].mean(axis=1) / never[pre].mean(axis=1)).dropna()
            if len(t) >= 20:
                ratios.append((float(t.median() / c.median()), len(t)))
        w = np.array([n for _, n in ratios], dtype=float)
        r = np.array([v for v, _ in ratios])
        out[name] = {"fuel": "gas" if fuel == "G" else "electricity",
                     "median_fractional_change": round(float((r * w).sum() / w.sum()) - 1, 4),
                     "n_treated": int(w.sum()),
                     "method": "DiD of median post/pre ratio (2y each side), install years 2010-2020"}
    return out


def calibrate_need() -> None:
    d = pd.read_csv(RAW / "need" / "anon2025_50k.csv")
    with_epc = d[d.EPC != "No EPC"]
    gas = with_epc[(with_epc.MAIN_HEAT_FUEL == 1) & (with_epc[f"GasValFlag{NEED_YEAR}"] == "V")]
    elec = with_epc[with_epc[f"ElecValFlag{NEED_YEAR}"] == "V"]
    out = {
        "source": f"DESNZ NEED anonymised data 2025, 50k sample ({NEED_URL})",
        "year": NEED_YEAR,
        "bands": {
            "PROP_AGE_BAND": {"1": "before 1930", "2": "1930-1972", "3": "1973-1999", "4": "2000 or later"},
            "FLOOR_AREA_BAND": {"1": "<=50", "2": "51-100", "3": "101-150", "4": "151-200", "5": ">200"},
        },
        "gas_kwh_gas_heated": fit_loglinear(gas, f"Gcons{NEED_YEAR}"),
        "elec_kwh_gas_heated": fit_loglinear(elec[elec.MAIN_HEAT_FUEL == 1], f"Econs{NEED_YEAR}"),
        "elec_kwh_not_gas_heated": fit_loglinear(elec[elec.MAIN_HEAT_FUEL == 2], f"Econs{NEED_YEAR}"),
        "retrofit_effects": need_retrofit_effects(d),
    }
    (CONFIG / "need_calibration.yaml").write_text(yaml.safe_dump(out, sort_keys=False))


def calibrate_degree_days() -> None:
    rows = []
    for la, (region, fname) in LA_REGIONS.items():
        lines = (RAW / "metoffice" / fname).read_text().splitlines()
        start = next(i for i, l in enumerate(lines) if l.startswith("year"))
        for line in lines[start + 1:]:
            parts = line.split()
            year = int(parts[0])
            if year < 2015:
                continue
            for m in range(1, 13):
                if m >= len(parts) or parts[m] == "---":
                    continue
                t = float(parts[m])
                days = calendar.monthrange(year, m)[1]
                rows.append({"region": region, "local_authority": la, "year": year, "month": m,
                             "tmean_c": t, "hdd": round(days * max(0.0, HDD_BASE_C - t), 1)})
    df = pd.DataFrame(rows)
    header = (f"# source: Met Office HadUK-Grid regional monthly Tmean ({METOFFICE_URL}).\n"
              f"# hdd = days_in_month * max(0, {HDD_BASE_C} - monthly mean); a monthly-mean approximation that\n"
              "# understates HDD in shoulder months. Swap for daily-based HDD if time allows.\n")
    path = CONFIG / "degree_days.csv"
    path.write_text(header + df.to_csv(index=False))


def calibrate_prices() -> None:
    ppd = pd.concat(
        [pd.read_csv(f, header=None, names=PPD_COLS) for f in sorted((RAW / "ppd").glob("pp-*.csv"))
         if f.stat().st_size > 0])
    ppd = ppd[(ppd.ppd_cat == "A") & ppd.district.isin([la.upper() for la in LA_REGIONS])]
    ppd["year"] = ppd.date.str[:4].astype(int)
    med = ppd.groupby(["district", "ptype"]).price.median()
    hpi = pd.read_csv(RAW / "hpi" / "UK-HPI-full-file-2026-07.csv", usecols=["Date", "RegionName", "Index"])
    hpi = hpi[hpi.RegionName.isin(LA_REGIONS)]
    hpi["Date"] = pd.to_datetime(hpi.Date, format="%d/%m/%Y")
    hpi = hpi[hpi.Date >= "2015-01-01"].sort_values(["RegionName", "Date"])
    out = {
        "source": f"HM Land Registry Price Paid Data, category A, {ppd.year.min()}-{ppd.year.max()} ({PPD_URL})",
        "years": sorted(int(y) for y in ppd.year.unique()),
        "median_price_by_type": {
            la: {t: int(med.get((la.upper(), t), np.nan)) for t in "DSTF" if (la.upper(), t) in med}
            for la in LA_REGIONS
        },
        "reference_date": f"{ppd.year.min()}-{ppd.year.max()} pooled; re-based with UK HPI by sale date",
        "n_sales": int(len(ppd)),
    }
    (CONFIG / "price_calibration.yaml").write_text(yaml.safe_dump(out, sort_keys=False))
    hpi_out = hpi.rename(columns={"RegionName": "local_authority", "Date": "date", "Index": "hpi"})
    (CONFIG / "hpi_index.csv").write_text(f"# source: UK House Price Index July 2026 ({HPI_URL})\n"
                                          + hpi_out.to_csv(index=False, date_format="%Y-%m-%d"))


def main() -> None:
    CONFIG.mkdir(exist_ok=True)
    calibrate_need()
    calibrate_degree_days()
    calibrate_prices()
    print(f"wrote calibration tables to {CONFIG}")


if __name__ == "__main__":
    main()
