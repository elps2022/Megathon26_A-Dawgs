"""Observed retrofit labels from NEED: per-home use before vs after a real install (ARCHITECTURE 2, 3.1).

For each NEED home with a loft insulation, cavity wall insulation or solar PV install in 2010-2020:
  pre  = mean annual use in the 2 years before the install year, post = mean in the 2 years after
  (the install year itself is dropped; all 4 readings must be valid "V" readings)
  change     = post / pre - 1
  did_change = (post / pre) / (median post / pre of never-treated homes over the same years) - 1
Homes with another intervention within 2 years of the install are dropped, gas labels (and their control
trend) only use gas-heated homes, and solar is measured on electricity. The rules match calibrate.need_retrofit_effects, which pools the
same comparison into one effect per measure; here each home keeps its own label and its fixed NEED split
(data/splits/need_split.csv), so the savings simulator can be validated on held-out real installs.

    uv run python -m greenhome.data.need_labels      (make labels)

Outputs: data/processed/need_retrofit_labels.parquet (gitignored, raw-derived) and
eval/results/need_retrofit_labels_summary.json (committed). The summary reports counts for every split but
medians for train + val only: the test labels stay unseen until the final savings evaluation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from greenhome.models.need_data import load_need, load_split

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "data" / "processed" / "need_retrofit_labels.parquet"
SUMMARY = ROOT / "eval" / "results" / "need_retrofit_labels_summary.json"
MEASURES = {"loft_insulation": ("LI", "gas"), "cavity_wall_insulation": ("CWI", "gas"), "solar_pv": ("PV", "elec")}
COLS = {"gas": ("Gcons", "GasValFlag"), "elec": ("Econs", "ElecValFlag")}
INSTALL_YEARS = range(2010, 2021)
WINDOW = 2


def window_use(d: pd.DataFrame, fuel: str, years: list[int]) -> pd.Series:
    """Mean annual use over the years, NaN unless every reading is valid."""
    cons, flag = COLS[fuel]
    valid = np.logical_and.reduce([d[f"{flag}{y}"] == "V" for y in years])
    return d[[f"{cons}{y}" for y in years]].mean(axis=1).where(valid)


def labels(d: pd.DataFrame, split: pd.Series) -> pd.DataFrame:
    never = (d.LI_FLAG == 0) & (d.CWI_FLAG == 0) & (d.PV_FLAG == 0)
    rows = []
    for measure, (code, fuel) in MEASURES.items():
        for year in INSTALL_YEARS:
            pre_y, post_y = list(range(year - WINDOW, year)), list(range(year + 1, year + WINDOW + 1))
            treated = (d[f"{code}_FLAG"] == 1) & (d[f"{code}_DATE"] == year)
            for other in {"LI", "CWI", "PV"} - {code}:
                treated &= ~d[f"{other}_DATE"].between(year - WINDOW, year + WINDOW)
            if fuel == "gas":
                treated &= d.MAIN_HEAT_FUEL == 1
            pre, post = window_use(d, fuel, pre_y), window_use(d, fuel, post_y)
            ratio = post / pre
            ctrl = never & (split != "test") & ((d.MAIN_HEAT_FUEL == 1) if fuel == "gas" else True)
            control = ratio[ctrl].median()  # reference trend: never-treated, non-test (gas-heated for gas)
            t = d.index[treated & ratio.notna() & (pre > 0)]
            rows.append(pd.DataFrame({
                "need_row": d.need_row[t], "split": split[t], "measure": measure, "fuel": fuel,
                "install_year": year, "pre_kwh": pre[t].round(1), "post_kwh": post[t].round(1),
                "change": (ratio[t] - 1).round(4), "control_ratio": round(float(control), 4),
                "did_change": (ratio[t] / control - 1).round(4),
                "prop_type": d.PROP_TYPE[t], "floor_area_band": d.FLOOR_AREA_BAND[t],
                "age_band": d.PROP_AGE_BAND[t], "region": d.REGION[t]}))
    return pd.concat(rows, ignore_index=True)


def main() -> None:
    d = load_need()
    out = labels(d, load_split(d))
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    seen = out[out.split != "test"]
    summary = {
        "source": "DESNZ NEED anonymised data 2025, 50k sample (valid readings only)",
        "method": f"per-home post/pre mean use, {WINDOW} years each side, install year dropped, installs "
                  f"{INSTALL_YEARS.start}-{INSTALL_YEARS.stop - 1}; DiD against never-treated non-test homes",
        "labels_by_measure_and_split": {m: g.split.value_counts().to_dict() for m, g in out.groupby("measure")},
        "train_val_median_did_change": {m: round(float(g.did_change.median()), 4) for m, g in seen.groupby("measure")},
        "train_val_median_change_kwh": {m: round(float((g.post_kwh - g.pre_kwh).median()), 1)
                                        for m, g in seen.groupby("measure")},
        "test_medians": "withheld until the final savings evaluation",
    }
    SUMMARY.write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
