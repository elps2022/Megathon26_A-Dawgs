"""NEED loading, cleaning, features and the fixed train/val/test split for the energy model (ARCHITECTURE §3.5).

Prediction moment: estimate a home's typical annual gas / electricity from its description only (no own
consumption history). Every feature must also be derivable for an EPC home (feature parity rule).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
NEED_PATH = ROOT / "data" / "raw" / "need" / "anon2025_50k.csv"
SPLIT_PATH = ROOT / "data" / "splits" / "need_split.csv"

TARGET_YEAR = 2023
EPC_ORDINAL = {"A/B": 1, "C": 2, "D": 3, "E": 4, "F/G": 5}  # "No EPC" -> NaN, with has_epc = 0
CATEGORICAL = ["prop_type", "region", "conservatory"]
INSTALLS = {"loft": "LI", "cavity": "CWI", "pv": "PV"}
# COUNCIL_TAX_BAND is deliberately excluded: EPC homes don't have it (feature parity).


def load_need(path: Path = NEED_PATH) -> pd.DataFrame:
    d = pd.read_csv(path)
    d.insert(0, "need_row", np.arange(len(d)))
    return d


def make_split(d: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """Fixed property-level 70/15/15 split, stratified by region x main heating fuel, shared by both fuels."""
    rng = np.random.default_rng(seed)
    split = pd.Series("train", index=d.index)
    for _, idx in d.groupby(["REGION", "MAIN_HEAT_FUEL"]).groups.items():
        idx = rng.permutation(np.asarray(idx))
        n_test, n_val = round(0.15 * len(idx)), round(0.15 * len(idx))
        split[idx[:n_test]] = "test"
        split[idx[n_test:n_test + n_val]] = "val"
    return pd.DataFrame({"need_row": d.need_row, "split": split})


def load_split(d: pd.DataFrame) -> pd.Series:
    """Create the split once and persist it, so the test set never moves between runs."""
    if not SPLIT_PATH.exists():
        SPLIT_PATH.parent.mkdir(parents=True, exist_ok=True)
        make_split(d).to_csv(SPLIT_PATH, index=False)
    s = pd.read_csv(SPLIT_PATH)
    return d[["need_row"]].merge(s, on="need_row", how="left")["split"].set_axis(d.index)


def features(d: pd.DataFrame, year: int = TARGET_YEAR) -> pd.DataFrame:
    x = pd.DataFrame(index=d.index)
    x["prop_type"] = d.PROP_TYPE
    x["age_band"] = d.PROP_AGE_BAND            # 1 = pre-1930 ... 4 = 2000+
    x["floor_area_band"] = d.FLOOR_AREA_BAND   # 1 = <=50 m2 ... 5 = >200 m2
    x["epc_band"] = d.EPC.map(EPC_ORDINAL)
    x["has_epc"] = (d.EPC != "No EPC").astype(int)
    x["region"] = d.REGION
    x["conservatory"] = d.CONSERVATORY_FLAG.map({0.0: "no", 1.0: "yes"}).fillna("unknown")
    x["imd_quintile"] = d.IMD_BAND_ENG.fillna(d.IMD_BAND_WALES)
    x["gas_heated"] = (d.MAIN_HEAT_FUEL == 1).astype(int)
    for name, code in INSTALLS.items():
        # Only installs completed before the target year are known at prediction time.
        before = (d[f"{code}_FLAG"] == 1) & (d[f"{code}_DATE"] < year)
        x[f"{name}_installed"] = before.astype(int)
        x[f"years_since_{name}"] = np.where(before, year - d[f"{code}_DATE"], np.nan)
    for c in CATEGORICAL:
        x[c] = x[c].astype("category")
    return x


def dataset(fuel: str, year: int = TARGET_YEAR, d: pd.DataFrame | None = None) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, log target, split) for fuel in {"gas", "elec"}; valid ('V') readings only."""
    d = load_need() if d is None else d
    split = load_split(d)
    if fuel == "gas":
        keep = (d.MAIN_HEAT_FUEL == 1) & (d[f"GasValFlag{year}"] == "V")
        y = d[f"Gcons{year}"]
    elif fuel == "elec":
        keep = d[f"ElecValFlag{year}"] == "V"
        y = d[f"Econs{year}"]
    else:
        raise ValueError(fuel)
    # Homes that installed a measure during the target year have a partly-treated target: drop them.
    for code in INSTALLS.values():
        keep &= d[f"{code}_DATE"] != year
    x = features(d)[keep]
    if fuel == "gas":
        x = x.drop(columns="gas_heated")
    return x, np.log(y[keep]).rename("log_kwh"), split[keep]
