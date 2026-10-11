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
# Fixed category lists, so NEED (training) and EPC homes (prediction) get identical pandas codes.
CATEGORIES = {
    "prop_type": ["Bungalow", "Detached", "End terrace", "Flat", "Mid terrace", "Semi detached"],
    "region": ["E12000001", "E12000002", "E12000003", "E12000004", "E12000005", "E12000006", "E12000007",
               "E12000008", "E12000009", "W92000004"],
    "conservatory": ["no", "yes", "unknown"],
}
# COUNCIL_TAX_BAND is deliberately excluded: EPC homes don't have it (feature parity).
# Feature sets (ARCHITECTURE 3.5 parity rule). "parity" is what EPC homes can supply and is the default:
# NEED's loft install flag has no EPC proxy (a deep loft can be original) and EPC has no install dates.
FULL_FEATURES = ["prop_type", "age_band", "floor_area_band", "epc_band", "has_epc", "region", "conservatory",
                 "imd_quintile", "gas_heated", "loft_installed", "years_since_loft", "cavity_installed",
                 "years_since_cavity", "pv_installed", "years_since_pv"]
PARITY_FEATURES = ["prop_type", "age_band", "floor_area_band", "epc_band", "has_epc", "region", "conservatory",
                   "imd_quintile", "gas_heated", "cavity_installed", "pv_installed"]
FEATURE_SETS = {"full": FULL_FEATURES, "parity": PARITY_FEATURES,
                "no_installs": [c for c in PARITY_FEATURES if not c.endswith("_installed")]}
AGE_CODES = {"pre-1930": 1, "1930-1972": 2, "1973-1999": 3, "2000+": 4}


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
    x["age_band"] = d.PROP_AGE_BAND.astype(float)            # 1 = pre-1930 ... 4 = 2000+
    x["floor_area_band"] = d.FLOOR_AREA_BAND.astype(float)   # 1 = <=50 m2 ... 5 = >200 m2
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
    return _categorise(x)


def _categorise(x: pd.DataFrame) -> pd.DataFrame:
    for c in CATEGORICAL:
        x[c] = pd.Categorical(x[c], categories=CATEGORIES[c])
    return x


def floor_area_band(area: pd.Series) -> pd.Series:
    """NEED FLOOR_AREA_BAND: 1 <=50, 2 51-100, 3 101-150, 4 151-200, 5 >200 m2 (NaN if unknown)."""
    area = pd.to_numeric(area, errors="coerce")  # None (e.g. from the API) -> NaN
    return pd.cut(area, [0, 50, 100, 150, 200, np.inf], labels=[1, 2, 3, 4, 5]).astype(float)


def epc_features(p: pd.DataFrame) -> pd.DataFrame:
    """The same feature columns, codes and dtypes as features(), for EPC homes (real book or interim
    synthetic properties). Anything an EPC home can't supply is NaN / "unknown", never invented."""
    import yaml
    region_codes = yaml.safe_load((ROOT / "config" / "epc_mapping.yaml").read_text())["need_region_codes"]
    rating = p.energy_rating.where(p.energy_rating.isin(list("ABCDEFG")))
    need_band = rating.map({"A": "A/B", "B": "A/B", "C": "C", "D": "D", "E": "E", "F": "F/G", "G": "F/G"})
    x = pd.DataFrame(index=p.index)
    x["prop_type"] = p.property_type.where(p.property_type.isin(CATEGORIES["prop_type"]))
    x["age_band"] = p.age_band.map(AGE_CODES).astype(float)
    x["floor_area_band"] = floor_area_band(p.floor_area_m2)
    x["epc_band"] = need_band.map(EPC_ORDINAL)
    x["has_epc"] = rating.notna().astype(int)
    x["region"] = p.region.map(region_codes)
    x["conservatory"] = "unknown"                     # EPC has no conservatory field (81% unknown in NEED too)
    x["imd_quintile"] = p["imd_quintile"].astype(float) if "imd_quintile" in p else np.nan
    x["gas_heated"] = (p.gas_meter.astype(bool) & p.heating.isin(["gas_condensing", "gas_non_condensing"])).astype(int)
    x["cavity_installed"] = (p["cavity_filled_retrofit"].astype(bool).astype(int)
                             if "cavity_filled_retrofit" in p else np.nan)
    x["pv_installed"] = (p.solar_kwp.fillna(0) > 0).astype(int)
    return _categorise(x)[PARITY_FEATURES]


def dataset(fuel: str, year: int = TARGET_YEAR, d: pd.DataFrame | None = None,
            feature_set: str = "parity") -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """(X, log target, split) for fuel in {"gas", "elec"}; valid ('V') readings only. feature_set picks
    the columns (FEATURE_SETS); "parity" is the default, matching what EPC homes can supply."""
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
    x = features(d)[keep][FEATURE_SETS[feature_set]]
    if fuel == "gas":
        x = x.drop(columns="gas_heated")
    return x, np.log(y[keep]).rename("log_kwh"), split[keep]
