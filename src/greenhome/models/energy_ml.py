"""Energy model first cut: baselines vs LightGBM per fuel on NEED (ARCHITECTURE §3.5, steps 1-4).

    uv run python -m greenhome.models.energy_ml

Reports 5-fold CV on train and the val score. The test split is never scored here. Uses the "parity" feature
set (what EPC homes can supply); `--feature-sets` compares full / parity / no_installs on CV instead.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from greenhome.models.need_data import ROOT, dataset

RESULTS = ROOT / "eval" / "results"
BANDS = ["prop_type", "age_band", "floor_area_band", "epc_band", "region"]
# Back-off chain for the "similar homes" median: drop the least important band first.
BACKOFF = [BANDS, ["prop_type", "age_band", "floor_area_band", "epc_band"],
           ["prop_type", "floor_area_band", "epc_band"], ["prop_type", "floor_area_band"], []]
MIN_CELL = 20

LGB_PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=31, min_child_samples=40,
                  feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
                  n_estimators=3000, verbose=-1, random_state=42)


def monotone(cols: list[str], fuel: str) -> list[int]:
    up = {"epc_band", "floor_area_band"}            # worse band / bigger home -> more energy
    down = {"loft_installed", "cavity_installed"} if fuel == "gas" else {"pv_installed"}
    return [1 if c in up else -1 if c in down else 0 for c in cols]


class CellMedian:
    """No-model reference: median log use of similar homes, backing off to coarser cells when sparse."""

    def fit(self, x: pd.DataFrame, y: pd.Series) -> "CellMedian":
        df = x[BANDS].astype(str).assign(y=y.values)
        self.tables = []
        for keys in BACKOFF[:-1]:
            g = df.groupby(keys).y.agg(["median", "size"])
            self.tables.append((keys, g[g["size"] >= MIN_CELL]["median"]))
        self.global_median = float(y.median())
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        df = x[BANDS].astype(str)
        out = pd.Series(np.nan, index=x.index)
        for keys, table in self.tables:
            idx = pd.MultiIndex.from_frame(df[keys]) if len(keys) > 1 else df[keys[0]]
            vals = pd.Series(table.reindex(idx).to_numpy(), index=x.index)
            out = out.fillna(vals)
        return out.fillna(self.global_median).to_numpy()


def log_linear() -> Pipeline:
    """The log-linear model (as in config/need_calibration.yaml), with all features one-hot."""
    pre = ColumnTransformer([("oh", OneHotEncoder(handle_unknown="ignore"),
                              BANDS + ["conservatory", "imd_quintile"])], remainder="drop")
    return Pipeline([("pre", pre), ("ridge", Ridge(alpha=1.0))])


PARAMS_PATH = ROOT / "config" / "energy_params.yaml"


def tuned_params(fuel: str) -> dict:
    """Optuna-tuned settings per fuel (config/energy_params.yaml, written by models/tune.py); {} if untuned."""
    import yaml
    if not PARAMS_PATH.exists():
        return {}
    return dict(yaml.safe_load(PARAMS_PATH.read_text()).get(fuel, {}))


class _RawTarget:
    """Fits a raw-kWh objective (Tweedie) on exp(log target) but predicts log kWh, like the other models."""

    def __init__(self, model):
        self.m = model
        self.best_iteration_ = model.best_iteration_

    def predict(self, x):
        return np.log(np.clip(self.m.predict(x), 1e-6, None))


def fit_lgb(xtr, ytr, xva, yva, fuel, tuned: bool = True, log_ratio_target: bool = False):
    """LightGBM on a log target. tuned=True uses config/energy_params.yaml. A log-ratio target (V1's residual,
    which can be negative) can't use Tweedie, so it keeps the other tuned settings with an L2 objective."""
    params = {**LGB_PARAMS, **(tuned_params(fuel) if tuned else {})}
    raw = params.get("objective") == "tweedie"
    if raw and log_ratio_target:
        params["objective"], raw = "regression", False
        params.pop("tweedie_variance_power", None)
    m = lgb.LGBMRegressor(**params, monotone_constraints=monotone(list(xtr.columns), fuel))
    t_tr, t_va = (np.exp(ytr), np.exp(yva)) if raw else (ytr, yva)
    m.fit(xtr, t_tr, eval_X=(xva,), eval_y=(t_va,), callbacks=[lgb.early_stopping(100, verbose=False)])
    return _RawTarget(m) if raw else m


def metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    y, p = np.exp(y_log), np.exp(p_log)
    ss_res = ((y_log - p_log) ** 2).sum()
    ss_tot = ((y_log - y_log.mean()) ** 2).sum()
    return {"mae_kwh": round(float(np.abs(y - p).mean()), 1),
            "mape_pct": round(float((np.abs(y - p) / y).mean() * 100), 2),
            "median_ape_pct": round(float(np.median(np.abs(y - p) / y) * 100), 2),
            "r2_log": round(float(1 - ss_res / ss_tot), 4)}


def predict_all(xtr, ytr, xev, fuel, xes=None, yes=None) -> dict[str, np.ndarray]:
    preds = {"global_median": np.full(len(xev), float(ytr.median())),
             "cell_median": CellMedian().fit(xtr, ytr).predict(xev),
             "log_linear": log_linear().fit(xtr.fillna({"imd_quintile": 0}), ytr).predict(xev.fillna({"imd_quintile": 0}))}
    if xes is None:  # carve a random early-stopping slice out of the training fold
        es = np.random.default_rng(0).random(len(xtr)) < 0.15
        m = fit_lgb(xtr[~es], ytr[~es], xtr[es], ytr[es], fuel)
    else:
        m = fit_lgb(xtr, ytr, xes, yes, fuel)
    preds["lightgbm"] = m.predict(xev)
    preds["_lgb_iters"] = m.best_iteration_
    return preds


def evaluate(fuel: str) -> dict:
    x, y, split = dataset(fuel)
    xtr, ytr = x[split == "train"], y[split == "train"]
    xva, yva = x[split == "val"], y[split == "val"]
    out = {"n_train": len(xtr), "n_val": len(xva), "cv": {}, "val": {}, "val_with_epc": {}}

    # 5-fold CV on train only.
    fold_scores: dict[str, list[dict]] = {}
    for tr, ev in KFold(5, shuffle=True, random_state=42).split(xtr):
        p = predict_all(xtr.iloc[tr], ytr.iloc[tr], xtr.iloc[ev], fuel)
        for k, v in p.items():
            if not k.startswith("_"):
                fold_scores.setdefault(k, []).append(metrics(ytr.iloc[ev].to_numpy(), v))
    for k, scores in fold_scores.items():
        out["cv"][k] = {m: round(float(np.mean([s[m] for s in scores])), 4) for m in scores[0]}
        out["cv"][k]["mae_kwh_sd"] = round(float(np.std([s["mae_kwh"] for s in scores])), 1)

    # Val: fit on train (early stopping on a train slice, so val stays a clean holdout).
    p = predict_all(xtr, ytr, xva, fuel)
    out["lgb_iterations"] = int(p.pop("_lgb_iters"))
    epc = (xva.has_epc == 1).to_numpy()
    for k, v in p.items():
        out["val"][k] = metrics(yva.to_numpy(), v)
        out["val_with_epc"][k] = metrics(yva.to_numpy()[epc], v[epc])
    return out


def compare_feature_sets() -> dict:
    """LightGBM 5-fold CV on train for each feature set (need_data.FEATURE_SETS): the parity check, 3.5."""
    from greenhome.models.need_data import FEATURE_SETS
    out = {}
    for fuel in ("gas", "elec"):
        out[fuel] = {}
        for name in FEATURE_SETS:
            x, y, split = dataset(fuel, feature_set=name)
            xtr, ytr = x[split == "train"], y[split == "train"]
            maes = []
            for tr, ev in KFold(5, shuffle=True, random_state=42).split(xtr):
                es = np.random.default_rng(0).random(len(tr)) < 0.15
                m = fit_lgb(xtr.iloc[tr][~es], ytr.iloc[tr][~es], xtr.iloc[tr][es], ytr.iloc[tr][es], fuel)
                maes.append(metrics(ytr.iloc[ev].to_numpy(), m.predict(xtr.iloc[ev]))["mae_kwh"])
            out[fuel][name] = {"cv_mae_kwh": round(float(np.mean(maes)), 1), "cv_mae_sd": round(float(np.std(maes)), 1),
                               "features": list(x.columns)}
            print(fuel, name, out[fuel][name]["cv_mae_kwh"], "+/-", out[fuel][name]["cv_mae_sd"])
    return out


def main() -> None:
    import sys
    RESULTS.mkdir(parents=True, exist_ok=True)
    if "--feature-sets" in sys.argv:
        (RESULTS / "energy_feature_sets.json").write_text(json.dumps(compare_feature_sets(), indent=2))
        return
    results = {fuel: evaluate(fuel) for fuel in ("gas", "elec")}
    (RESULTS / "energy_first_cut.json").write_text(json.dumps(results, indent=2))
    for fuel, r in results.items():
        print(f"\n== {fuel}  (train {r['n_train']:,}, val {r['n_val']:,}, lgb iters {r['lgb_iterations']})")
        rows = []
        for k in r["cv"]:
            rows.append({"model": k, **{f"cv_{m}": v for m, v in r["cv"][k].items()},
                         "val_mae": r["val"][k]["mae_kwh"], "val_epc_mae": r["val_with_epc"][k]["mae_kwh"],
                         "val_r2_log": r["val"][k]["r2_log"]})
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
