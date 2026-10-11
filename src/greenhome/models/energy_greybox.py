"""V1 grey-box energy model: physics x learned residual (ARCHITECTURE 3.5 steps 1 and 3), scored on NEED.

    log(prediction) = log(physics) + residual(parity features, log physics)

The residual is a LightGBM regression on log(actual / physics), with the same settings, features, folds and
splits as V2 (energy_ml), so the V1 vs V2 comparison is like for like. Also reported: physics only (zero
training) and physics with one global scale factor fitted on the training fold.

    uv run python -m greenhome.models.energy_greybox      -> eval/results/energy_v1.json

Bill calibration (k = actual / physics heating slope, step 2) applies only to homes with monthly bills, so the
headline NEED metrics use k = 1. The test split is never scored here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from greenhome.models.energy_ml import RESULTS, fit_lgb, metrics, predict_all
from greenhome.models.need_data import dataset
from greenhome.models.physics import need_physics


def with_physics(x: pd.DataFrame, fuel: str) -> pd.DataFrame:
    xx = x.assign(gas_heated=1) if fuel == "gas" and "gas_heated" not in x else x
    return x.assign(log_physics=np.log(need_physics(xx)[f"{fuel}_kwh"].to_numpy()))


def predict_v1(xtr, ytr, xev, fuel) -> dict[str, np.ndarray]:
    """Physics-only, physics x global scale, and V1 (physics x residual GBDT) log predictions for xev."""
    ptr, pev = xtr.log_physics.to_numpy(), xev.log_physics.to_numpy()
    scale = float(np.median(ytr.to_numpy() - ptr))
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    resid = ytr - xtr.log_physics
    m = fit_lgb(xtr[~es], resid[~es], xtr[es], resid[es], fuel, log_ratio_target=True)
    # Duan smearing factor: exp(log prediction) is the typical home; x smearing gives the expected (mean) use,
    # which totals (carbon, portfolio) need. Reported here, applied by the EnergyModel interface.
    smear = float(np.mean(np.exp(ytr.to_numpy() - (ptr + m.predict(xtr)))))
    return {"physics_only": pev, "physics_scaled": pev + scale, "v1_greybox": pev + m.predict(xev),
            "_scale": scale, "_iters": m.best_iteration_, "_smear": smear}


def evaluate(fuel: str) -> dict:
    x, y, split = dataset(fuel)
    x = with_physics(x, fuel)
    xtr, ytr, xva, yva = x[split == "train"], y[split == "train"], x[split == "val"], y[split == "val"]
    plain = lambda df: df.drop(columns="log_physics")   # V2 and the baselines never see the physics column
    out = {"n_train": len(xtr), "n_val": len(xva), "cv": {}, "val": {}}
    folds: dict[str, list] = {}
    for tr, ev in KFold(5, shuffle=True, random_state=42).split(xtr):
        a, b = xtr.iloc[tr], xtr.iloc[ev]
        preds = {**predict_all(plain(a), ytr.iloc[tr], plain(b), fuel), **predict_v1(a, ytr.iloc[tr], b, fuel)}
        for k, v in preds.items():
            if not k.startswith("_"):
                folds.setdefault(k, []).append(metrics(ytr.iloc[ev].to_numpy(), v))
    for k, s in folds.items():
        out["cv"][k] = {m: round(float(np.mean([f[m] for f in s])), 4) for m in s[0]}
        out["cv"][k]["mae_kwh_sd"] = round(float(np.std([f["mae_kwh"] for f in s])), 1)
    preds = {**predict_all(plain(xtr), ytr, plain(xva), fuel), **predict_v1(xtr, ytr, xva, fuel)}
    out["physics_scale_factor"] = round(float(np.exp(preds.pop("_scale"))), 3)
    out["v1_iterations"] = int(preds.pop("_iters"))
    out["v1_smearing_factor"] = round(preds.pop("_smear"), 4)
    preds.pop("_lgb_iters", None)
    out["val"] = {k: metrics(yva.to_numpy(), v) for k, v in preds.items()}
    return out


def main() -> None:
    results = {fuel: evaluate(fuel) for fuel in ("gas", "elec")}
    (RESULTS / "energy_v1.json").write_text(json.dumps(results, indent=2))
    for fuel, r in results.items():
        print(f"\n== {fuel} (physics scale factor {r['physics_scale_factor']}, V1 iters {r['v1_iterations']})")
        rows = [{"model": k, "cv_mae": r["cv"][k]["mae_kwh"], "cv_sd": r["cv"][k]["mae_kwh_sd"],
                 "cv_med_ape": r["cv"][k]["median_ape_pct"], "cv_r2_log": r["cv"][k]["r2_log"],
                 "val_mae": r["val"][k]["mae_kwh"], "val_r2_log": r["val"][k]["r2_log"]} for k in r["cv"]]
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
