"""V1 grey-box energy model: physics x learned residual (ARCHITECTURE 3.5 steps 1 and 3), scored on NEED.

    log(prediction) = elasticity x log(physics) + residual(parity features)

The residual is a LightGBM regression on log(actual) - elasticity x log(physics), with the same settings,
features, folds and splits as V2 (energy_ml), so the V1 vs V2 comparison is like for like. The residual doesn't
see the physics value, so a retrofit changes the prediction by exactly (physics after / before)^elasticity:
smooth and never the wrong sign. (An earlier version fed log physics to the trees as a feature; its retrofit
deltas were step-shaped and, for ~18% of gas homes, said insulation raises use.) The gas elasticity is fitted
on the observed loft and cavity installs in the TRAIN split (real homes use about half the change physics
predicts: the prebound effect); the val-split installs check it. Electricity's is picked by CV (config).
Also reported: physics only (zero training) and physics with one global scale factor fitted on the training fold.

    uv run python -m greenhome.models.energy_greybox      -> eval/results/energy_v1.json

Bill calibration (k = actual / physics heating slope, step 2) applies only to homes with monthly bills, so the
headline NEED metrics use k = 1. The test split is never scored here.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from greenhome.models import physics
from greenhome.models.energy_ml import RESULTS, fit_lgb, metrics, predict_all
from greenhome.models.need_data import ROOT, dataset, features, load_need
from greenhome.models.physics import need_physics

LABELS = ROOT / "data" / "processed" / "need_retrofit_labels.parquet"
GAS_MEASURES = ("loft_insulation", "cavity_wall_insulation")


def with_physics(x: pd.DataFrame, fuel: str) -> pd.DataFrame:
    xx = x.assign(gas_heated=1) if fuel == "gas" and "gas_heated" not in x else x
    return x.assign(log_physics=np.log(need_physics(xx)[f"{fuel}_kwh"].to_numpy()))


def physics_retrofit_ratios(split: str | tuple[str, ...]) -> dict[str, pd.DataFrame]:
    """Per observed NEED install (need_labels) in the split(s): physics gas after / before and the observed
    difference-in-differences ratio. NEED has no loft depth, so a loft install is the age-band default -> 300 mm."""
    lab = pd.read_parquet(LABELS)
    lab = lab[lab.split.isin([split] if isinstance(split, str) else split)]
    f = features(load_need())
    eff = physics.cfg("physics.yaml")["need_gas_boiler_efficiency"]
    gas = lambda fab: (lambda d: (d.space_heat_kwh + d.hot_water_kwh) / eff)(physics.demands(fab)).to_numpy()
    out = {}
    for measure in GAS_MEASURES:
        L = lab[lab.measure == measure]
        x = f.loc[L.need_row.to_numpy()].assign(gas_heated=1)
        if measure == "cavity_wall_insulation":
            before, after = physics.need_fabric(x.assign(cavity_installed=0)), physics.need_fabric(x.assign(cavity_installed=1))
        else:
            before = physics.need_fabric(x)
            after = before.assign(loft_insulation_mm=np.maximum(before.loft_insulation_mm.astype(float), 300.0))
        out[measure] = pd.DataFrame({"physics_ratio": gas(after) / gas(before), "observed_ratio": 1 + L.did_change.to_numpy()})
    return out


def gas_elasticity() -> float:
    """Pooled elasticity of real use to the physics change, from TRAIN-split installs only:
    sum_m n_m log(observed median ratio_m) / sum_m n_m median log(physics ratio_m)."""
    r = physics_retrofit_ratios("train")
    num = sum(len(d) * np.log(np.median(d.observed_ratio)) for d in r.values())
    den = sum(len(d) * np.median(np.log(d.physics_ratio)) for d in r.values())
    return round(float(num / den), 3)


def elasticity(fuel: str) -> float:
    return gas_elasticity() if fuel == "gas" else float(physics.cfg("physics.yaml")["v1_elec_physics_elasticity"])


def fit_v1(xtr, ytr, fuel, beta: float):
    """V1 residual GBDT on log(actual) - beta x log(physics), from the description only (no physics column).
    Early stopping on a random 15% slice of the training rows, as everywhere else."""
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    resid = ytr - beta * xtr.log_physics
    f = xtr.drop(columns="log_physics")
    return fit_lgb(f[~es], resid[~es], f[es], resid[es], fuel, log_ratio_target=True)


def v1_log_pred(m, beta: float, x: pd.DataFrame) -> np.ndarray:
    return beta * x.log_physics.to_numpy() + m.predict(x.drop(columns="log_physics"))


def predict_v1(xtr, ytr, xev, fuel, beta: float) -> dict[str, np.ndarray]:
    """Physics-only, physics x global scale, and V1 (physics^beta x residual GBDT) log predictions for xev."""
    ptr, pev = xtr.log_physics.to_numpy(), xev.log_physics.to_numpy()
    scale = float(np.median(ytr.to_numpy() - ptr))
    m = fit_v1(xtr, ytr, fuel, beta)
    # Duan smearing factor: exp(log prediction) is the typical home; x smearing gives the expected (mean) use,
    # which totals (carbon, portfolio) need. Reported here, applied by the EnergyModel interface.
    smear = float(np.mean(np.exp(ytr.to_numpy() - v1_log_pred(m, beta, xtr))))
    return {"physics_only": pev, "physics_scaled": pev + scale, "v1_greybox": v1_log_pred(m, beta, xev),
            "_scale": scale, "_iters": m.best_iteration_, "_smear": smear}


def retrofit_check(beta: float) -> dict:
    """Predicted vs observed median change for real installs, train (fitted on) and val (held out). Test unseen."""
    out = {}
    for split in ("train", "val"):
        for measure, d in physics_retrofit_ratios(split).items():
            out.setdefault(measure, {})[split] = {
                "n": len(d), "observed_median_change": round(float(np.median(d.observed_ratio) - 1), 4),
                "physics_only_median_change": round(float(np.median(d.physics_ratio) - 1), 4),
                "v1_median_change": round(float(np.median(d.physics_ratio ** beta) - 1), 4)}
    return out


def evaluate(fuel: str) -> dict:
    x, y, split = dataset(fuel)
    x = with_physics(x, fuel)
    xtr, ytr, xva, yva = x[split == "train"], y[split == "train"], x[split == "val"], y[split == "val"]
    plain = lambda df: df.drop(columns="log_physics")   # V2 and the baselines never see the physics column
    beta = elasticity(fuel)
    out = {"n_train": len(xtr), "n_val": len(xva), "physics_elasticity": beta, "cv": {}, "val": {}}
    folds: dict[str, list] = {}
    for tr, ev in KFold(5, shuffle=True, random_state=42).split(xtr):
        a, b = xtr.iloc[tr], xtr.iloc[ev]
        preds = {**predict_all(plain(a), ytr.iloc[tr], plain(b), fuel), **predict_v1(a, ytr.iloc[tr], b, fuel, beta)}
        for k, v in preds.items():
            if not k.startswith("_"):
                folds.setdefault(k, []).append(metrics(ytr.iloc[ev].to_numpy(), v))
    for k, s in folds.items():
        out["cv"][k] = {m: round(float(np.mean([f[m] for f in s])), 4) for m in s[0]}
        out["cv"][k]["mae_kwh_sd"] = round(float(np.std([f["mae_kwh"] for f in s])), 1)
    preds = {**predict_all(plain(xtr), ytr, plain(xva), fuel), **predict_v1(xtr, ytr, xva, fuel, beta)}
    out["physics_scale_factor"] = round(float(np.exp(preds.pop("_scale"))), 3)
    out["v1_iterations"] = int(preds.pop("_iters"))
    out["v1_smearing_factor"] = round(preds.pop("_smear"), 4)
    preds.pop("_lgb_iters", None)
    out["val"] = {k: metrics(yva.to_numpy(), v) for k, v in preds.items()}
    if fuel == "gas":
        out["retrofit_check"] = retrofit_check(beta)
    return out


def main() -> None:
    results = {fuel: evaluate(fuel) for fuel in ("gas", "elec")}
    (RESULTS / "energy_v1.json").write_text(json.dumps(results, indent=2))
    for fuel, r in results.items():
        print(f"\n== {fuel} (physics scale factor {r['physics_scale_factor']}, elasticity {r['physics_elasticity']}, "
              f"V1 iters {r['v1_iterations']})")
        rows = [{"model": k, "cv_mae": r["cv"][k]["mae_kwh"], "cv_sd": r["cv"][k]["mae_kwh_sd"],
                 "cv_med_ape": r["cv"][k]["median_ape_pct"], "cv_r2_log": r["cv"][k]["r2_log"],
                 "val_mae": r["val"][k]["mae_kwh"], "val_r2_log": r["val"][k]["r2_log"]} for k in r["cv"]]
        print(pd.DataFrame(rows).to_string(index=False))
        if "retrofit_check" in r:
            print(json.dumps(r["retrofit_check"], indent=1))


if __name__ == "__main__":
    main()
