"""Hyperparameter tuning for the description-only energy model (ARCHITECTURE 3.5 tuning steps 2-4).

Optuna TPE search per fuel over LightGBM settings and the objective (L2 or Huber on log kWh, or Tweedie on raw
kWh), scored by 5-fold CV MAE in kWh on the NEED train split only, with the same folds as energy_ml and early
stopping inside each fold. Val and test are never seen. The best settings go to config/energy_params.yaml,
which energy_ml.fit_lgb reads, so V2 and V1's residual model both use them (like-for-like comparison).

    uv run python -m greenhome.models.tune --trials 50
"""

from __future__ import annotations

import argparse
import json

import lightgbm as lgb
import numpy as np
import optuna
import yaml
from sklearn.model_selection import KFold

from greenhome.models.energy_ml import LGB_PARAMS, RESULTS, monotone
from greenhome.models.need_data import ROOT, dataset

PARAMS_PATH = ROOT / "config" / "energy_params.yaml"


def cv_mae(params: dict, objective: str, x, y, fuel: str) -> tuple[float, float, int]:
    """5-fold CV MAE (kWh) of LightGBM with these settings; returns (mean, sd, median best iteration)."""
    maes, iters = [], []
    for tr, ev in KFold(5, shuffle=True, random_state=42).split(x):
        xtr, ytr, xev, yev = x.iloc[tr], y.iloc[tr], x.iloc[ev], y.iloc[ev]
        es = np.random.default_rng(0).random(len(xtr)) < 0.15
        raw = objective == "tweedie"
        t = np.exp(ytr) if raw else ytr
        m = lgb.LGBMRegressor(**{**LGB_PARAMS, **params, "objective": objective},
                              monotone_constraints=monotone(list(x.columns), fuel))
        m.fit(xtr[~es], t[~es], eval_X=(xtr[es],), eval_y=(t[es],), callbacks=[lgb.early_stopping(100, verbose=False)])
        p = m.predict(xev)
        pred_kwh = p if raw else np.exp(p)
        maes.append(float(np.abs(pred_kwh - np.exp(yev.to_numpy())).mean()))
        iters.append(m.best_iteration_)
    return float(np.mean(maes)), float(np.std(maes)), int(np.median(iters))


def tune(fuel: str, trials: int) -> dict:
    x, y, split = dataset(fuel)
    x, y = x[split == "train"], y[split == "train"]
    base_mae, base_sd, _ = cv_mae({}, "regression", x, y, fuel)

    def objective(trial: optuna.Trial) -> float:
        obj = trial.suggest_categorical("objective", ["regression", "huber", "tweedie"])
        params = {
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 4, 128, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 10, 400, log=True),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1.0),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-3, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-3, 30.0, log=True),
            "min_split_gain": trial.suggest_float("min_split_gain", 0.0, 0.5),
        }
        if obj == "huber":
            params["alpha"] = trial.suggest_float("huber_alpha", 0.1, 1.5)
        if obj == "tweedie":
            params["tweedie_variance_power"] = trial.suggest_float("tweedie_variance_power", 1.1, 1.9)
        mae, sd, it = cv_mae(params, obj, x, y, fuel)
        trial.set_user_attr("sd", sd)
        trial.set_user_attr("iterations", it)
        return mae

    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=trials)
    best = dict(study.best_params)
    obj = best.pop("objective")
    if "huber_alpha" in best:
        best["alpha"] = best.pop("huber_alpha")
    return {"untuned_cv_mae": round(base_mae, 1), "untuned_cv_sd": round(base_sd, 1),
            "tuned_cv_mae": round(study.best_value, 1), "tuned_cv_sd": round(study.best_trial.user_attrs["sd"], 1),
            "objective": obj, "params": {k: (round(v, 5) if isinstance(v, float) else v) for k, v in best.items()},
            "trials": trials, "best_trial": study.best_trial.number,
            "best_iterations": study.best_trial.user_attrs["iterations"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=50)
    trials = ap.parse_args().trials
    results = {fuel: tune(fuel, trials) for fuel in ("gas", "elec")}
    (RESULTS / "energy_tuning.json").write_text(json.dumps(results, indent=2))
    PARAMS_PATH.write_text(
        "# source: Optuna TPE search (models/tune.py), 5-fold CV MAE on the NEED train split only; see\n"
        "# eval/results/energy_tuning.json. Read by energy_ml.fit_lgb (V2 and V1's residual model).\n"
        + yaml.safe_dump({fuel: {"objective": r["objective"], **r["params"]} for fuel, r in results.items()}, sort_keys=False))
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
