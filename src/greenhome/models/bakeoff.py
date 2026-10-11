"""Library bake-off and feature-interaction check for V1's residual model (ARCHITECTURE 3.5 tuning steps 5-6).

Run AFTER the single test scoring, so this is a post-test check on the train split only (5-fold CV, the same folds
as everywhere else); val and test are not used. Compared on the shipped V1 target, log(use) - beta x log(physics):
  lightgbm_tuned     the shipped settings (config/energy_params.yaml, 50 Optuna trials)
  catboost_tuned     CatBoost, Optuna TPE on depth / learning rate / L2 / bagging, same folds
  average            mean of the two (the "top-2 average" option)
  lgb_interactions   LightGBM + explicit type x floor-area and EPC x age features
The shipped model only changes if a contender is better beyond fold noise; then the test score would no longer
be untouched, which EVALUATION.md has to say.

    uv run python -m greenhome.models.bakeoff --trials 20   -> eval/results/energy_bakeoff.json
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
import optuna
import pandas as pd
from sklearn.model_selection import KFold

from greenhome.models.energy_greybox import elasticity, fit_v1, with_physics
from greenhome.models.energy_ml import RESULTS, metrics
from greenhome.models.need_data import dataset

CAT_COLS = ["prop_type", "region", "conservatory"]


def folds(x):
    return list(KFold(5, shuffle=True, random_state=42).split(x))


def _cat_frame(x: pd.DataFrame) -> pd.DataFrame:
    x = x.drop(columns="log_physics").copy()
    for c in CAT_COLS:
        x[c] = x[c].astype(str)
    return x


def fit_catboost(xtr, rtr, params: dict):
    from catboost import CatBoostRegressor
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    f = _cat_frame(xtr)
    m = CatBoostRegressor(loss_function="Huber:delta=0.5", iterations=3000, od_type="Iter", od_wait=100,
                          random_seed=42, verbose=False, thread_count=-1, cat_features=CAT_COLS,
                          allow_writing_files=False, **params)
    # CatBoost's Huber starts from 0, not the mean, and the capped gradient never climbs to the target's level
    # (~5.4) before early stopping: centre the target and add the offset back.
    off = float(np.median(rtr))
    m.fit(f[~es], rtr[~es] - off, eval_set=(f[es], rtr[es] - off))
    return lambda xe: m.predict(_cat_frame(xe)) + off


def with_interactions(x: pd.DataFrame) -> pd.DataFrame:
    return x.assign(type_x_area=pd.Categorical(x.prop_type.astype(str) + "_" + x.floor_area_band.astype(str)),
                    epc_x_age=x.epc_band.fillna(0) * 10 + x.age_band)


def cv(x, y, fuel, beta, predict_fn) -> dict:
    maes, r2s = [], []
    for tr, ev in folds(x):
        p = beta * x.iloc[ev].log_physics.to_numpy() + predict_fn(x.iloc[tr], y.iloc[tr], x.iloc[ev])
        m = metrics(y.iloc[ev].to_numpy(), p)
        maes.append(m["mae_kwh"]), r2s.append(m["r2_log"])
    return {"cv_mae_kwh": round(float(np.mean(maes)), 1), "cv_mae_sd": round(float(np.std(maes)), 1),
            "cv_r2_log": round(float(np.mean(r2s)), 4), "fold_mae": [round(v, 1) for v in maes]}


def run(fuel: str, trials: int) -> dict:
    x, y, split = dataset(fuel)
    x = with_physics(x, fuel)
    x, y = x[split == "train"], y[split == "train"]
    beta = elasticity(fuel)
    resid = lambda xt, yt: yt - beta * xt.log_physics

    def lgb_fn(xt, yt, xe):
        m = fit_v1(xt, yt, fuel, beta)
        return m.predict(xe.drop(columns="log_physics"))

    def lgb_int_fn(xt, yt, xe):
        m = fit_v1(with_interactions(xt), yt, fuel, beta)
        return m.predict(with_interactions(xe).drop(columns="log_physics"))

    out = {"beta": beta, "n_train": len(x)}
    t = time.time()
    out["lightgbm_tuned"] = cv(x, y, fuel, beta, lgb_fn)
    out["lgb_interactions"] = cv(x, y, fuel, beta, lgb_int_fn)

    # CatBoost: Optuna on the first 3 folds (cost), then the best settings scored on all 5.
    fs = folds(x)[:3]

    def objective(trial):
        params = {"depth": trial.suggest_int("depth", 4, 8),
                  "learning_rate": trial.suggest_float("learning_rate", 0.02, 0.15, log=True),
                  "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 30.0, log=True),
                  "subsample": trial.suggest_float("subsample", 0.6, 1.0), "bootstrap_type": "Bernoulli"}
        maes = []
        for tr, ev in fs:
            xt, yt, xe = x.iloc[tr], y.iloc[tr], x.iloc[ev]
            p = beta * xe.log_physics.to_numpy() + fit_catboost(xt, resid(xt, yt), params)(xe)
            maes.append(metrics(y.iloc[ev].to_numpy(), p)["mae_kwh"])
        return float(np.mean(maes))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=trials)
    best = {**study.best_params, "bootstrap_type": "Bernoulli"}
    out["catboost_params"] = best
    cat_fn = lambda xt, yt, xe: fit_catboost(xt, resid(xt, yt), best)(xe)
    out["catboost_tuned"] = cv(x, y, fuel, beta, cat_fn)
    out["average"] = cv(x, y, fuel, beta, lambda xt, yt, xe: (lgb_fn(xt, yt, xe) + cat_fn(xt, yt, xe)) / 2)
    out["catboost_trials"] = trials
    out["seconds"] = round(time.time() - t)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--trials", type=int, default=20)
    trials = ap.parse_args().trials
    results = {"note": "post-test check: train split, 5-fold CV only", **{f: run(f, trials) for f in ("gas", "elec")}}
    (RESULTS / "energy_bakeoff.json").write_text(json.dumps(results, indent=2))
    for fuel in ("gas", "elec"):
        r = results[fuel]
        for k in ("lightgbm_tuned", "lgb_interactions", "catboost_tuned", "average"):
            print(f"{fuel:4s} {k:18s} CV MAE {r[k]['cv_mae_kwh']:8.1f} +/- {r[k]['cv_mae_sd']:5.1f}  R2 {r[k]['cv_r2_log']:.4f}")


if __name__ == "__main__":
    main()
