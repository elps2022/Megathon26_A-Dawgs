"""Mode 1, "this home": energy model that also sees the home's own previous-year use (ARCHITECTURE 3.5).

Mode 2 ("similar homes", energy_ml / energy_greybox) predicts from the description only; it is what the
"worse than similar homes" detector and homes without bills need. Mode 1 is what a lender has for existing
customers: last year's bill plus the home's description. Scored on real NEED use (2022 -> 2023), same splits.

Strong no-model references, which the models must beat (ml-data-prep checklist):
  last_year            predict 2023 = the home's own 2022 use
  last_year_x_trend    2022 use x the median 2023/2022 ratio on the training fold
Homes without a valid 2022 reading fall back to the description-only V1 prediction in every row, so all
rows score the same homes.

    uv run python -m greenhome.models.energy_bills      -> eval/results/energy_own_bills.json
"""

from __future__ import annotations

import json

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from greenhome.models.energy_greybox import with_physics
from greenhome.models.energy_ml import LGB_PARAMS, RESULTS, fit_lgb, metrics
from greenhome.models.need_data import dataset


def within(y_log: np.ndarray, p_log: np.ndarray) -> dict:
    a, b = np.exp(y_log), np.exp(p_log)
    ape = np.abs(b - a) / a
    return {"within_20pct": round(float((ape <= 0.2).mean()), 4), "within_30pct": round(float((ape <= 0.3).mean()), 4)}


def predict(xtr, ytr, xev, fuel) -> dict[str, np.ndarray]:
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    desc = lambda df: df.drop(columns=["log_prior"])
    # description-only V1 (fallback for homes without last year's bill)
    r = ytr - xtr.log_physics
    v1_desc = fit_lgb(desc(xtr)[~es], r[~es], desc(xtr)[es], r[es], fuel, log_ratio_target=True)
    fallback = xev.log_physics.to_numpy() + v1_desc.predict(desc(xev))
    prior = xev.log_prior.to_numpy()
    has = ~np.isnan(prior)
    trend = float(np.nanmedian(ytr.to_numpy() - xtr.log_prior.to_numpy()))
    plain = lambda df: df.drop(columns=["log_physics"])
    v2 = fit_lgb(plain(xtr)[~es], ytr[~es], plain(xtr)[es], ytr[es], fuel)
    v1 = fit_lgb(xtr[~es], r[~es], xtr[es], r[es], fuel, log_ratio_target=True)
    # Shipped mode-1 model: GBDT on the change from last year, log(y / prior), with an L1 (median) objective,
    # which is what minimises kWh error; homes without last year's bill use the description-only V1.
    mm = xtr.log_prior.notna().to_numpy()
    xm, ym, esm = xtr[mm], (ytr - xtr.log_prior)[mm], es[mm]
    chg = lgb.LGBMRegressor(**{**LGB_PARAMS, "objective": "l1"})
    chg.fit(xm[~esm], ym[~esm], eval_X=(xm[esm],), eval_y=(ym[esm],), callbacks=[lgb.early_stopping(100, verbose=False)])
    out = {"last_year": np.where(has, prior, fallback), "last_year_x_trend": np.where(has, prior + trend, fallback),
           "v2_with_own_bill": v2.predict(plain(xev)),
           "v1_with_own_bill": xev.log_physics.to_numpy() + v1.predict(xev),
           "change_model_l1": np.where(has, prior + chg.predict(xev), fallback),
           "description_only_v1": fallback}
    return out


def evaluate(fuel: str) -> dict:
    x, y, split = dataset(fuel, own_bills=True)
    x = with_physics(x, fuel)
    xtr, ytr, xva, yva = x[split == "train"], y[split == "train"], x[split == "val"], y[split == "val"]
    folds: dict[str, list] = {}
    for tr, ev in KFold(5, shuffle=True, random_state=42).split(xtr):
        for k, v in predict(xtr.iloc[tr], ytr.iloc[tr], xtr.iloc[ev], fuel).items():
            yv = ytr.iloc[ev].to_numpy()
            folds.setdefault(k, []).append({**metrics(yv, v), **within(yv, v)})
    out = {"n_train": len(xtr), "n_val": len(xva), "share_with_prior_bill_val": round(float(xva.log_prior.notna().mean()), 4),
           "cv": {}, "val": {}}
    for k, s in folds.items():
        out["cv"][k] = {m: round(float(np.mean([f[m] for f in s])), 4) for m in s[0]}
        out["cv"][k]["mae_kwh_sd"] = round(float(np.std([f["mae_kwh"] for f in s])), 1)
    for k, v in predict(xtr, ytr, xva, fuel).items():
        out["val"][k] = {**metrics(yva.to_numpy(), v), **within(yva.to_numpy(), v)}
    return out


def main() -> None:
    results = {fuel: evaluate(fuel) for fuel in ("gas", "elec")}
    (RESULTS / "energy_own_bills.json").write_text(json.dumps(results, indent=2))
    for fuel, r in results.items():
        print(f"\n== {fuel} (val homes with last year's bill: {r['share_with_prior_bill_val']:.1%})")
        rows = [{"model": k, "cv_mae": r["cv"][k]["mae_kwh"], "cv_sd": r["cv"][k]["mae_kwh_sd"], "cv_r2_log": r["cv"][k]["r2_log"],
                 "val_mae": r["val"][k]["mae_kwh"], "val_r2_log": r["val"][k]["r2_log"],
                 "val_within20": r["val"][k]["within_20pct"], "val_within30": r["val"][k]["within_30pct"]} for k in r["cv"]]
        print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
