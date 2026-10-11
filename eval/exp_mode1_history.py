"""Experiment (post-test, CV + val only): does mode 1 improve with more than one year of the home's own history?

    uv run python eval/exp_mode1_history.py      -> eval/results/exp_mode1_history.json

Mode 1 (models/energy_bills.py) predicts 2023 use from the description plus 2022 use, as a GBDT on log(2023 / 2022)
with an L1 objective. Here the same change model also sees 2021 (deployable: customer bills cover 24 months) and
2021 + 2020 (exploratory upper bound; three years of bills are not available for EPC homes today). Each variant
is anchored on the latest prior year and compared with no-model references built from the same history:
last year; last year x median trend; mean of the last two (three) years x median trend.

Rows are the same in every column: homes without a valid 2022 reading use the description-only V1 everywhere;
homes with 2022 but no valid older year use the one-year model inside the multi-year variants. The NEED test split
has already been scored once and is not touched here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.models.energy_bills import within  # noqa: E402
from greenhome.models.energy_greybox import elasticity, fit_v1, v1_log_pred, with_physics  # noqa: E402
from greenhome.models.energy_ml import LGB_PARAMS, metrics  # noqa: E402
from greenhome.models.need_data import dataset, load_need  # noqa: E402

RESULTS = ROOT / "eval" / "results"
YEAR = 2023


def history(d: pd.DataFrame, fuel: str, index: pd.Index) -> pd.DataFrame:
    cons, flag = ("Gcons", "GasValFlag") if fuel == "gas" else ("Econs", "ElecValFlag")
    out = {f"log_p{k}": np.log(d[f"{cons}{YEAR - k}"].where(d[f"{flag}{YEAR - k}"] == "V")) for k in (2, 3)}
    return pd.DataFrame(out).loc[index]


def change_model(x: pd.DataFrame, y: pd.Series, cols: list[str], es: np.ndarray):
    m = x.log_prior.notna().to_numpy().copy()
    for c in cols:
        m &= x[c].notna().to_numpy()
    xm, ym = x[m], (y - x.log_prior)[m]
    model = lgb.LGBMRegressor(**{**LGB_PARAMS, "objective": "l1"})
    model.fit(xm[~es[m]], ym[~es[m]], eval_X=(xm[es[m]],), eval_y=(ym[es[m]],),
              callbacks=[lgb.early_stopping(100, verbose=False)])
    return model


def predict(xtr, ytr, xev, fuel, beta) -> dict[str, np.ndarray]:
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    hist = ["log_p2", "log_p3"]
    desc = lambda df: df.drop(columns=["log_prior", *hist])
    fallback = v1_log_pred(fit_v1(desc(xtr), ytr, fuel, beta), beta, desc(xev))
    p1 = xev.log_prior.to_numpy()
    has1 = ~np.isnan(p1)
    trend = float(np.nanmedian(ytr - xtr.log_prior))
    out = {"last_year": np.where(has1, p1, fallback), "last_year_x_trend": np.where(has1, p1 + trend, fallback)}

    # The shipped one-year change model (same features and settings as energy_bills).
    one = change_model(xtr.drop(columns=hist), ytr, [], es)
    pred1 = np.where(has1, p1 + one.predict(xev.drop(columns=hist)), fallback)
    out["change_1yr_shipped"] = pred1
    for name, cols in (("2yr", ["log_p2"]), ("3yr", ["log_p2", "log_p3"])):
        keep = [c for c in hist if c not in cols]
        xtr_k, xev_k = xtr.drop(columns=keep), xev.drop(columns=keep)
        has = has1 & xev[cols].notna().all(axis=1).to_numpy()
        # No-model reference: mean of the available years (log), x the median trend from that mean on train.
        mean_tr = xtr[["log_prior", *cols]].mean(axis=1, skipna=False)
        mean_ev = xev[["log_prior", *cols]].mean(axis=1, skipna=False).to_numpy()
        t = float(np.nanmedian(ytr - mean_tr))
        out[f"mean_{name}_x_trend"] = np.where(has, mean_ev + t, pred1)
        m = change_model(xtr_k, ytr, cols, es)
        out[f"change_{name}"] = np.where(has, p1 + m.predict(xev_k), pred1)
    return out


def evaluate(fuel: str, d: pd.DataFrame) -> dict:
    x, y, split = dataset(fuel, d=d, own_bills=True)
    x = with_physics(x, fuel).join(history(d, fuel, x.index))
    xtr, ytr, xva, yva = x[split == "train"], y[split == "train"], x[split == "val"], y[split == "val"]
    beta = elasticity(fuel)
    folds: dict[str, list] = {}
    for tr, ev in KFold(5, shuffle=True, random_state=42).split(xtr):
        yv = ytr.iloc[ev].to_numpy()
        for k, v in predict(xtr.iloc[tr], ytr.iloc[tr], xtr.iloc[ev], fuel, beta).items():
            folds.setdefault(k, []).append({**metrics(yv, v), **within(yv, v)})
    has1 = xva.log_prior.notna()
    out = {"n_train": len(xtr), "n_val": len(xva),
           "val_share_with": {"1yr": round(float(has1.mean()), 4),
                              "2yr": round(float((has1 & xva.log_p2.notna()).mean()), 4),
                              "3yr": round(float((has1 & xva.log_p2.notna() & xva.log_p3.notna()).mean()), 4)},
           "cv": {}, "val": {}, "val_only_homes_with_3yr": {}}
    for k, s in folds.items():
        out["cv"][k] = {m: round(float(np.mean([f[m] for f in s])), 4) for m in s[0]}
        out["cv"][k]["mae_kwh_sd"] = round(float(np.std([f["mae_kwh"] for f in s])), 1)
    preds = predict(xtr, ytr, xva, fuel, beta)
    full = (has1 & xva.log_p2.notna() & xva.log_p3.notna()).to_numpy()
    for k, v in preds.items():
        out["val"][k] = {**metrics(yva.to_numpy(), v), **within(yva.to_numpy(), v)}
        out["val_only_homes_with_3yr"][k] = metrics(yva.to_numpy()[full], v[full])
    # Paired bootstrap on val: change in MAE (kWh) of each multi-year model vs the shipped one-year model.
    rng = np.random.default_rng(0)
    a = np.exp(yva.to_numpy())
    err = {k: np.abs(np.exp(v) - a) for k, v in preds.items()}
    idx = rng.integers(0, len(a), (2000, len(a)))
    out["val_mae_change_vs_shipped_pct"] = {}
    for k in ("change_2yr", "change_3yr", "mean_2yr_x_trend", "last_year"):
        r = err[k][idx].mean(1) / err["change_1yr_shipped"][idx].mean(1) - 1
        out["val_mae_change_vs_shipped_pct"][k] = {
            "mean": round(float(100 * (err[k].mean() / err["change_1yr_shipped"].mean() - 1)), 2),
            "ci95": [round(float(100 * v), 2) for v in np.percentile(r, [2.5, 97.5])]}
    return out


def main() -> None:
    d = load_need()
    results = {"note": "post-test experiment: train 5-fold CV and val only; test untouched",
               **{fuel: evaluate(fuel, d) for fuel in ("gas", "elec")}}
    (RESULTS / "exp_mode1_history.json").write_text(json.dumps(results, indent=2))
    for fuel in ("gas", "elec"):
        r = results[fuel]
        print(f"\n== {fuel}  val share with history: {r['val_share_with']}")
        rows = [{"model": k, "cv_mae": r["cv"][k]["mae_kwh"], "cv_sd": r["cv"][k]["mae_kwh_sd"],
                 "cv_mdape": r["cv"][k]["median_ape_pct"], "val_mae": r["val"][k]["mae_kwh"],
                 "val_mdape": r["val"][k]["median_ape_pct"], "val_r2_log": r["val"][k]["r2_log"],
                 "val_w20": r["val"][k]["within_20pct"], "val3yr_mae": r["val_only_homes_with_3yr"][k]["mae_kwh"]}
                for k in r["cv"]]
        print(pd.DataFrame(rows).to_string(index=False))
        print("val MAE change vs shipped:", r["val_mae_change_vs_shipped_pct"])


if __name__ == "__main__":
    main()
