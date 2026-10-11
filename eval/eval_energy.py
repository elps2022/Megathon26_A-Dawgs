"""The single scoring of the NEED test split (ARCHITECTURE 3.5 step 7). Run once, after every model choice.

    uv run python eval/eval_energy.py      (make eval-test)

Scores the SHIPPED models (src/greenhome/models/artifacts/, from `make train`: fitted on train, bands
conformalised on val) and the no-model baselines refitted on train, on real metered 2023 use of homes never
used for fitting, tuning or calibration. Writes:
  eval/results/energy_test.json      headline metrics + 95% bootstrap CIs, band coverage, mode 1 vs last
                                     year's bill, errors by subgroup (fairness audit), group-level R2,
                                     portfolio totals, top-25% precision / recall, test-split retrofit installs
  eval/results/test_predictions.csv  one row per test home and fuel: actual and every model's prediction
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.models.energy import BAND, GreyboxModel, MLModel  # noqa: E402
from greenhome.models.energy_bills import predict as mode1_predict  # noqa: E402
from greenhome.models.energy_greybox import physics_retrofit_ratios, with_physics  # noqa: E402
from greenhome.models.energy_ml import CellMedian, log_linear, metrics  # noqa: E402
from greenhome.models.need_data import dataset  # noqa: E402

RESULTS = ROOT / "eval" / "results"
SUBGROUPS = {"region": "region", "prop_type": "property type", "age_band": "age band (1 = pre-1930 ... 4 = 2000+)",
             "epc_band": "EPC band (1 = A/B ... 5 = F/G, blank = no EPC)", "floor_area_band": "floor area band",
             "imd_quintile": "deprivation quintile (1 = most deprived)"}
AGE = {1: "pre-1930", 2: "1930-1972", 3: "1973-1999", 4: "2000+"}
EPC = {1: "A/B", 2: "C", 3: "D", 4: "E", 5: "F/G"}
N_BOOT = 1000


def within(y, p) -> dict:
    ape = np.abs(np.exp(p) - np.exp(y)) / np.exp(y)
    return {"within_20pct": round(float((ape <= 0.2).mean()), 4), "within_30pct": round(float((ape <= 0.3).mean()), 4)}


def r2_raw(y, p) -> float:
    a, b = np.exp(y), np.exp(p)
    return round(float(1 - ((a - b) ** 2).sum() / ((a - a.mean()) ** 2).sum()), 4)


def boot_ci(y, p, rng) -> dict:
    """95% bootstrap CI over test homes (independent properties) for MAE and R2 (log)."""
    n, mae, r2 = len(y), [], []
    for _ in range(N_BOOT):
        i = rng.integers(0, n, n)
        m = metrics(y[i], p[i])
        mae.append(m["mae_kwh"]), r2.append(m["r2_log"])
    q = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    return {"mae_kwh_95ci": q(mae), "r2_log_95ci": q(r2)}


def top_quartile(y, p) -> dict:
    """Precision / recall for picking the top 25% of users (random pick: 0.25 / 0.25)."""
    true, pred = y >= np.quantile(y, 0.75), p >= np.quantile(p, 0.75)
    tp = (true & pred).sum()
    return {"precision": round(float(tp / pred.sum()), 4), "recall": round(float(tp / true.sum()), 4)}


def portfolio(y, p_mean, rng, sizes=(100, 1000), draws=500) -> dict:
    """Median absolute % error of the predicted total for random groups of homes (smeared, i.e. mean, predictions)."""
    a, b, out = np.exp(y), p_mean, {}
    for k in sizes:
        errs = [abs(b[i].sum() / a[i].sum() - 1) for i in (rng.choice(len(a), k, replace=False) for _ in range(draws))]
        out[f"homes_{k}"] = {"median_abs_pct_error": round(float(np.median(errs)) * 100, 2),
                             "p90_abs_pct_error": round(float(np.percentile(errs, 90)) * 100, 2)}
    return out


def group_r2(x, y, p, keys=("region", "prop_type", "age_band", "floor_area_band"), min_n=10) -> dict:
    """R2 of cell-average use: test homes grouped by description, actual vs predicted mean kWh per group.
    Labelled group level: single-home noise cancels inside each group."""
    df = x[list(keys)].astype(str).assign(a=np.exp(y), b=np.exp(p))
    g = df.groupby(list(keys), observed=True).agg(a=("a", "mean"), b=("b", "mean"), n=("a", "size"))
    g = g[g.n >= min_n]
    w = g.n / g.n.sum()
    abar = (w * g.a).sum()
    r2 = 1 - (w * (g.a - g.b) ** 2).sum() / (w * (g.a - abar) ** 2).sum()
    return {"groups": int(len(g)), "homes": int(g.n.sum()), "min_homes_per_group": min_n, "r2_weighted": round(float(r2), 4)}


def subgroup_table(x, y, p, lo, hi) -> dict:
    out = {}
    for col, label in SUBGROUPS.items():
        v = x[col]
        v = v.map(AGE) if col == "age_band" else v.map(EPC).fillna("no EPC") if col == "epc_band" else v
        rows = {}
        for level, idx in pd.Series(np.arange(len(y))).groupby(v.astype(str).to_numpy()).groups.items():
            i = np.asarray(idx)
            if len(i) < 30:
                continue
            ya, pa = np.exp(y[i]), np.exp(p[i])
            rows[level] = {"n": int(len(i)), "mae_kwh": round(float(np.abs(pa - ya).mean()), 1),
                           "median_ape_pct": round(float(np.median(np.abs(pa - ya) / ya) * 100), 2),
                           "median_bias_pct": round(float(np.median(pa / ya - 1) * 100), 2),
                           "band_coverage": round(float(((y[i] >= lo[i]) & (y[i] <= hi[i])).mean()), 4)}
        out[col] = {"label": label, "levels": rows}
    return out


def evaluate(fuel: str, v1: GreyboxModel, v2: MLModel, rng) -> tuple[dict, pd.DataFrame]:
    x, y, split = dataset(fuel, own_bills=True)
    x = with_physics(x, fuel)
    tr, te = (split == "train").to_numpy(), (split == "test").to_numpy()
    xd = x.drop(columns="log_prior")                      # description only (mode 2)
    xtr, ytr, xte, yte = xd[tr], y[tr], xd[te], y[te]
    yt = yte.to_numpy()

    # Shipped V1 / V2, exactly as deployed (NEED physics convention, same features as training).
    beta = v1.meta[fuel]["physics_elasticity"]
    f1 = xte[v1.meta[fuel]["features"]]
    p_v1 = beta * xte.log_physics.to_numpy() + v1.boosters[fuel].predict(f1)
    lo_q, hi_q = (beta * xte.log_physics.to_numpy() + m.predict(f1) for m in v1.quantiles[fuel])
    q = v1.meta[fuel]["band"]["conformal_q"]
    lo, hi = lo_q - q, hi_q + q
    raw2 = v2.boosters[fuel].predict(xte[v2.meta[fuel]["features"]])
    p_v2 = np.log(np.clip(raw2, 1e-6, None)) if v2.meta[fuel]["raw_target"] else raw2

    plain = xtr.drop(columns="log_physics")
    scale = float(np.median(ytr.to_numpy() - xtr.log_physics.to_numpy()))
    preds = {
        "global_median": np.full(len(yt), float(ytr.median())),
        "cell_median": CellMedian().fit(plain, ytr).predict(xte.drop(columns="log_physics")),
        "log_linear": log_linear().fit(plain.fillna({"imd_quintile": 0}), ytr).predict(
            xte.drop(columns="log_physics").fillna({"imd_quintile": 0})),
        "physics_scaled": xte.log_physics.to_numpy() + scale,
        "v2_ml": p_v2,
        "v1_greybox": p_v1,
    }
    # Mode 1 ("this home"): refit on train exactly as energy_bills does; homes without 2022 fall back to V1.
    m1 = mode1_predict(x[tr], ytr, x[te], fuel, beta)
    for k in ("last_year", "last_year_x_trend", "change_model_l1"):
        preds[f"mode1_{k}"] = m1[k]

    out = {"n_test": int(te.sum()), "n_train": int(tr.sum()),
           "share_with_prior_bill": round(float(x[te].log_prior.notna().mean()), 4), "models": {}}
    for k, p in preds.items():
        out["models"][k] = {**metrics(yt, p), **within(yt, p), "r2_raw": r2_raw(yt, p)}
        if k in ("v1_greybox", "v2_ml", "log_linear", "mode1_last_year", "mode1_change_model_l1"):
            out["models"][k].update(boot_ci(yt, p, rng))
    # Paired bootstrap: is mode 1 better than last year's bill on the same homes?
    a = np.exp(yt)
    e1, e0 = np.abs(np.exp(preds["mode1_change_model_l1"]) - a), np.abs(np.exp(preds["mode1_last_year"]) - a)
    diffs = [(e1[i].mean() / e0[i].mean() - 1) * 100 for i in (rng.integers(0, len(a), len(a)) for _ in range(N_BOOT))]
    out["mode1_vs_last_year_mae_change_pct"] = {"point": round(float((e1.mean() / e0.mean() - 1) * 100), 2),
                                                "95ci": [round(float(np.percentile(diffs, 2.5)), 2),
                                                         round(float(np.percentile(diffs, 97.5)), 2)]}
    out["band"] = {"target_coverage": BAND[1] - BAND[0], "test_coverage": round(float(((yt >= lo) & (yt <= hi)).mean()), 4),
                   "below_p10": round(float((yt < lo).mean()), 4), "above_p90": round(float((yt > hi).mean()), 4),
                   "median_width_kwh": round(float(np.median(np.exp(hi) - np.exp(lo))), 1),
                   "median_width_pct_of_p50": round(float(np.median((np.exp(hi) - np.exp(lo)) / np.exp(p_v1))) * 100, 1),
                   "val_coverage": v1.meta[fuel]["band"]["val_coverage"]}
    smear = v1.meta[fuel]["smearing_factor"]
    out["portfolio_totals_v1"] = portfolio(yt, np.exp(p_v1) * smear, rng)
    out["group_level_r2_v1"] = group_r2(xte, yt, p_v1)
    if fuel == "elec":  # added after the first test run (exploratory): heating type drives electricity
        out["group_level_r2_v1_with_heating_exploratory"] = group_r2(
            xte, yt, p_v1, keys=("region", "prop_type", "age_band", "floor_area_band", "gas_heated"))
    out["top_quartile_v1"] = top_quartile(yt, p_v1)
    out["top_quartile_v2"] = top_quartile(yt, p_v2)
    out["subgroups_v1"] = subgroup_table(xte, yt, p_v1, lo, hi)

    df = pd.DataFrame({"need_row": x.index[te] if "need_row" not in x else x.need_row[te], "fuel": fuel,
                       "actual_kwh": np.round(a, 1),
                       "v1_p10": np.round(np.exp(lo), 1), "v1_p50": np.round(np.exp(p_v1), 1),
                       "v1_p90": np.round(np.exp(hi), 1), "v1_mean": np.round(np.exp(p_v1) * smear, 1),
                       **{f"{k}_kwh": np.round(np.exp(p), 1) for k, p in preds.items() if k != "v1_greybox"}})
    return out, df


def retrofit_test() -> dict:
    """Observed vs predicted median change for the TEST-split real installs (never seen before this run)."""
    beta = GreyboxModel().meta["gas"]["physics_elasticity"]
    out = {}
    for measure, d in physics_retrofit_ratios("test").items():
        out[measure] = {"n": len(d), "observed_median_change": round(float(np.median(d.observed_ratio) - 1), 4),
                        "physics_only_median_change": round(float(np.median(d.physics_ratio) - 1), 4),
                        "v1_median_change": round(float(np.median(d.physics_ratio ** beta) - 1), 4)}
    return out


def main() -> None:
    rng = np.random.default_rng(42)
    v1, v2 = GreyboxModel(), MLModel()
    results, frames = {"split": "NEED test (15%, fixed, data/splits/need_split.csv); scored once",
                       "models_scored": "shipped artifacts: V1 greybox_v1, V2 ml_v2 (fitted on train, bands on val)"}, []
    for fuel in ("gas", "elec"):
        results[fuel], df = evaluate(fuel, v1, v2, rng)
        frames.append(df)
    results["retrofit_installs_test"] = retrofit_test()
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "energy_test.json").write_text(json.dumps(results, indent=2))
    pd.concat(frames).to_csv(RESULTS / "test_predictions.csv", index=False)
    for fuel in ("gas", "elec"):
        r = results[fuel]
        print(f"\n== {fuel}: {r['n_test']:,} test homes")
        print(pd.DataFrame({k: {m: v[m] for m in ("mae_kwh", "median_ape_pct", "r2_log", "r2_raw", "within_20pct")}
                            for k, v in r["models"].items()}).T.to_string())
        print("band:", r["band"]); print("mode 1 vs last year:", r["mode1_vs_last_year_mae_change_pct"])
        print("group R2:", r["group_level_r2_v1"], "| portfolio:", r["portfolio_totals_v1"], "| top 25%:", r["top_quartile_v1"])
    print(json.dumps(results["retrofit_installs_test"], indent=1))


if __name__ == "__main__":
    main()
