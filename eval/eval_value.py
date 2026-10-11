"""The single scoring of the value model on the test split (ARCHITECTURE 3.8; deliverable G: value interval coverage).

    uv run python eval/eval_value.py      (make eval-value, after make train-value)

Scores the SHIPPED value artifacts (fitted on train-split customers, bands calibrated on val) on real matched
resales of test-split customers, never used before. Baselines are refitted on train. Writes
eval/results/value_test.json: level accuracy + 95% bootstrap CIs, p10-p90 coverage, the "this home" check on
repeat sales, errors and coverage by local authority / type / rating band, and the rating premium re-estimated
on the test sales alone as a stability check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.models.value_model import (  # noqa: E402
    _load, band_premia, baselines, features, fe_ols, hedonic_design, training_sales, vmetrics)

RESULTS = ROOT / "eval" / "results"


def main() -> None:
    rng = np.random.default_rng(42)
    meta = _load()
    s = training_sales()
    tr, te = s[s.split == "train"], s[s.split == "test"].reset_index(drop=True)
    x, y = features(te), te.log_value_now.to_numpy()
    b, q = meta["boosters"], meta["conformal_q"]
    p50 = b["level"].predict(x)
    lo, hi = b["q10"].predict(x) - q, b["q90"].predict(x) + q
    preds = {"value_model": p50, **baselines(tr, te)}
    out = {"split": "test-split customers (data/processed/real/splits.parquet); scored once",
           "valuation_month": meta["valuation_month"], "n_test_sales": len(te), "models": {}}
    for k, p in preds.items():
        out["models"][k] = vmetrics(y, p)
    boot = [vmetrics(y[i], p50[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(500))]
    out["models"]["value_model"]["median_ape_pct_95ci"] = [round(float(np.percentile([m["median_ape_pct"] for m in boot], q)), 2) for q in (2.5, 97.5)]
    out["models"]["value_model"]["r2_log_95ci"] = [round(float(np.percentile([m["r2_log"] for m in boot], q)), 4) for q in (2.5, 97.5)]
    out["band"] = {"target": 0.8, "test_coverage": round(float(((y >= lo) & (y <= hi)).mean()), 4),
                   "below_p10": round(float((y < lo).mean()), 4), "above_p90": round(float((y > hi).mean()), 4),
                   "median_width_pct_of_p50": round(float(np.median((np.exp(hi) - np.exp(lo)) / np.exp(p50))) * 100, 1),
                   "val_coverage": meta["val_coverage"]}
    hp = te.prior_log_value_now.notna().to_numpy()
    w = meta["blend_weight_last_sale"]
    prior = te.prior_log_value_now.to_numpy()[hp]
    out["repeat_sales"] = {"n": int(hp.sum()), "indexed_last_sale": vmetrics(y[hp], prior),
                           "value_model": vmetrics(y[hp], p50[hp]),
                           "blend": vmetrics(y[hp], w * prior + (1 - w) * p50[hp]), "blend_weight_last_sale": w}
    sub = {}
    for col in ("local_authority", "property_type", "energy_rating"):
        rows = {}
        for level, idx in pd.Series(np.arange(len(y))).groupby(te[col].astype(str).to_numpy()).groups.items():
            i = np.asarray(idx)
            if len(i) >= 50:
                m = vmetrics(y[i], p50[i])
                rows[level] = {"n": int(len(i)), "median_ape_pct": m["median_ape_pct"], "within_20pct": m["within_20pct"],
                               "median_bias_pct": round(float(np.median(np.exp(p50[i] - y[i]) - 1)) * 100, 2),
                               "band_coverage": round(float(((y[i] >= lo[i]) & (y[i] <= hi[i])).mean()), 4)}
        sub[col] = rows
    out["subgroups"] = sub
    sec = te.postcode_sector.notna()
    beta, cov, r2 = fe_ols(hedonic_design(te[sec]).fillna(0), te.log_value_now[sec], te.postcode_sector[sec])
    out["premia_refit_on_test_only"] = band_premia(beta, cov)
    out["premia_shipped_train"] = meta["hedonic"]["premia"]
    (RESULTS / "value_test.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n_test_sales", "models", "band", "repeat_sales")}, indent=1))
    print("premia (test refit vs shipped):", {k: (v["pct"], out["premia_shipped_train"][k]["pct"]) for k, v in out["premia_refit_on_test_only"].items()})
    for col, rows in sub.items():
        print(col, {k: (v["n"], v["median_ape_pct"], v["band_coverage"]) for k, v in rows.items()})


if __name__ == "__main__":
    main()
