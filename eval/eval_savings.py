"""Savings-simulator checks against real installs and the catalogue (ARCHITECTURE 3.5 savings validity, 3.6).

    uv run python eval/eval_savings.py      -> eval/results/savings_checks.json

1. Solar: the simulator's rule (kWp x yield x self-consumption, capped at a share of import; config/simulator.yaml)
   vs the observed change in metered electricity for real NEED PV installs (data/need_labels.py), every split.
   NEED doesn't record kWp or location within the region, so the default kWp and yield are used.
2. Catalogue sanity bounds: on real homes, the share of model-based fabric savings (% of heating demand) that
   fall outside the catalogue's typical % range. Those ranges are placeholders (VERIFY), so outside != wrong;
   they are flagged, as ARCHITECTURE 3.5 asks.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.engine.feasibility import RETROFIT_CATALOGUE  # noqa: E402
from greenhome.engine.report import build_report  # noqa: E402
from greenhome.engine.savings import FABRIC, _cfg  # noqa: E402
from greenhome.schema import Property  # noqa: E402

RESULTS = ROOT / "eval" / "results"
LABELS = ROOT / "data" / "processed" / "need_retrofit_labels.parquet"
REAL = ROOT / "data" / "processed" / "real" / "properties_share.parquet"


def solar_check() -> dict:
    sol = _cfg("simulator.yaml")["solar"]
    lab = pd.read_parquet(LABELS)
    pv = lab[lab.measure == "solar_pv"]
    reduction = sol["default_kwp"] * sol["yield_kwh_per_kwp"]["default"] * sol["self_consumption"]
    predicted = -np.minimum(reduction, sol["max_share_of_import"] * pv.pre_kwh) / pv.pre_kwh
    out = {"fitted_on": "train (self_consumption, config/simulator.yaml); val and test are checks",
           "rule": f"{sol['default_kwp']} kWp x {sol['yield_kwh_per_kwp']['default']} kWh/kWp x "
                   f"{sol['self_consumption']} self-consumption, capped at {sol['max_share_of_import']:.0%} of import"}
    for split, g in pv.groupby("split"):
        p = predicted[g.index]
        out[split] = {"n": len(g), "observed_median_change": round(float(g.did_change.median()), 4),
                      "observed_median_did_kwh": round(float((g.did_change * g.pre_kwh).median()), 1),
                      "rule_median_change": round(float(p.median()), 4),
                      "observed_median_kwh": round(float((g.post_kwh - g.pre_kwh).median()), 1),
                      "rule_median_kwh": round(float((p * g.pre_kwh).median()), 1)}
    return out


def catalogue_check(n: int = 1000) -> dict:
    props = pd.read_parquet(REAL).sample(n, random_state=7)
    fields = Property.model_fields
    pct: dict[str, list[float]] = {r: [] for r in FABRIC}
    for _, row in props.iterrows():
        d = {k: (None if (isinstance(v, float) and v != v) or v is pd.NA else v) for k, v in row.items() if k in fields}
        r = build_report(Property(**d))
        if not r.energy_before or r.energy_before.heat_demand_kwh <= 0:
            continue
        heat = r.energy_before.heat_demand_kwh
        for rec in r.recommendations:
            if rec.retrofit in pct and "energy model (trained" in rec.reason:
                pct[rec.retrofit].append(100 * rec.marginal_saving.total_kwh / heat)
    out = {}
    for name, v in pct.items():
        if not v:
            continue
        v = np.array(v)
        lo, hi = RETROFIT_CATALOGUE[name]["saving_pct_heating"]
        out[name] = {"n": int(len(v)), "catalogue_range_pct": [lo, hi], "median_pct_of_heating": round(float(np.median(v)), 1),
                     "below_range": round(float((v < lo).mean()), 3), "above_range": round(float((v > hi).mean()), 3)}
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    out = {"solar_vs_need_pv_installs": solar_check(), "model_savings_vs_catalogue_ranges": catalogue_check()}
    (RESULTS / "savings_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
