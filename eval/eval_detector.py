"""Detector check on the NEED test split (ARCHITECTURE 3.4): is "above p90 for similar homes" a fair flag?

    uv run python eval/eval_detector.py      -> eval/results/detector_check.json

Reads the already-scored test predictions (eval/results/test_predictions.csv: real metered use, V1 p90 from the
description only) and reports the flag rate overall and by region, type, age, EPC band and deprivation. By
construction about 10% of similar homes should be flagged in every group; a group flagged much more often would
be penalised for who it is rather than how it uses energy. Reporting only: nothing is fitted or tuned here.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.models.need_data import features, load_need  # noqa: E402

RESULTS = ROOT / "eval" / "results"
AGE = {1: "pre-1930", 2: "1930-1972", 3: "1973-1999", 4: "2000+"}
EPC = {1: "A/B", 2: "C", 3: "D", 4: "E", 5: "F/G"}


def main() -> None:
    t = pd.read_csv(RESULTS / "test_predictions.csv")
    f = features(load_need())
    t = t.join(f[["region", "prop_type", "age_band", "epc_band", "imd_quintile"]], on="need_row")
    t["flag"] = t.actual_kwh > t.v1_p90
    t["age_band"] = t.age_band.map(AGE)
    t["epc_band"] = t.epc_band.map(EPC).fillna("no EPC")
    out = {"rule": "billed use > V1 p90 for homes with the same description", "expected_rate": 0.10, "fuels": {}}
    for fuel, g in t.groupby("fuel"):
        groups = {}
        for col in ("region", "prop_type", "age_band", "epc_band", "imd_quintile"):
            r = g.groupby(g[col].astype(str)).flag.agg(["mean", "size"])
            groups[col] = {k: {"flag_rate": round(float(v["mean"]), 4), "n": int(v["size"])} for k, v in r.iterrows() if v["size"] >= 30}
        rates = [v["flag_rate"] for c in groups.values() for v in c.values()]
        out["fuels"][fuel] = {"n": len(g), "flag_rate": round(float(g.flag.mean()), 4),
                              "min_group_rate": min(rates), "max_group_rate": max(rates), "groups": groups}
    (RESULTS / "detector_check.json").write_text(json.dumps(out, indent=2))
    for fuel, r in out["fuels"].items():
        print(fuel, "flag rate", r["flag_rate"], "group range", r["min_group_rate"], "-", r["max_group_rate"])
        for col, g in r["groups"].items():
            print("  ", col, {k: v["flag_rate"] for k, v in g.items()})


if __name__ == "__main__":
    main()
