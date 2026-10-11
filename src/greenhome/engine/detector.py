"""'Worse than similar homes' detector (ARCHITECTURE 3.4).

A home is flagged for a fuel when its billed annual use is above the p90 of V1's conformalised band for a home
with the same description (mode 2: the model never sees the home's own use). The band is calibrated so that 10%
of similar homes use more than p90, which gives the flag a stated false-alarm rate, held on the NEED test split
in every region, type, age and rating group (eval/eval_detector.py). A big house is compared with similar big
houses, so it isn't flagged for its size.

Unavailable (never a silent "fine") when the fuel has no bill, the model can't place the home, the trained
models aren't installed, or the fuel isn't modelled (gas in a home not heated by gas: NEED has no gas readings
for such homes).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from greenhome.models.energy import ModelUnavailable, load_model
from greenhome.schema import Property

FUELS = {"elec": "electricity", "gas": "gas"}


def similar_homes_check(p: Property) -> dict:
    """{"status", "fuels": {fuel: {...}}, "flagged": [fuel, ...], "method"}; per fuel: billed kWh, the similar-home
    median (p50) and p90, the ratio to the median, and whether it is above p90."""
    out = {"status": "unavailable", "fuels": {}, "flagged": [],
           "method": "Billed use vs the 80% band of V1 predictions for homes with the same description "
                     "(type, age, size, rating, region, heating); flagged above the 90th percentile"}
    try:
        model = load_model("greybox")
    except ModelUnavailable:
        out["reason"] = "energy model not installed"
        return out
    r = model.predict_frame(pd.DataFrame([p.model_dump()])).iloc[0]
    for fuel in FUELS:
        billed = getattr(p, f"annual_{fuel}_kwh")
        p90 = r.get(f"{fuel}_p90", np.nan)
        if billed is None or not np.isfinite(p90):
            out["fuels"][fuel] = {"status": "unavailable",
                                  "reason": "no bill" if billed is None else "not modelled for this home"}
            continue
        p50 = float(r[f"{fuel}_kwh"]) / model.smear[fuel]           # the typical (median) similar home
        above = bool(billed > p90)
        out["fuels"][fuel] = {"status": "checked", "billed_kwh": round(float(billed), 1),
                              "similar_median_kwh": round(p50, 1), "similar_p90_kwh": round(float(p90), 1),
                              "ratio_to_similar_median": round(float(billed) / p50, 2), "above_p90": above}
        if above:
            out["flagged"].append(fuel)
    if any(v["status"] == "checked" for v in out["fuels"].values()):
        out["status"] = "checked"
    return out


def key_issue(check: dict) -> list[str]:
    """Plain-language key issues for flagged fuels (numbers come from the check, for the LLM to cite)."""
    issues = []
    for fuel in check.get("flagged", []):
        f = check["fuels"][fuel]
        issues.append(f"Uses more {FUELS[fuel]} than 9 in 10 similar homes ({f['billed_kwh']:,.0f} kWh/yr vs a "
                      f"typical {f['similar_median_kwh']:,.0f} kWh/yr, {f['ratio_to_similar_median']:.1f}x)")
    return issues
