"""Check: does weather-correcting each year of a home's bill history improve the mode 1 forecast?

    uv run python eval/eval_weather_history.py      -> eval/results/weather_history_check.json

Mode 1 is trained on NEED, whose gas is weather corrected, but customer bills are billed as metered. Since this
change, data/annualise.py rescales each history year (recent / prior 12 months) of the heating fuel to the typical
year with the home's own degree-day fit (data/degree_day_fit.year_weather_factors). NEED can't test this (already
corrected), so the reference is the synthetic generator's hidden truth: each home's typical-year gas under its
hidden occupancy and behaviour (data/generate.EnergyWorld.annual, labels only). Gas-heated synthetic homes without
a retrofit event and with both history years. Optimistic: synthetic bills are exactly linear in degree days, so
the correction works better here than on real bills. Nothing is fitted or tuned on this check.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.data.generate import AGE_LABELS, GAS_HEATING, EnergyWorld, load_config  # noqa: E402
from greenhome.data.degree_day_fit import year_weather_factors  # noqa: E402
from greenhome.models.energy import load_model  # noqa: E402

GEN = ROOT / "data" / "generated"
RESULTS = ROOT / "eval" / "results"


def main() -> None:
    props = pd.read_csv(GEN / "properties.csv")
    lat = pd.read_csv(GEN / "_hidden" / "latents.csv").set_index("property_id")
    monthly = pd.read_csv(GEN / "monthly_energy.csv")
    events = set(pd.read_csv(GEN / "retrofit_events.csv").property_id)
    syn, need, _, hdd, _ = load_config()
    world = EnergyWorld(syn, need, hdd)
    fac = year_weather_factors(props, monthly).pivot(index="property_id", columns="year", values="factor")
    ok = (props.heating.isin(GAS_HEATING) & ~props.property_id.isin(events) & props.property_id.isin(fac.index)
          & props.recent_gas_kwh.notna() & props.prior_gas_kwh.notna())
    p = props[ok].copy()
    f = fac.loc[p.property_id]
    # properties.csv now holds corrected history; recover the as-billed years for the comparison
    p["recent_billed"] = p.recent_gas_kwh.to_numpy() / f["recent"].to_numpy()
    p["prior_billed"] = p.prior_gas_kwh.to_numpy() / f["prior"].to_numpy()
    age = {v: k for k, v in AGE_LABELS.items()}
    truth = []
    for _, r in p.iterrows():
        L = lat.loc[r.property_id]
        h = dict(local_authority=r.local_authority, property_type=r.property_type, age_code=age[r.age_band],
                 floor_area_m2=float(L.true_floor_area_m2), heating=r.heating, wall_type=r.wall_type,
                 wall_insulation=r.wall_insulation, has_own_roof=bool(r.has_own_roof),
                 loft_insulation_mm=None if pd.isna(r.loft_insulation_mm) else int(r.loft_insulation_mm),
                 glazing=r.glazing, solar_kwp=float(r.solar_kwp), smart_controls=r.smart_controls, heat_mult=1.0,
                 scop=float(L.heat_pump_scop), true_band=L.true_band)
        truth.append(world.annual(h, {k: float(L[k]) for k in ("occupancy", "behaviour", "idio_gas", "idio_elec")})[1])
    t = np.array(truth)
    m = load_model("greybox")
    base = p.drop(columns=["recent_billed", "prior_billed"])
    billed = m.forecast_frame(base.assign(recent_gas_kwh=p.recent_billed.to_numpy(),
                                          prior_gas_kwh=p.prior_billed.to_numpy())).gas_kwh.to_numpy()
    corrected = m.forecast_frame(base).gas_kwh.to_numpy()
    stat = lambda x: {"mae_kwh": round(float(np.nanmean(np.abs(x - t))), 1),
                      "median_ape_pct": round(float(np.nanmedian(np.abs(x - t) / t) * 100), 2),
                      "median_ratio_to_truth": round(float(np.nanmedian(x / t)), 4)}
    out = {"homes": int(len(p)), "reference": "hidden true typical-year gas (synthetic generator)",
           "median_weather_factor": {"recent": round(float(f["recent"].median()), 4),
                                     "prior": round(float(f["prior"].median()), 4)},
           "inputs": {"recent_as_billed": stat(p.recent_billed.to_numpy()), "recent_corrected": stat(p.recent_gas_kwh.to_numpy()),
                      "annualised_24_months": stat(p.annual_gas_kwh.to_numpy())},
           "mode1_forecast": {"history_as_billed": stat(billed), "history_weather_corrected": stat(corrected)},
           "caveat": "optimistic: synthetic bills are exactly linear in degree days"}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "weather_history_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
