"""Recommendation ranking check (ARCHITECTURE 3.9): does the engine put the best-value retrofits first?

    uv run python eval/eval_recommend.py      -> eval/results/recommend_check.json

Reference labels come from the synthetic generator's hidden "true world" (data/generate.py EnergyWorld), not from
the engine: for each home and each retrofit that both the generator and the engine consider possible, the true
annual bill saving is the generator's typical-year change under the home's hidden occupancy, behaviour and
per-home retrofit effect (mean of 20 effect draws), priced at config/energy_prices.yaml. True value = that saving
per pound of mid indicative cost. The generator uses NEED observed effects and its own assumptions, while the engine
uses the grey-box model, so the check is not circular, but both share the NEED calibration and the price and cost
tables (labelled in the output).

Compared rankings over the shared retrofits of each home:
- engine:   the report's `rank` (suitability from payback and constraints, then payback, then carbon);
- baseline: "recommend everything feasible" in the fixed catalogue order (fabric, controls, heating, generation);
- cheapest: cheapest mid cost first (a strong no-model reference: costs differ far more than savings);
- random:   mean over 20 shuffles.
Metrics: NDCG@3 (gain = true value per pound, floored at 0), top-1 hit rate (the first pick is the truly best value),
and the same with carbon (true kg CO2e cut per pound) as a secondary target. Home-level bootstrap 95% CIs. Nothing
is fitted or tuned here; the suitability thresholds were set before this check ran. Uses data/generated/_hidden for
labels only, never as features.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.data.generate import AGE_LABELS, EnergyWorld, apply_retrofit, candidate_retrofits, load_config  # noqa: E402
from greenhome.engine.carbon import calculate_carbon_kg  # noqa: E402
from greenhome.engine.feasibility import RETROFIT_CATALOGUE  # noqa: E402
from greenhome.engine.portfolio import _clean, _warm_energy_cache  # noqa: E402
from greenhome.engine.report import build_report  # noqa: E402
from greenhome.engine.savings import _cfg  # noqa: E402
from greenhome.schema import Property  # noqa: E402

GEN = ROOT / "data" / "generated"
RESULTS = ROOT / "eval" / "results"
TO_ENGINE = {"glazing_upgrade": "double_glazing"}
N_HOMES, N_DRAWS, K = 2000, 20, 3


def ndcg(order: list[str], gain: dict[str, float], k: int = K) -> float | None:
    ideal = sorted(gain.values(), reverse=True)[:k]
    idcg = sum(g / math.log2(i + 2) for i, g in enumerate(ideal))
    if idcg <= 0:
        return None
    return sum(gain[r] / math.log2(i + 2) for i, r in enumerate(order[:k])) / idcg


def true_values(h: dict, lat: dict, world: EnergyWorld, syn: dict, need: dict, prices: dict, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    e0, g0 = world.annual(h, lat)
    out = {}
    for r in candidate_retrofits(h):
        d_e = d_g = 0.0
        for _ in range(N_DRAWS):
            e1, g1 = world.annual(apply_retrofit(rng, h, r, syn, need), lat)
            d_e, d_g = d_e + (e0 - e1) / N_DRAWS, d_g + (g0 - g1) / N_DRAWS
        name = TO_ENGINE.get(r, r)
        cost = sum(RETROFIT_CATALOGUE[name]["indicative_cost"]) / 2
        out[name] = {"gbp": (d_e * prices["electricity"] + d_g * prices["gas"]) / cost,
                     "carbon": calculate_carbon_kg(d_e, d_g) / cost}
    return out


def boot(x: np.ndarray, rng: np.random.Generator, n: int = 2000) -> list[float]:
    idx = rng.integers(0, len(x), (n, len(x)))
    return [round(float(v), 3) for v in np.percentile(x[idx].mean(1), [2.5, 97.5])]


def main() -> None:
    syn, need, _, hdd, _ = load_config()
    world = EnergyWorld(syn, need, hdd)
    prices = _cfg("energy_prices.yaml")["gbp_per_kwh"]
    props = pd.read_csv(GEN / "properties.csv")
    lat = pd.read_csv(GEN / "_hidden" / "latents.csv").set_index("property_id")
    events = set(pd.read_csv(GEN / "retrofit_events.csv").property_id)
    # Homes whose bills are all pre-retrofit and whose description is complete enough to rebuild the true home.
    ok = (~props.property_id.isin(events) & props.floor_area_m2.notna()
          & (props.loft_insulation_mm.notna() | ~props.has_own_roof.astype(bool)))
    props = props[ok].sample(min(N_HOMES, int(ok.sum())), random_state=0)
    _warm_energy_cache(props)

    age_code = {v: k for k, v in AGE_LABELS.items()}
    rows = []
    shuffle = np.random.default_rng(1)
    for i, (_, row) in enumerate(props.iterrows()):
        L = lat.loc[row.property_id]
        h = {"local_authority": row.local_authority, "property_type": row.property_type,
             "age_code": age_code[row.age_band], "floor_area_m2": float(L.true_floor_area_m2),
             "heating": row.heating, "wall_type": row.wall_type, "wall_insulation": row.wall_insulation,
             "has_own_roof": bool(row.has_own_roof),
             "loft_insulation_mm": None if pd.isna(row.loft_insulation_mm) else int(row.loft_insulation_mm),
             "glazing": row.glazing, "solar_kwp": float(row.solar_kwp), "smart_controls": row.smart_controls,
             "heat_mult": 1.0, "scop": float(L.heat_pump_scop), "true_band": L.true_band}
        truth = true_values(h, {k: float(L[k]) for k in ("occupancy", "behaviour", "idio_gas", "idio_elec")},
                            world, syn, need, prices, seed=i)
        try:
            report = build_report(Property(**_clean(row.to_dict())))
        except Exception:
            continue
        engine = [r.retrofit for r in sorted(report.recommendations, key=lambda r: r.rank or 99) if r.retrofit in truth]
        if len(engine) < 2:
            continue
        shared = {r: truth[r] for r in engine}
        baseline = sorted(engine, key=lambda r: RETROFIT_CATALOGUE[r]["priority"])
        cheapest = sorted(engine, key=lambda r: sum(RETROFIT_CATALOGUE[r]["indicative_cost"]))
        rec = {"n": len(engine)}
        for target in ("gbp", "carbon"):
            gain = {r: max(v[target], 0.0) for r, v in shared.items()}
            best = max(shared, key=lambda r: shared[r][target])
            rnd = [ndcg(list(shuffle.permutation(engine)), gain) for _ in range(20)]
            rec |= {f"{target}_ndcg_engine": ndcg(engine, gain), f"{target}_ndcg_baseline": ndcg(baseline, gain),
                    f"{target}_ndcg_cheapest": ndcg(cheapest, gain),
                    f"{target}_ndcg_random": None if rnd[0] is None else float(np.mean(rnd)),
                    f"{target}_top1_engine": engine[0] == best, f"{target}_top1_baseline": baseline[0] == best,
                    f"{target}_top1_cheapest": cheapest[0] == best,
                    f"{target}_top1_random": 1 / len(engine)}
        rows.append(rec)

    df = pd.DataFrame(rows)
    rng = np.random.default_rng(0)
    out = {"what": "Ranking of feasible retrofits vs true value from the synthetic generator's hidden world",
           "homes": len(df), "shared_retrofits_per_home_mean": round(float(df.n.mean()), 2),
           "reference": "true annual bill saving (or carbon cut) per pound of mid indicative cost, hidden latents, "
                        f"mean of {N_DRAWS} per-home effect draws; prices config/energy_prices.yaml",
           "shared_assumptions": "generator and engine share NEED calibration, energy prices and catalogue costs",
           "targets": {}}
    for target in ("gbp", "carbon"):
        t = {}
        for m in ("ndcg", "top1"):
            for who in ("engine", "baseline", "cheapest", "random"):
                x = df[f"{target}_{m}_{who}"].dropna().astype(float).to_numpy()
                t[f"{m}@{K if m == 'ndcg' else 1}_{who}"] = {"mean": round(float(x.mean()), 3), "ci95": boot(x, rng)}
            d = (df[f"{target}_{m}_engine"].astype(float) - df[f"{target}_{m}_baseline"].astype(float)).dropna().to_numpy()
            t[f"{m}_engine_minus_baseline"] = {"mean": round(float(d.mean()), 3), "ci95": boot(d, rng)}
            d = (df[f"{target}_{m}_engine"].astype(float) - df[f"{target}_{m}_cheapest"].astype(float)).dropna().to_numpy()
            t[f"{m}_engine_minus_cheapest"] = {"mean": round(float(d.mean()), 3), "ci95": boot(d, rng)}
        out["targets"]["bill_saving_per_gbp" if target == "gbp" else "carbon_per_gbp"] = t
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "recommend_check.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
