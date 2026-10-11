"""Green Home Score: 7 monotonic sub-scores (0-100) combined with fixed weights.

Input: a dict with the Property field names from docs/ARCHITECTURE.md section 4.
Output: dict shaped like ScoreResult (total, classification, components).
Missing inputs get the configured default and a flag; they are never zeroed.
"""
import copy
from functools import lru_cache
from pathlib import Path
import yaml

_CFG = Path(__file__).resolve().parents[3] / "config" / "score_mappings.yaml"


def load_mappings(path=_CFG) -> dict:
    """Parsed config, cached per file and modification time (parsing it on every call dominated portfolio runs).
    Returns a copy, so a caller changing it can't affect later calls."""
    return copy.deepcopy(_parsed(str(path), Path(path).stat().st_mtime))


@lru_cache(maxsize=8)
def _parsed(path: str, mtime: float) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _norm(v) -> str:
    return str(v).strip().lower().replace(" ", "_").replace("-", "_")


def _interp(x: float, points) -> float:
    """Piecewise-linear through (x, y) points sorted by x, clipped at both ends."""
    if x <= points[0][0]:
        return float(points[0][1])
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        if x <= x1:
            return y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return float(points[-1][1])


def classify(total: float, m: dict) -> str:
    for band in m["bands"]:
        if total >= band["min"]:
            return band["name"]
    return m["bands"][-1]["name"]


def _lookup(table: dict, value, field: str, default: float):
    """Categorical lookup. Returns (score, inputs_used, flags)."""
    if value is None:
        return default, {field: None}, [f"missing_{field}"]
    key = _norm(value)
    if key not in table:
        return default, {field: value}, [f"unknown_{field}"]
    return float(table[key]), {field: value}, []


def _energy_efficiency(p, m):
    cfg, d = m["energy_efficiency"], m["default_score"]
    if p.get("sap_score") is not None:
        s = min(100.0, max(1.0, float(p["sap_score"])))
        return s, {"sap_score": p["sap_score"]}, []
    return _lookup(cfg["rating_midpoints"], p.get("energy_rating"), "energy_rating", d)


def _expected_purchased(p) -> float | None:
    """Expected purchased energy (kWh/yr) for a typical similar home, or None: the shipped energy model's median
    (V1, models/energy.similar_home_kwh), or the NEED log-linear table when the model isn't trained."""
    from greenhome.models.energy import similar_home_kwh
    e = similar_home_kwh(dict(p), statistic="median", score_reference=True)
    return None if e["elec"] is None or e["gas"] is None else e["elec"] + e["gas"]


def _consumption(p, m):
    """Purchased energy relative to a typical similar home (same type, size, age band, EPC band), so a home
    isn't penalised for being big or small. kWh/m2 is reported alongside but not scored."""
    cfg, d = m["consumption_intensity"], m["default_score"]
    elec, gas, area = p.get("annual_elec_kwh"), p.get("annual_gas_kwh"), p.get("floor_area_m2")
    if elec is None or gas is None or not area or area <= 0:
        return d, {"annual_elec_kwh": elec, "annual_gas_kwh": gas, "floor_area_m2": area}, ["missing_consumption_inputs"]
    used = {"kwh_per_m2": round((elec + gas) / area, 1), "actual_kwh": round(elec + gas, 1)}
    if "heating_fuel_unmetered" in (p.get("data_flags") or []):
        # oil / LPG / district heat isn't in the bills: the home's real consumption can't be judged
        return d, used, ["heating_fuel_unmetered"]
    expected = _expected_purchased(p)
    if not expected:
        return d, used, ["expected_use_unavailable"]
    ratio = (elec + gas) / expected
    used.update(expected_kwh_similar_homes=round(expected, 1), ratio_to_similar_homes=round(ratio, 3))
    return _interp(ratio, cfg["ratio_points"]), used, []


def _insulation(p, m):
    cfg, d = m["insulation"], m["default_score"]
    flags, used = [], {}
    loft_mm = p.get("loft_insulation_mm")
    if loft_mm is None:
        loft, flags, used["loft_insulation_mm"] = d, flags + ["missing_loft_insulation_mm"], None
    else:
        loft, used["loft_insulation_mm"] = _interp(float(loft_mm), cfg["loft_mm_points"]), loft_mm
    wall, wused, wflags = _lookup(cfg["wall"], p.get("wall_insulation"), "wall_insulation", d)
    used.update(wused)
    score = cfg["loft_weight"] * loft + cfg["wall_weight"] * wall
    return score, used, flags + wflags


def _heating(p, m):
    return _lookup(m["heating"]["scores"], p.get("heating"), "heating", m["default_score"])


def _glazing(p, m):
    return _lookup(m["glazing"]["scores"], p.get("glazing"), "glazing", m["default_score"])


def _renewables(p, m):
    cfg, d = m["renewables"], m["default_score"]
    kwp = p.get("solar_kwp")
    if kwp is None:
        return d, {"solar_kwp": None}, ["missing_solar_kwp"]
    if kwp <= 0:
        return 0.0, {"solar_kwp": kwp}, []          # genuinely no solar = 0
    demand = p.get("annual_elec_kwh")
    if demand is None or demand <= 0:
        return d, {"solar_kwp": kwp, "annual_elec_kwh": demand}, ["missing_elec_demand"]
    generation = kwp * cfg["solar_yield_kwh_per_kwp"]
    return min(100.0, 100.0 * generation / demand), {"solar_kwp": kwp, "generation_kwh": generation}, []


def _smart_controls(p, m):
    return _lookup(m["smart_controls"]["scores"], p.get("smart_controls"), "smart_controls", m["default_score"])


_COMPONENTS = {
    "energy_efficiency": _energy_efficiency,
    "consumption_intensity": _consumption,
    "insulation": _insulation,
    "heating": _heating,
    "glazing": _glazing,
    "renewables": _renewables,
    "smart_controls": _smart_controls,
}


def score_home(p: dict, mappings: dict | None = None) -> dict:
    m = mappings or load_mappings()
    components, total = [], 0.0
    for name, fn in _COMPONENTS.items():
        s, used, flags = fn(p, m)
        s = max(0.0, min(100.0, s))
        w = m["weights"][name]
        components.append({
            "component": name, "weight": w, "score": round(s, 1),
            "contribution": round(w * s, 2), "inputs_used": used, "flags": flags,
        })
        total += w * s
    total_int = int(round(total))
    return {"total": total_int, "classification": classify(total_int, m), "components": components}


def calculate_green_home_score(p, mappings: dict | None = None):
    """Calculate Green Home Score and return typed ScoreResult for Person B engine."""
    from greenhome.schema import ScoreResult, SubScore
    data = p.model_dump() if hasattr(p, "model_dump") else (dict(p) if isinstance(p, dict) else p.__dict__)
    res = score_home(data, mappings)
    components = [SubScore(**c) for c in res["components"]]
    return ScoreResult(
        total=res["total"],
        classification=res["classification"],
        components=components,
    )

