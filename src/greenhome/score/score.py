"""Green Home Score: 7 monotonic sub-scores (0-100) combined with fixed weights.

Input: a dict with the Property field names from docs/ARCHITECTURE.md section 4.
Output: dict shaped like ScoreResult (total, classification, components).
Missing inputs get the configured default and a flag; they are never zeroed.
"""
from pathlib import Path
import yaml

_CFG = Path(__file__).resolve().parents[3] / "config" / "score_mappings.yaml"


def load_mappings(path=_CFG) -> dict:
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


def _consumption(p, m):
    cfg, d = m["consumption_intensity"], m["default_score"]
    elec, gas, area = p.get("annual_elec_kwh"), p.get("annual_gas_kwh"), p.get("floor_area_m2")
    if elec is None or gas is None or not area or area <= 0:
        return d, {"annual_elec_kwh": elec, "annual_gas_kwh": gas, "floor_area_m2": area}, ["missing_consumption_inputs"]
    intensity = (elec + gas) / area
    return _interp(intensity, cfg["points"]), {"kwh_per_m2": round(intensity, 1)}, []


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
