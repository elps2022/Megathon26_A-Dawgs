"""Hard feasibility rules for retrofits (docs/ARCHITECTURE.md 3.9).

Input: a dict with Property field names, plus optional extras wall_type
("solid" / "cavity"), roof_usable (bool), listed_building (bool).
Output: feasible retrofits in stage order, plus infeasible ones with the rule that blocked them.
Unknown data is flagged, never guessed.
"""
from pathlib import Path
import yaml

_CFG = Path(__file__).resolve().parents[3] / "config" / "retrofit_catalogue.yaml"


def load_catalogue(path=_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _norm(v):
    return None if v is None else str(v).strip().lower().replace(" ", "_").replace("-", "_")


def _loft(p, t):
    if _norm(p.get("property_type")) == "flat":
        return "Flats have no private loft space.", []
    mm = p.get("loft_insulation_mm")
    if mm is None:
        return None, ["unknown_loft_insulation_mm"]
    if mm >= t["loft_full_mm"]:
        return f"Loft already insulated to {mm} mm (target {t['loft_full_mm']} mm).", []
    return None, []


def _cavity(p, t):
    if _norm(p.get("wall_insulation")) == "full":
        return "Walls are already fully insulated.", []
    wt = _norm(p.get("wall_type"))
    if wt is None:
        return "Wall construction unknown: a survey is needed before cavity wall insulation.", ["unknown_wall_type"]
    if wt == "solid":
        return "Solid walls cannot take cavity wall insulation.", []
    return None, []


def _solid(p, t):
    if _norm(p.get("wall_insulation")) == "full":
        return "Walls are already fully insulated.", []
    wt = _norm(p.get("wall_type"))
    if wt is None:
        return "Wall construction unknown: a survey is needed before solid wall insulation.", ["unknown_wall_type"]
    if wt == "cavity":
        return "Cavity walls: use cavity wall insulation instead.", []
    return None, []


def _glazing(p, t):
    g = _norm(p.get("glazing"))
    if g in ("double_older", "double_post_2002", "triple"):
        return "Windows are already double or triple glazed.", []
    return None, ([] if g else ["unknown_glazing"])


def _smart(p, t):
    if _norm(p.get("smart_controls")) in ("smart_thermostat", "smart_trvs"):
        return "A smart thermostat is already installed.", []
    return None, []


def _heat_pump(p, t):
    if _norm(p.get("heating")) == "heat_pump":
        return "Home already has a heat pump.", []
    mm = p.get("loft_insulation_mm")
    flags = ["insulate_first"] if (mm is None or mm < t["heat_pump_min_loft_mm"]) else []
    return None, flags


def _solar(p, t):
    if (p.get("solar_kwp") or 0) > 0:
        return "Home already has solar panels.", []
    if _norm(p.get("property_type")) == "flat":
        return "Flats usually have no private roof access.", []
    if p.get("roof_usable") is False:
        return "No usable roof space.", []
    if p.get("listed_building") is True:
        return "Listed building: solar panels are restricted.", []
    return None, []


_RULES = {
    "loft_insulation": _loft,
    "cavity_wall_insulation": _cavity,
    "solid_wall_insulation": _solid,
    "double_glazing": _glazing,
    "smart_thermostat": _smart,
    "heat_pump": _heat_pump,
    "solar_pv": _solar,
}


def assess(p: dict, catalogue: dict | None = None) -> dict:
    cat = catalogue or load_catalogue()
    t, stages = cat["thresholds"], cat["stages"]
    feasible, infeasible, flags = [], [], {}
    for name, entry in cat["retrofits"].items():
        rule, fl = _RULES[name](p, t)
        if fl:
            flags[name] = fl
        if rule:
            infeasible.append({"retrofit": name, "rule": rule})
        else:
            feasible.append(name)
    feasible.sort(key=lambda n: stages.index(cat["retrofits"][n]["stage"]))  # stable sort
    return {"feasible": feasible, "infeasible": infeasible, "flags": flags}
