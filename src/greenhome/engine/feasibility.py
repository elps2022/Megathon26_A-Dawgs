"""Hard feasibility rules for retrofits (docs/ARCHITECTURE.md 3.9).

Input: a dict with Property field names, plus optional extras wall_type
("solid" / "cavity"), roof_usable (bool), listed_building (bool).
Output: feasible retrofits in stage order, plus infeasible ones with the rule that blocked them.
Unknown data is flagged, never guessed.
"""
from pathlib import Path
import yaml

from greenhome.schema import Infeasible, Property

_CFG = Path(__file__).resolve().parents[3] / "config" / "retrofit_catalogue.yaml"
DECLINED_RULE = "Previously declined by customer preference."  # a preference, not a hard rule


def load_catalogue(path=_CFG) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _norm(v):
    if v is None:
        return None
    n = str(v).strip().lower().replace(" ", "_").replace("-", "_")
    return None if n in ("", "unknown", "nan") else n  # "unknown" placeholders count as missing data


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
    if g in ("double_old", "double_new", "triple"):
        return "Windows are already double or triple glazed.", []
    return None, ([] if g else ["unknown_glazing"])


def _smart(p, t):
    if _norm(p.get("smart_controls")) in ("smart_thermostat", "smart_trvs"):
        return "A smart thermostat is already installed.", []
    return None, []


def _heat_pump(p, t):
    heating = _norm(p.get("heating"))
    if heating == "heat_pump":
        return "Home already has a heat pump.", []
    if heating is None:
        return "Current heating system unknown: a survey is needed before a heat pump.", ["unknown_heating"]
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


# --- Person B Engine Integration & Memory Support ---

def _engine_catalogue(cat: dict) -> dict:
    """Name, cost range and stage priority per retrofit, read from config/retrofit_catalogue.yaml (with sources)."""
    priority = {stage: i + 1 for i, stage in enumerate(cat["stages"])}
    return {
        key: {
            "name": r.get("name", key.replace("_", " ").title()),
            "indicative_cost": tuple(float(c) for c in r["cost_gbp"]),
            "priority": priority[r["stage"]],
            "saving_pct_heating": r.get("typical_saving_pct_heating"),
            "source": r["source"],
        }
        for key, r in cat["retrofits"].items()
    }


RETROFIT_CATALOGUE = _engine_catalogue(load_catalogue())


def check_feasibility(
    p: Property,
    memory_constraints: list[str] | None = None,
) -> tuple[list[str], list[Infeasible]]:
    """Determine feasible and infeasible retrofits for a Property object, respecting customer memory."""
    p_dict = p.model_dump() if hasattr(p, "model_dump") else (dict(p) if isinstance(p, dict) else p.__dict__)
    
    # Assess via official catalogue rules
    res = assess(p_dict)
    feasible = list(res["feasible"])
    infeasible = [Infeasible(retrofit=item["retrofit"], rule=item["rule"]) for item in res["infeasible"]]
    constraints = set(memory_constraints or [])

    # Filter by tenure or budget constraints from customer memory
    if "tenant" in constraints:
        for r in ("solid_wall_insulation", "heat_pump"):
            if r in feasible:
                feasible.remove(r)
                infeasible.append(Infeasible(retrofit=r, rule="Structural works restricted by tenancy agreement."))

    # Filter out retrofits explicitly declined by customer
    final_feasible = []
    for r in feasible:
        if f"reject_{r}" in constraints:
            infeasible.append(Infeasible(retrofit=r, rule=DECLINED_RULE))
        else:
            final_feasible.append(r)

    # Sort feasible by priority order (Fabric -> Controls -> Heating -> Generation)
    final_feasible.sort(key=lambda x: RETROFIT_CATALOGUE.get(x, {}).get("priority", 99))
    return final_feasible, infeasible
