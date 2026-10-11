"""Report orchestration engine.
Constructs comprehensive PropertyReport and runs what-if scenarios.
Defined in docs/ARCHITECTURE.md section 4.
"""
import csv
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from greenhome.engine.feasibility import DECLINED_RULE, RETROFIT_CATALOGUE, check_feasibility
from greenhome.engine.savings import (
    EnergyUnavailable,
    apply_retrofits_to_property,
    baseline_energy,
    simulate_package_savings,
)
from greenhome.schema import Infeasible, Property, PropertyReport
from greenhome.models.energy import ModelUnavailable

from greenhome.score.score import calculate_green_home_score

_CONFIG = Path(__file__).resolve().parents[3] / "config"
POOR_CLASSES = {"Poor", "Very poor"}
VALUE_CAVEAT = ("Estimated potential property-value impact based on historical and market relationships in the "
                "dataset. Indicative estimate, not a guaranteed or proven causal increase.")

# Kept for existing callers; the implementation lives with the simulator.
mutate_property_for_scenario = apply_retrofits_to_property


@lru_cache(maxsize=1)
def _value_cfg() -> dict:
    return yaml.safe_load((_CONFIG / "value_uplift.yaml").read_text())


@lru_cache(maxsize=1)
def _emission_assumption() -> str:
    with open(_CONFIG / "emission_factors.csv") as f:
        rows = {r["fuel"]: r for r in csv.DictReader(f)}
    e, g = rows["electricity"], rows["gas"]
    return (f"Emission factors (official conversion factors {e['year']}): {float(e['kg_co2e_per_kwh']):.3f} kg CO2e/kWh "
            f"electricity, {float(g['kg_co2e_per_kwh']):.3f} kg CO2e/kWh gas (config/emission_factors.csv)")


def estimate_indicative_value_uplift(
    base_val: float | None,
    score_before: int,
    score_after: int,
    retrofits: list[str],
) -> dict[str, Any]:
    """Indicative property value impact range (amount and %), from config/value_uplift.yaml.

    Market association based on rating-band improvement; never presented as causal. Interim until the value
    model (ARCHITECTURE 3.8) exists. Returns status "unavailable" when there is no valuation to apply it to.
    """
    c = _value_cfg()
    bands_gained = max(score_after - score_before, 0) / c["score_points_per_band"]
    pct = {}
    for side in ("low", "high"):
        floor = c["min_pct_if_any_retrofit"][side] if retrofits else 0.0
        pct[side] = round(min(max(bands_gained * c["pct_per_band"][side], floor), c["max_pct"][side]), 1)
    method = "Interim rating-band uplift rule (config/value_uplift.yaml)"
    if base_val is None:
        return {"status": "unavailable", "baseline": None, "low": None, "high": None,
                "pct_low": pct["low"], "pct_high": pct["high"], "method": method,
                "caveat": "No current valuation for this property, so the value impact can't be estimated. " + VALUE_CAVEAT}
    return {
        "status": "estimated",
        "baseline": base_val,
        "low": round(base_val * (1.0 + pct["low"] / 100.0), -2),
        "high": round(base_val * (1.0 + pct["high"] / 100.0), -2),
        "pct_low": pct["low"],
        "pct_high": pct["high"],
        "method": method,
        "caveat": VALUE_CAVEAT,
    }


def build_report(
    p: Property,
    retrofits: list[str] | None = None,
    memory: dict | None = None,
    energy_model: str = "greybox",
) -> PropertyReport:
    """Build single-source-of-truth PropertyReport for baseline or what-if scenario."""
    # 1. Baseline Green Home Score
    score_before = calculate_green_home_score(p)
    is_poor = score_before.classification in POOR_CLASSES

    # 2. Key issues detection
    key_issues = []
    for comp in score_before.components:
        if comp.score < 40.0:
            key_issues.append(f"Low {comp.component.lower()} score ({comp.score:.0f}/100)")
    # "Worse than similar homes": billed use above the 90th percentile for homes with the same description
    from greenhome.engine.detector import key_issue, similar_homes_check
    similar = similar_homes_check(p)
    key_issues.extend(key_issue(similar))
    if is_poor and not key_issues:
        key_issues.append("Low overall energy efficiency classification")

    # 3. Feasibility analysis
    constraints = []
    if memory and "constraints" in memory:
        constraints = memory["constraints"]
    feasible_all, infeasible = check_feasibility(p, constraints)

    assumptions = [
        "Sequential interaction ordering applied (Fabric -> Controls -> Heating -> Solar)",
        _emission_assumption(),
        "Solar PV offsets grid import, not home energy demand",
        "All savings labelled as estimates, not guarantees",
    ]

    # What-if: simulate only requested retrofits that pass the hard feasibility rules. A retrofit the customer
    # declined earlier may still be explored when asked for explicitly. Otherwise use every feasible retrofit.
    if retrofits is not None:
        blocked = {i.retrofit: i.rule for i in infeasible if i.rule != DECLINED_RULE}
        unknown = [r for r in retrofits if r not in RETROFIT_CATALOGUE]
        infeasible = infeasible + [Infeasible(retrofit=r, rule="Unknown retrofit: not in the catalogue.") for r in unknown]
        active_retrofits = [r for r in dict.fromkeys(retrofits) if r in RETROFIT_CATALOGUE and r not in blocked]
        skipped = [f"{r} ({blocked[r]})" for r in retrofits if r in blocked]
        if skipped or unknown:
            assumptions.append("Requested but not simulated: " + "; ".join(skipped + [f"{r} (unknown retrofit)" for r in unknown]))
    else:
        active_retrofits = feasible_all
    p_after = apply_retrofits_to_property(p, active_retrofits)

    # 4. Sequential savings simulation (unavailable, never 0, when the baseline can't be established)
    try:
        energy_before, energy_after, recs, package_saving, package_carbon = simulate_package_savings(
            p, active_retrofits, energy_model
        )
        # Spec 3.11: If customer declined a retrofit in past session, down-rank in suitability
        # with reason shown: "You declined X on <date>".
        if memory and memory.get("declined_retrofits"):
            declined = memory["declined_retrofits"]
            for r in recs:
                if r.retrofit in declined:
                    r.suitability = "Low"
                    val = declined[r.retrofit]
                    date_str = val.get("date", "12 Oct 2026") if isinstance(val, dict) else str(val)
                    r.reason = f"You declined this on {date_str}. {r.reason}"
            from greenhome.engine.savings import rank_recommendations
            rank_recommendations(recs)
    except EnergyUnavailable as exc:
        energy_before = energy_after = package_saving = package_carbon = None
        recs = []
        improvement_potential = None
        p = p.model_copy(update={"data_flags": [*p.data_flags, "energy_unavailable"]})
        assumptions.append(f"Energy savings unavailable: {exc}")
        p_after.annual_elec_kwh = p_after.annual_gas_kwh = None
    else:
        flags = baseline_energy(p, energy_model)[2]
        if flags:
            p = p.model_copy(update={"data_flags": [*p.data_flags, *flags]})
            source = ("NEED log-linear model, config/need_calibration.yaml"
                      if "_fallback" in energy_before.model_version
                      else f"{energy_before.model_version} energy model trained on metered use (NEED)")
            assumptions.append(f"Missing bills replaced by expected use for similar homes ({source}); "
                               "flagged in data_flags")
        if energy_before.model_version.endswith("+mode1"):
            assumptions.append("Baseline use is next year's forecast from this home's own bills and description "
                               "(mode 1 energy model; more accurate than last year's bill alone on held-out metered "
                               "homes); the billed figures are in property.annual_elec_kwh / annual_gas_kwh")
        if "_fallback" in energy_before.model_version:
            assumptions.append(f"Trained energy model '{energy_model}' not found (run `make train`); fabric "
                               "savings use the physics heat-loss model instead")

        # 5. Improvement potential: 100 * (E_baseline - E_after) / E_baseline
        purchased_base = energy_before.elec_kwh + energy_before.gas_kwh
        purchased_after = energy_after.elec_kwh + energy_after.gas_kwh
        improvement_potential = (round(100.0 * (purchased_base - purchased_after) / purchased_base, 1)
                                 if purchased_base > 0 else None)
        # Post-scenario energy feeds the consumption-intensity sub-score. The "before" score reads the home's billed
        # use, so the simulated change (after / before baseline, per fuel) is applied to the same billed figures;
        # otherwise a mode-1 forecast above the bill could make a retrofit look like it lowers the score.
        def _after(billed, before, after):
            if billed is None:
                return after
            return round(billed * after / before, 1) if before > 0 else after
        p_after.annual_elec_kwh = _after(p.annual_elec_kwh, energy_before.elec_kwh, energy_after.elec_kwh)
        p_after.annual_gas_kwh = _after(p.annual_gas_kwh, energy_before.gas_kwh, energy_after.gas_kwh)

    # 6. Post-scenario Green Home Score
    score_after = calculate_green_home_score(p_after)

    # 7. Indicative value range: the value model (real matched sales; rating-band premium with postcode-sector
    # fixed effects, models/value_model.py), or the interim rating-band rule if it isn't trained.
    try:
        from greenhome.models.value_model import value_estimate
        val_estimate = value_estimate(p.model_dump(), p_after.model_dump(), active_retrofits)
    except ModelUnavailable:
        val_estimate = estimate_indicative_value_uplift(
            p.current_value_estimate,
            score_before.total,
            score_after.total,
            active_retrofits,
        )
        assumptions.append("Value model not installed (make models SRC=...); value impact uses the interim rule")

    return PropertyReport(
        property=p,
        score=score_before,
        score_after=score_after,
        is_poor=is_poor,
        key_issues=key_issues,
        similar_homes=similar,
        improvement_potential_pct=improvement_potential,
        recommendations=recs,
        infeasible=infeasible,
        package_saving=package_saving,
        package_carbon_kg=package_carbon,
        value=val_estimate,
        energy_before=energy_before,
        energy_after=energy_after,
        assumptions=assumptions,
    )
