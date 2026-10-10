"""Report orchestration engine.

Constructs comprehensive PropertyReport and runs what-if scenarios.
Defined in docs/ARCHITECTURE.md section 4.
"""

from typing import Any
from greenhome.engine.feasibility import check_feasibility
from greenhome.engine.savings import simulate_package_savings
from greenhome.schema import Property, PropertyReport, ScoreResult
from greenhome.score.score import calculate_green_home_score


def estimate_indicative_value_uplift(
    base_val: float | None,
    score_before: int,
    score_after: int,
    retrofits: list[str],
) -> dict[str, Any]:
    """Calculate indicative property value impact range (£ and %).

    Market association based on EPC band improvement. Never presented as causal.
    """
    baseline = base_val or 350000.0
    score_gain = max(score_after - score_before, 0)

    # 10 score points roughly equals 1 full EPC band
    # Literature association (e.g. BEIS/DESNZ): ~1.5% to 3.5% per EPC band
    bands_gained = score_gain / 12.0
    pct_low = round(min(max(bands_gained * 1.5, 0.5 if retrofits else 0.0), 8.0), 1)
    pct_high = round(min(max(bands_gained * 3.5, 1.5 if retrofits else 0.0), 14.0), 1)

    low_val = round(baseline * (1.0 + pct_low / 100.0), -2)
    high_val = round(baseline * (1.0 + pct_high / 100.0), -2)

    return {
        "baseline": baseline,
        "low": low_val,
        "high": high_val,
        "pct_low": pct_low,
        "pct_high": pct_high,
        "method": "Hedonic market association indexed by score uplift band",
        "caveat": "Estimated potential property-value impact based on historical and market relationships in the dataset. Indicative estimate, not a guaranteed or proven causal increase.",
    }


def mutate_property_for_scenario(p: Property, retrofits: list[str]) -> Property:
    """Create a post-retrofit property clone with upgraded features for score recalculation."""
    clone = p.model_copy(deep=True)
    r_set = set(retrofits)

    if "loft_insulation" in r_set:
        clone.loft_insulation_mm = 300.0
    if "cavity_wall_insulation" in r_set or "solid_wall_insulation" in r_set:
        clone.wall_insulation = "full"
    if "double_glazing" in r_set:
        clone.glazing = "double_new"
    if "smart_thermostat" in r_set:
        clone.smart_controls = "smart_thermostat"
    if "heat_pump" in r_set:
        clone.heating = "heat_pump"
    if "solar_pv" in r_set:
        clone.solar_kwp = max(clone.solar_kwp, 3.5)

    return clone


def build_report(
    p: Property,
    retrofits: list[str] | None = None,
    memory: dict | None = None,
    energy_model: str = "greybox",
) -> PropertyReport:
    """Build single-source-of-truth PropertyReport for baseline or what-if scenario."""
    # 1. Baseline Green Home Score
    score_before = calculate_green_home_score(p)
    is_poor = score_before.total < 50

    # 2. Key issues detection
    key_issues = []
    for comp in score_before.components:
        if comp.score < 40.0:
            key_issues.append(f"Low {comp.component.lower()} score ({comp.score:.0f}/100)")
    if is_poor and not key_issues:
        key_issues.append("Low overall energy efficiency classification")

    # 3. Feasibility analysis
    constraints = []
    if memory and "constraints" in memory:
        constraints = memory["constraints"]

    feasible_all, infeasible = check_feasibility(p, constraints)

    # If explicit retrofits passed (what-if scenario), use them; otherwise evaluate all feasible retrofits
    active_retrofits = retrofits if retrofits is not None else feasible_all

    # 4. Sequential savings simulation
    energy_before, energy_after, recs, package_saving, package_carbon = simulate_package_savings(
        p, active_retrofits
    )

    # 5. Improvement potential percentage
    # 100 * (E_baseline - E_after) / E_baseline
    purchased_base = energy_before.elec_kwh + energy_before.gas_kwh
    purchased_after = energy_after.elec_kwh + energy_after.gas_kwh
    if purchased_base > 0:
        improvement_potential = round(100.0 * (purchased_base - purchased_after) / purchased_base, 1)
    else:
        improvement_potential = None

    # 6. Post-scenario Green Home Score
    p_after = mutate_property_for_scenario(p, active_retrofits)
    # Update post-scenario energy fields for consumption intensity scoring
    p_after.annual_elec_kwh = energy_after.elec_kwh
    p_after.annual_gas_kwh = energy_after.gas_kwh
    score_after = calculate_green_home_score(p_after)

    # 7. Indicative value uplift range
    val_estimate = estimate_indicative_value_uplift(
        p.current_value_estimate,
        score_before.total,
        score_after.total,
        active_retrofits,
    )

    return PropertyReport(
        property=p,
        score=score_before,
        score_after=score_after,
        is_poor=is_poor,
        key_issues=key_issues,
        improvement_potential_pct=improvement_potential,
        recommendations=recs,
        infeasible=infeasible,
        package_saving=package_saving,
        package_carbon_kg=package_carbon,
        value=val_estimate,
        energy_before=energy_before,
        energy_after=energy_after,
        assumptions=[
            "Sequential interaction ordering applied (Fabric -> Controls -> Heating -> Solar)",
            "UK DESNZ 2024 emission factors used: 0.207 kg CO2e/kWh elec, 0.183 kg CO2e/kWh gas",
            "Solar PV offsets grid import, not home energy demand",
            "All savings labelled as estimates, not guarantees",
        ],
    )
