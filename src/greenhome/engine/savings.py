"""Sequential retrofit savings simulator with fuel separation and interaction handling.
Implements sequential interaction order: Fabric -> Controls -> Heating -> Generation.
Defined in docs/ARCHITECTURE.md section 3.6.

Every constant is read from config/ (with a source). Missing bills fall back to the flagged NEED estimate for
similar homes; if that is impossible too, EnergyUnavailable is raised so the report can say "unavailable".
Interim: per-retrofit cuts use catalogue ranges until the V1 grey-box energy model (3.5) is wired in.
"""
from functools import lru_cache
from pathlib import Path

import yaml

from greenhome.engine.carbon import calculate_carbon_kg
from greenhome.engine.feasibility import RETROFIT_CATALOGUE
from greenhome.models.expected_use import expected_annual_kwh
from greenhome.schema import (
    Element,
    EnergyBreakdown,
    FuelDelta,
    Property,
    RetrofitRec,
)

_CONFIG = Path(__file__).resolve().parents[3] / "config"
FABRIC_AND_CONTROLS = ("loft_insulation", "cavity_wall_insulation", "solid_wall_insulation",
                       "double_glazing", "smart_thermostat")
REASONS = {
    "loft_insulation": "Loft insulation top-up cuts heat escaping through the roof.",
    "cavity_wall_insulation": "Cavity wall insulation fills unfilled cavities, cutting heat loss through the walls.",
    "solid_wall_insulation": "External/internal wall insulation stops heat loss through solid masonry.",
    "double_glazing": "A-rated double glazing reduces window heat loss and draughts.",
    "smart_thermostat": "Zone scheduling and smart modulation prevent overheating rooms.",
    "heat_pump": "Heat pump replaces gas heating; gas falls to zero while electricity use rises.",
    "solar_pv": "Solar panels generate electricity on site, reducing grid imports (not underlying demand).",
}


class EnergyUnavailable(ValueError):
    """Baseline energy can't be established from bills or from similar homes."""


@lru_cache(maxsize=None)
def _cfg(name: str) -> dict:
    return yaml.safe_load((_CONFIG / name).read_text())


def apply_retrofits_to_property(p: Property, retrofits: list[str]) -> Property:
    """Post-retrofit clone of the property (feature changes only; energy is simulated separately)."""
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
        clone.solar_kwp = max(clone.solar_kwp, _cfg("simulator.yaml")["solar"]["default_kwp"])
    return clone


def build_home_elements(p: Property) -> list[Element]:
    """Envelope geometry and U-values (config/geometry.yaml, config/u_values.yaml). Empty if floor area unknown."""
    if p.floor_area_m2 is None:
        return []
    geo, u = _cfg("geometry.yaml"), _cfg("u_values.yaml")
    area = p.floor_area_m2
    is_flat = "flat" in p.property_type.lower()
    storeys = geo["storeys"]["flat" if is_flat else "house"]
    footprint = area / storeys

    roof_area = footprint if p.has_own_roof else 0.0
    perimeter = 4.0 * (footprint ** 0.5)
    window_area = geo["glazing_ratio_of_floor_area"] * area
    wall_net = max(perimeter * geo["storey_height_m"] * storeys - window_area, geo["min_net_wall_area_m2"])

    t = geo["loft_thresholds_mm"]
    mm = p.loft_insulation_mm
    if not p.has_own_roof or (mm is not None and mm >= t["standard"]):
        u_roof = u["roof"]["standard"]
    elif mm is not None and mm >= t["partial"]:
        u_roof = u["roof"]["partial"]
    else:
        u_roof = u["roof"]["uninsulated"]
    insulated = "insulated" if p.wall_insulation == "full" else "uninsulated"
    u_wall = u["wall"][f"{'solid' if p.wall_type == 'solid' else 'cavity'}_{insulated}"]
    u_win = u["window"].get(p.glazing, u["window"]["double_new"])

    losses = [(name, a, uv, round(a * uv, 2)) for name, a, uv in
              [("roof", roof_area, u_roof), ("wall", wall_net, u_wall),
               ("window", window_area, u_win), ("floor", footprint, u["floor"]["standard"])]]
    total = sum(l for *_, l in losses) or 1.0
    return [Element(name=n, area_m2=round(a, 1), u_value=uv, heat_loss_w_per_k=l, share=round(l / total, 3))
            for n, a, uv, l in losses]


def baseline_energy(p: Property) -> tuple[float, float, list[str]]:
    """(elec, gas, flags): bills first, then the flagged NEED estimate for similar homes."""
    flags: list[str] = []
    expected = None
    elec, gas = p.annual_elec_kwh, p.annual_gas_kwh
    if gas is None and not p.gas_meter:
        gas = 0.0  # no gas supply: a real zero, not missing data
    if elec is None or gas is None:
        expected = expected_annual_kwh(p)
    if elec is None:
        elec = expected["elec"]
        flags.append("elec_estimated_from_similar_homes")
    if gas is None:
        gas = expected["gas"]
        flags.append("gas_estimated_from_similar_homes")
    if elec is None or gas is None:
        raise EnergyUnavailable("No usable bills and too little property data to estimate expected use.")
    return float(elec), float(gas), flags


def _heating_efficiency(heating: str) -> float:
    systems = _cfg("efficiencies.yaml")["heating_systems"]
    return systems[heating]["efficiency"]


def simulate_package_savings(
    p: Property,
    retrofits: list[str],
) -> tuple[EnergyBreakdown, EnergyBreakdown, list[RetrofitRec], FuelDelta, float]:
    """Run sequential simulation across selected retrofits in priority order.

    Order: Fabric -> Controls -> Heating -> Generation.
    Returns:
        (energy_before, energy_after, recommendations, package_saving, package_carbon_kg)
    Raises EnergyUnavailable when the baseline can't be established.
    """
    sim = _cfg("simulator.yaml")
    base_elec, base_gas, baseline_flags = baseline_energy(p)

    # Split each fuel into space heating vs base load (hot water, cooking, appliances).
    gas_heat_demand = base_gas * sim["gas_heating_share"]["value"] if p.gas_meter else 0.0
    gas_base_load = base_gas - gas_heat_demand
    is_elec_heated = p.heating in ("storage_heaters", "direct_electric", "heat_pump")
    elec_heat_demand = base_elec * sim["electric_heating_share"]["value"] if is_elec_heated else 0.0
    elec_base_load = base_elec - elec_heat_demand

    current_heat_demand = gas_heat_demand + elec_heat_demand  # in delivered-fuel kWh
    current_elec = base_elec
    current_gas = base_gas

    ordered_retrofits = sorted(
        [r for r in retrofits if r in RETROFIT_CATALOGUE],
        key=lambda r: RETROFIT_CATALOGUE[r]["priority"],
    )

    recs: list[RetrofitRec] = []
    tot_saved_elec = tot_saved_gas = 0.0
    tot_low = tot_high = 0.0

    for r in ordered_retrofits:
        delta_elec = delta_gas = 0.0
        band = None  # (low, high) total-kWh range; defaults to the point estimate

        if r in FABRIC_AND_CONTROLS:
            # Fractional cut to the heating demand left by earlier steps (catalogue range, sourced).
            lo, hi = RETROFIT_CATALOGUE[r]["saving_pct_heating"]
            mid = (lo + hi) / 2
            saved_heat = current_heat_demand * mid / 100
            band = (saved_heat * lo / mid, saved_heat * hi / mid)
            current_heat_demand -= saved_heat
            if p.gas_meter and current_gas > 0:
                delta_gas = saved_heat
            else:
                delta_elec = saved_heat

        elif r == "heat_pump":
            hp = _cfg("efficiencies.yaml")["heating_systems"]["heat_pump"]
            boiler_eff = _heating_efficiency(p.heating) if p.gas_meter else 1.0
            # Useful heat = remaining gas heating x boiler efficiency (+ hot-water share of base load).
            useful_heat = (current_heat_demand + gas_base_load * sim["heat_pump_hot_water_share_of_base_load"]["value"]) * boiler_eff
            elec_added = useful_heat / hp["efficiency"]
            delta_gas = current_gas
            delta_elec = -elec_added  # negative saving: electricity use rises
            # Higher SCOP -> less electricity added -> larger net saving; the band follows the SCOP range.
            band = (delta_gas - useful_heat / hp["efficiency_low"], delta_gas - useful_heat / hp["efficiency_high"])
            current_gas = 0.0
            current_heat_demand = 0.0

        elif r == "solar_pv":
            sol = sim["solar"]
            yield_kwh = sol["yield_kwh_per_kwp"].get(p.local_authority, sol["yield_kwh_per_kwp"]["default"])
            import_reduction = sol["default_kwp"] * yield_kwh * sol["self_consumption"]
            delta_elec = min(import_reduction, current_elec * sol["max_share_of_import"])

        tot_saved_elec += delta_elec
        tot_saved_gas += delta_gas
        current_elec = max(current_elec - delta_elec, 0.0)
        current_gas = max(current_gas - delta_gas, 0.0)

        tot_kwh = delta_elec + delta_gas
        low, high = sorted(band) if band else (tot_kwh, tot_kwh)
        tot_low, tot_high = tot_low + low, tot_high + high
        carbon_cut = calculate_carbon_kg(delta_elec, delta_gas)
        cost_range = RETROFIT_CATALOGUE[r]["indicative_cost"]

        # Suitability from carbon cut, energy saved and capital cost (thresholds owned by B, unchanged).
        suitability = "High" if carbon_cut > 400 or (tot_kwh > 1500 and cost_range[0] < 3000) else "Medium"
        if cost_range[0] > 8000 and carbon_cut < 600:
            suitability = "Low"

        recs.append(RetrofitRec(
            retrofit=r,
            suitability=suitability,  # type: ignore
            reason=REASONS[r],
            marginal_saving=FuelDelta(elec_kwh=round(delta_elec, 1), gas_kwh=round(delta_gas, 1),
                                      total_kwh=round(tot_kwh, 1), low=round(low, 1), high=round(high, 1)),
            carbon_kg=round(carbon_cut, 1),
            indicative_cost=cost_range,
        ))

    package_kwh = tot_saved_elec + tot_saved_gas
    package_saving = FuelDelta(elec_kwh=round(tot_saved_elec, 1), gas_kwh=round(tot_saved_gas, 1),
                               total_kwh=round(package_kwh, 1), low=round(tot_low, 1), high=round(tot_high, 1))
    package_carbon = calculate_carbon_kg(tot_saved_elec, tot_saved_gas)

    elements_before = build_home_elements(p)
    elements_after = build_home_elements(apply_retrofits_to_property(p, ordered_retrofits))
    hlc = lambda els: round(sum(e.heat_loss_w_per_k for e in els), 1) if els else None
    version = "simulator_v0" + ("+need_expected_use" if baseline_flags else "")

    energy_before = EnergyBreakdown(
        elec_kwh=round(base_elec, 1), gas_kwh=round(base_gas, 1),
        heat_demand_kwh=round(gas_heat_demand + elec_heat_demand, 1),
        base_load_kwh=round(gas_base_load + elec_base_load, 1),
        hlc_w_per_k=hlc(elements_before), calibration_k=None, calibrated=False, model_version=version,
    )
    energy_after = EnergyBreakdown(
        elec_kwh=round(current_elec, 1), gas_kwh=round(current_gas, 1),
        heat_demand_kwh=round(current_heat_demand, 1),
        base_load_kwh=round(gas_base_load + elec_base_load, 1),
        hlc_w_per_k=hlc(elements_after), calibration_k=None, calibrated=False, model_version=version,
    )
    return energy_before, energy_after, recs, package_saving, package_carbon
