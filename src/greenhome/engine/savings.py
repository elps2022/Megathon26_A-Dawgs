"""Sequential retrofit savings simulator with fuel separation and interaction handling.

Implements sequential interaction order: Fabric -> Controls -> Heating -> Generation.
Defined in docs/ARCHITECTURE.md section 3.6.
"""

from greenhome.engine.carbon import calculate_carbon_kg
from greenhome.engine.feasibility import RETROFIT_CATALOGUE
from greenhome.schema import (
    Element,
    EnergyBreakdown,
    FuelDelta,
    HomeState,
    Property,
    RetrofitRec,
)

# Benchmark U-values by element state (W/m²K)
# Based on RdSAP default lookup tables
U_VALUES = {
    "roof": {
        "uninsulated": 2.3,
        "partial": 0.65,    # ~50-100mm
        "standard": 0.16,   # ~270-300mm
    },
    "wall": {
        "solid_uninsulated": 2.1,
        "solid_insulated": 0.30,
        "cavity_uninsulated": 1.6,
        "cavity_insulated": 0.45,
    },
    "window": {
        "single": 4.8,
        "partial": 3.4,
        "double_old": 2.8,
        "double_new": 1.4,
        "triple": 0.8,
    },
    "floor": {
        "standard": 0.70,
    },
}


def build_home_elements(p: Property) -> list[Element]:
    """Estimate envelope geometry and U-values for the property (Thermal Twin elements)."""
    area = p.floor_area_m2 or 85.0

    # Geometry approximations based on dwelling type
    is_flat = "flat" in p.property_type.lower()
    storeys = 1.0 if is_flat else 2.0
    footprint = area / storeys

    # Envelope areas (m²)
    roof_area = footprint if p.has_own_roof else 0.0
    floor_area = footprint
    perimeter = 4.0 * (footprint ** 0.5)
    wall_gross = perimeter * 2.6 * storeys
    window_area = 0.20 * area  # ~20% glazing ratio
    wall_net = max(wall_gross - window_area, 20.0)

    # Determine U-values
    # Roof
    if not p.has_own_roof or (p.loft_insulation_mm is not None and p.loft_insulation_mm >= 270):
        u_roof = U_VALUES["roof"]["standard"]
    elif p.loft_insulation_mm is not None and p.loft_insulation_mm >= 50:
        u_roof = U_VALUES["roof"]["partial"]
    else:
        u_roof = U_VALUES["roof"]["uninsulated"]

    # Wall
    if p.wall_type == "solid":
        u_wall = U_VALUES["wall"]["solid_insulated"] if p.wall_insulation == "full" else U_VALUES["wall"]["solid_uninsulated"]
    else:
        u_wall = U_VALUES["wall"]["cavity_insulated"] if p.wall_insulation == "full" else U_VALUES["wall"]["cavity_uninsulated"]

    # Glazing
    u_win = U_VALUES["window"].get(p.glazing, U_VALUES["window"]["double_new"])
    u_flr = U_VALUES["floor"]["standard"]

    elements_raw = [
        ("roof", roof_area, u_roof),
        ("wall", wall_net, u_wall),
        ("window", window_area, u_win),
        ("floor", floor_area, u_flr),
    ]

    # Calculate heat loss W/K per element
    losses = [(name, a, u, round(a * u, 2)) for name, a, u in elements_raw]
    total_loss = sum(l for _, _, _, l in losses) or 1.0

    return [
        Element(
            name=name,
            area_m2=round(a, 1),
            u_value=round(u, 2),
            heat_loss_w_per_k=round(loss, 1),
            share=round(loss / total_loss, 3),
        )
        for name, a, u, loss in losses
    ]


def simulate_package_savings(
    p: Property,
    retrofits: list[str],
) -> tuple[EnergyBreakdown, EnergyBreakdown, list[RetrofitRec], FuelDelta, float]:
    """Run sequential simulation across selected retrofits in priority order.

    Order: Fabric -> Controls -> Heating -> Generation.
    Returns:
        (energy_before, energy_after, recommendations, package_saving, package_carbon_kg)
    """
    # Baseline annual energy consumption (kWh)
    base_elec = p.annual_elec_kwh or 3200.0
    base_gas = p.annual_gas_kwh or (12000.0 if p.gas_meter else 0.0)

    # Establish heating demand share vs base load
    # For gas heated: ~80% gas is space heat, 20% domestic hot water/cooking base load
    gas_heat_demand = base_gas * 0.82 if p.gas_meter else 0.0
    gas_base_load = base_gas * 0.18 if p.gas_meter else 0.0

    # For electric heated (storage/direct electric): ~65% elec is space heat
    is_elec_heated = p.heating in ("storage_heaters", "direct_electric", "heat_pump")
    elec_heat_demand = base_elec * 0.65 if is_elec_heated else 0.0
    elec_base_load = base_elec - elec_heat_demand

    current_heat_demand = gas_heat_demand + elec_heat_demand
    current_elec = base_elec
    current_gas = base_gas

    elements_before = build_home_elements(p)
    hlc_before = sum(e.heat_loss_w_per_k for e in elements_before)

    # Sort requested retrofits by priority order to strictly enforce sequence
    ordered_retrofits = sorted(
        [r for r in retrofits if r in RETROFIT_CATALOGUE],
        key=lambda r: RETROFIT_CATALOGUE[r]["priority"],
    )

    recs: list[RetrofitRec] = []
    tot_saved_elec = 0.0
    tot_saved_gas = 0.0

    for r in ordered_retrofits:
        delta_elec = 0.0
        delta_gas = 0.0
        low_factor = 0.85
        high_factor = 1.15
        reason = ""

        if r == "loft_insulation":
            # 12-18% reduction in space heating demand
            pct = 0.15
            saved_heat = current_heat_demand * pct
            current_heat_demand -= saved_heat
            if p.gas_meter and current_gas > 0:
                delta_gas = saved_heat
            else:
                delta_elec = saved_heat
            reason = "Loft insulation top-up cuts roof heat escape by ~15%."

        elif r == "cavity_wall_insulation":
            # 14-22% reduction in space heating demand
            pct = 0.18
            saved_heat = current_heat_demand * pct
            current_heat_demand -= saved_heat
            if p.gas_meter and current_gas > 0:
                delta_gas = saved_heat
            else:
                delta_elec = saved_heat
            reason = "Cavity wall insulation insulates unbonded cavity voids (~18% heat demand drop)."

        elif r == "solid_wall_insulation":
            # 22-30% reduction in space heating demand
            pct = 0.25
            saved_heat = current_heat_demand * pct
            current_heat_demand -= saved_heat
            if p.gas_meter and current_gas > 0:
                delta_gas = saved_heat
            else:
                delta_elec = saved_heat
            reason = "External/internal wall insulation stops severe thermal transmission through solid masonry."

        elif r == "double_glazing":
            # 8-12% reduction in space heating demand
            pct = 0.10
            saved_heat = current_heat_demand * pct
            current_heat_demand -= saved_heat
            if p.gas_meter and current_gas > 0:
                delta_gas = saved_heat
            else:
                delta_elec = saved_heat
            reason = "A-rated double glazing eliminates single pane drafts and reduces window heat loss."

        elif r == "smart_thermostat":
            # 6-10% reduction in remaining heating demand
            pct = 0.08
            saved_heat = current_heat_demand * pct
            current_heat_demand -= saved_heat
            if p.gas_meter and current_gas > 0:
                delta_gas = saved_heat
            else:
                delta_elec = saved_heat
            reason = "Zone scheduling and smart modulation prevents room overheating."

        elif r == "heat_pump":
            # Replaces gas heating entirely. Efficiency SCOP ~3.0
            # Eliminates remaining gas space heating + portion of hot water gas
            gas_eliminated = current_gas
            # Required electricity to deliver heat demand with SCOP 3.0:
            elec_added = (current_heat_demand + gas_base_load * 0.5) / 3.0

            delta_gas = gas_eliminated
            delta_elec = -elec_added  # NEGATIVE saving = increased electricity use!

            current_gas = 0.0
            current_heat_demand = 0.0
            reason = "Heat pump replaces fossil gas with high efficiency (~300% SCOP), cutting gas to zero while increasing electricity."

        elif r == "solar_pv":
            # Rooftop 3.5 kWp array produces ~900 kWh/kWp = ~3,150 kWh/year
            # Solar cuts grid import, not underlying demand
            # Typical domestic self-consumption ratio ~40% (direct use) = ~1,260 kWh import reduction
            annual_gen = 3.5 * 900.0  # ~3150 kWh
            self_consumption = 0.40
            import_reduction = annual_gen * self_consumption

            delta_elec = min(import_reduction, current_elec * 0.50)
            delta_gas = 0.0
            reason = "3.5 kWp solar array generates on-site clean electricity, directly reducing grid imports."

        # Marginal updates
        tot_saved_elec += delta_elec
        tot_saved_gas += delta_gas
        current_elec = max(current_elec - delta_elec, 500.0)
        current_gas = max(current_gas - delta_gas, 0.0)

        tot_kwh = delta_elec + delta_gas
        fuel_delta = FuelDelta(
            elec_kwh=round(delta_elec, 1),
            gas_kwh=round(delta_gas, 1),
            total_kwh=round(tot_kwh, 1),
            low=round(tot_kwh * low_factor, 1),
            high=round(tot_kwh * high_factor, 1),
        )

        carbon_cut = calculate_carbon_kg(delta_elec, delta_gas)
        cost_range = RETROFIT_CATALOGUE[r]["indicative_cost"]

        # Determine suitability based on payback and carbon cut
        suitability = "High" if carbon_cut > 400 or (tot_kwh > 1500 and cost_range[0] < 3000) else "Medium"
        if cost_range[0] > 8000 and carbon_cut < 600:
            suitability = "Low"

        recs.append(
            RetrofitRec(
                retrofit=r,
                suitability=suitability,  # type: ignore
                reason=reason,
                marginal_saving=fuel_delta,
                carbon_kg=round(carbon_cut, 1),
                indicative_cost=cost_range,
            )
        )

    package_kwh = tot_saved_elec + tot_saved_gas
    package_saving = FuelDelta(
        elec_kwh=round(tot_saved_elec, 1),
        gas_kwh=round(tot_saved_gas, 1),
        total_kwh=round(package_kwh, 1),
        low=round(package_kwh * 0.85, 1),
        high=round(package_kwh * 1.15, 1),
    )
    package_carbon = calculate_carbon_kg(tot_saved_elec, tot_saved_gas)

    # Post-retrofit elements for thermal house
    elements_after = build_home_elements(p)
    hlc_after = hlc_before * max(1.0 - (package_kwh / (base_elec + base_gas + 1.0)), 0.4)

    energy_before = EnergyBreakdown(
        elec_kwh=round(base_elec, 1),
        gas_kwh=round(base_gas, 1),
        heat_demand_kwh=round(gas_heat_demand + elec_heat_demand, 1),
        base_load_kwh=round(gas_base_load + elec_base_load, 1),
        hlc_w_per_k=round(hlc_before, 1),
        calibration_k=1.0,
        calibrated=True,
        model_version="greybox_v1",
    )

    energy_after = EnergyBreakdown(
        elec_kwh=round(current_elec, 1),
        gas_kwh=round(current_gas, 1),
        heat_demand_kwh=round(current_heat_demand, 1),
        base_load_kwh=round(gas_base_load + elec_base_load, 1),
        hlc_w_per_k=round(hlc_after, 1),
        calibration_k=1.0,
        calibrated=True,
        model_version="greybox_v1",
    )

    return energy_before, energy_after, recs, package_saving, package_carbon
