"""Sequential retrofit savings simulator with fuel separation and interaction handling.
Implements sequential interaction order: Fabric -> Controls -> Heating -> Generation.
Defined in docs/ARCHITECTURE.md section 3.6.

Every constant is read from config/ (with a source). Missing bills fall back to the energy model's flagged
estimate for similar homes; if that is impossible too, EnergyUnavailable is raised so the report can say
"unavailable".

Fabric retrofits go through the energy model (models/energy.py): the retrofit edits the property, the model
predicts again, and the predicted change is applied to the home's own use (calibrated to its bills):
  - gas-heated home, V1: saving = current gas x (1 - V1(after) / V1(before)), which is the physics change
    raised to the elasticity real installs show (energy_greybox); V2 only for cavity walls, the one fabric
    measure it can see.
  - other cases (electric heating, which the NEED electricity model can't separate, or a measure the model
    can't see): saving = current heating demand x (1 - (physics space heat after / before)^elasticity).
  - band: the same physics change at the low / high elasticity implied by loft vs cavity installs.
  - no prediction possible (e.g. floor area unknown): the catalogue's % range, labelled.
Controls use the catalogue range (no physics hook yet); heat pumps and solar use the simulator rules below.
If the trained models aren't on disk (`make train`), physics deltas and the log-linear estimate are used.
"""
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from greenhome.engine.carbon import calculate_carbon_kg
from greenhome.engine.feasibility import RETROFIT_CATALOGUE, assess, load_catalogue
from greenhome.models.energy import ModelUnavailable, load_model, physics_elasticity, similar_home_kwh
from greenhome.schema import (
    Element,
    EnergyBreakdown,
    FuelDelta,
    Property,
    RetrofitRec,
)

_CONFIG = Path(__file__).resolve().parents[3] / "config"
FABRIC = ("loft_insulation", "cavity_wall_insulation", "solid_wall_insulation", "double_glazing")
FABRIC_AND_CONTROLS = (*FABRIC, "smart_thermostat")
# Fabric measures each trained model responds to (V2 only sees cavity walls, via cavity_installed); for the rest
# the simulator uses physics deltas (ARCHITECTURE 3.5).
MODEL_LEARNS = {"greybox": set(FABRIC), "ml": {"cavity_wall_insulation"}, "physics": set()}
METHOD_NOTE = {
    "model": "Saving estimated by the {model} energy model (trained on metered use), calibrated to this home.",
    "physics": "Saving estimated from the physics heat-loss model, scaled to what real installs achieve, "
               "calibrated to this home.",
    "catalogue": "Saving estimated from the catalogue's typical % range (too little data for the energy model).",
    "controls": "Saving estimated from the catalogue's typical % range for heating controls.",
}
MODEL_NAMES = {"greybox_v1": "V1 grey-box", "ml_v2": "V2 machine-learning", "physics": "physics-only"}
REASONS = {
    "loft_insulation": "Loft insulation top-up cuts heat escaping through the roof.",
    "cavity_wall_insulation": "Cavity wall insulation fills unfilled cavities, cutting heat loss through the walls.",
    "solid_wall_insulation": "External/internal wall insulation stops heat loss through solid masonry.",
    "double_glazing": "A-rated double glazing reduces window heat loss and draughts.",
    "smart_thermostat": "Zone scheduling and smart modulation prevent overheating rooms.",
    "heat_pump": "Heat pump replaces the current heating and hot water; that fuel falls while heat-pump electricity is added.",
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
    if "cavity_wall_insulation" in r_set:
        clone.cavity_filled_retrofit = True  # what the energy model sees as a cavity fill
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
    """Per-element heat loss for the house diagram, from the energy model's physics (models/physics.py), so the
    diagram and the model share one implementation. Empty if floor area unknown."""
    if p.floor_area_m2 is None:
        return []
    import pandas as pd
    from greenhome.models.physics import elements, envelope, property_fabric
    fabric, _ = property_fabric(pd.DataFrame([p.model_dump()]))
    return [Element(**e) for e in elements(envelope(fabric).iloc[0])]


def energy_model(name: str = "greybox"):
    """(model, fallback): the selected energy model, or physics only when its artifacts aren't trained."""
    try:
        return load_model(name), False
    except ModelUnavailable:
        return load_model("physics"), name != "physics"


def _baseline(p: Property, energy_model_name: str = "greybox") -> tuple[float, float, list[str], bool]:
    """(elec, gas, flags, forecast_used). A billed fuel is forecast for next year by mode 1 (the home's own bill
    + description, models/energy.py), which beats the bill alone on held-out NEED homes; the bill itself is used
    when the model isn't trained (or the physics model is chosen). A missing bill falls back to the flagged
    estimate for similar homes (mode 2, energy.similar_home_kwh)."""
    flags: list[str] = []
    elec, gas = p.annual_elec_kwh, p.annual_gas_kwh
    if gas is None and not p.gas_meter:
        gas = 0.0  # no gas supply: a real zero, not missing data
    forecast_used = False
    model, fallback = energy_model(energy_model_name)
    if not fallback and hasattr(model, "forecast_frame") and (p.annual_elec_kwh or p.annual_gas_kwh):
        f = model.forecast_frame(pd.DataFrame([p.model_dump()])).iloc[0]
        if p.annual_elec_kwh and np.isfinite(f.elec_kwh):
            elec, forecast_used = round(float(f.elec_kwh), 1), True
        if p.annual_gas_kwh and np.isfinite(f.gas_kwh):
            gas, forecast_used = round(float(f.gas_kwh), 1), True
    if elec is None or gas is None:
        expected = similar_home_kwh(p.model_dump(), "mean", energy_model_name)
        if elec is None:
            elec = expected["elec"]
            flags.append("elec_estimated_from_similar_homes")
        if gas is None:
            gas = expected["gas"]
            flags.append("gas_estimated_from_similar_homes")
    if elec is None or gas is None:
        raise EnergyUnavailable("No usable bills and too little property data to estimate expected use.")
    return float(elec), float(gas), flags, forecast_used


def baseline_energy(p: Property, energy_model_name: str = "greybox") -> tuple[float, float, list[str]]:
    """(elec, gas, flags): the bill-based forecast first, then the flagged estimate for similar homes."""
    return _baseline(p, energy_model_name)[:3]


def _heating_efficiency(heating: str) -> float:
    systems = _cfg("efficiencies.yaml")["heating_systems"]
    return systems[heating]["efficiency"]


def _heating_fuel(p: Property) -> str | None:
    """Metered fuel the space heating draws from: "gas", "electricity", or None when it is unmetered
    (oil, LPG, district heat). Unknown heating with a gas meter is treated as gas heating, as before."""
    system = _cfg("efficiencies.yaml")["heating_systems"].get(p.heating)
    if system is None:
        return "gas" if p.gas_meter else None
    fuel = system["fuel"]
    if fuel == "gas" and not p.gas_meter:
        return None
    return fuel if fuel in ("gas", "electricity") else None


def _fabric_ratios(model, fallback: bool, p: Property, ordered: list[str]) -> tuple[list[dict], pd.Series]:
    """Per retrofit in package order, the model's gas ratio after / before (NaN if it can't predict) and the
    physics space-heat ratio after / before, each step applied on top of the previous ones; plus the model's
    prediction for the home as it is."""
    states = [p] + [apply_retrofits_to_property(p, ordered[:i + 1]) for i in range(len(ordered))]
    pred = model.predict_frame(pd.DataFrame([s.model_dump() for s in states]))
    gas, space = pred.gas_kwh.to_numpy(), pred.physics_space_heat_kwh.to_numpy()
    phys_gas = space + pred.physics_hot_water_kwh.to_numpy()
    with np.errstate(divide="ignore", invalid="ignore"):
        model_ratio = np.full(len(ordered), np.nan) if fallback else gas[1:] / gas[:-1]
        physics_ratio = space[1:] / space[:-1]
        physics_gas_ratio = phys_gas[1:] / phys_gas[:-1]
    return [{"model": float(m), "physics": float(ph), "physics_gas": float(pg)}
            for m, ph, pg in zip(model_ratio, physics_ratio, physics_gas_ratio)], pred.iloc[0]


def _suitability(bill_gbp: float, carbon_kg: float, cost: tuple[float, float], capped: bool) -> tuple[str, float | None]:
    """High / Medium / Low from saving per pound (simple payback) and constraints (ARCHITECTURE 3.9; thresholds and
    source in config/retrofit_catalogue.yaml). Monotone: more saving, more carbon cut or lower cost never lowers it."""
    t = load_catalogue()["suitability"]
    payback = (sum(cost) / 2) / bill_gbp if bill_gbp > 0 else None
    if payback is not None and payback <= t["high_max_payback_years"]:
        level = "High"
    elif (payback is not None and payback <= t["medium_max_payback_years"]) or carbon_kg >= t["medium_min_carbon_kg"]:
        level = "Medium"
    else:
        level = "Low"
    if capped and level == "High":
        level = "Medium"
    return level, payback


def rank_recommendations(recs: list[RetrofitRec]) -> None:
    """Set rank (1 = best value): suitability first, then shortest payback, then biggest carbon cut. The list itself
    stays in simulation order (fabric, controls, heating, generation), which the marginal savings depend on."""
    tier = {"High": 0, "Medium": 1, "Low": 2}
    key = lambda r: (tier[r.suitability], r.payback_years if r.payback_years is not None else float("inf"), -r.carbon_kg)
    for i, r in enumerate(sorted(recs, key=key), start=1):
        r.rank = i


def simulate_package_savings(
    p: Property,
    retrofits: list[str],
    energy_model_name: str = "greybox",
) -> tuple[EnergyBreakdown, EnergyBreakdown, list[RetrofitRec], FuelDelta, float]:
    """Run sequential simulation across selected retrofits in priority order.

    Order: Fabric -> Controls -> Heating -> Generation.
    Returns:
        (energy_before, energy_after, recommendations, package_saving, package_carbon_kg)
    Raises EnergyUnavailable when the baseline can't be established.
    """
    sim = _cfg("simulator.yaml")
    model, fallback = energy_model(energy_model_name)
    base_elec, base_gas, baseline_flags, forecast_used = _baseline(p, energy_model_name)

    # Split each fuel into space heating vs base load (hot water, cooking, appliances). Only the fuel the
    # heating system actually uses carries heating demand: a gas meter in an electrically heated home is cooking.
    heating_fuel = _heating_fuel(p)
    # The home's own space-heating share from its bills (degree-day fit) when known, else the sourced default.
    own = p.heating_share if p.heating_share is not None and 0 < p.heating_share <= 1 else None
    gas_share = own if own is not None else sim["gas_heating_share"]["value"]
    elec_share = own if own is not None else sim["electric_heating_share"]["value"]
    gas_heat_demand = base_gas * gas_share if heating_fuel == "gas" else 0.0
    gas_base_load = base_gas - gas_heat_demand
    elec_heat_demand = base_elec * elec_share if heating_fuel == "electricity" else 0.0
    elec_base_load = base_elec - elec_heat_demand

    current_heat_demand = gas_heat_demand + elec_heat_demand  # in delivered-fuel kWh
    current_elec = base_elec
    current_gas = base_gas

    ordered_retrofits = sorted(
        [r for r in retrofits if r in RETROFIT_CATALOGUE],
        key=lambda r: RETROFIT_CATALOGUE[r]["priority"],
    )

    ratios, pred_now = _fabric_ratios(model, fallback, p, ordered_retrofits)
    learns = set() if fallback else MODEL_LEARNS.get(energy_model_name, set())
    # Physics-only (explicitly chosen) keeps the raw physics change; every other path uses the observed elasticity.
    el = ({"beta": 1.0, "low": 1.0, "high": 1.0} if energy_model_name == "physics" and not fallback
          else physics_elasticity())
    model_label = MODEL_NAMES.get(model.version, model.version)

    recs: list[RetrofitRec] = []
    prices = _cfg("energy_prices.yaml")["gbp_per_kwh"]
    insulate_first = "insulate_first" in assess(p.model_dump())["flags"].get("heat_pump", [])
    tot_saved_elec = tot_saved_gas = 0.0
    tot_low = tot_high = 0.0

    for i, r in enumerate(ordered_retrofits):
        delta_elec = delta_gas = 0.0
        band = None  # (low, high) total-kWh range; defaults to the point estimate
        method = None

        if r in FABRIC_AND_CONTROLS:
            ratio = ratios[i]
            clip = lambda kwh: min(max(kwh, 0.0), current_heat_demand)
            if r in FABRIC and heating_fuel == "gas" and r in learns and np.isfinite(ratio["model"]):
                # Change the home, predict again: the model's relative change, applied to this home's gas.
                saved_heat = clip(current_gas * (1 - ratio["model"]))
                band = tuple(clip(current_gas * (1 - ratio["physics_gas"] ** el[s])) for s in ("low", "high"))
                method = "model"
            elif r in FABRIC and np.isfinite(ratio["physics"]):
                saved_heat = clip(current_heat_demand * (1 - ratio["physics"] ** el["beta"]))
                band = tuple(clip(current_heat_demand * (1 - ratio["physics"] ** el[s])) for s in ("low", "high"))
                method = "physics"
            else:
                # Fractional cut to the heating demand left by earlier steps (catalogue range, sourced).
                lo, hi = RETROFIT_CATALOGUE[r]["saving_pct_heating"]
                mid = (lo + hi) / 2
                saved_heat = current_heat_demand * mid / 100
                band = (saved_heat * lo / mid, saved_heat * hi / mid)
                method = "catalogue" if r in FABRIC else "controls"
            current_heat_demand -= saved_heat
            if heating_fuel == "gas":
                delta_gas = saved_heat
            elif heating_fuel == "electricity":
                delta_elec = saved_heat

        elif r == "heat_pump":
            hp = _cfg("efficiencies.yaml")["heating_systems"]["heat_pump"]
            if heating_fuel == "gas":
                # Replaced: remaining gas space heating + the hot-water share of gas base load (cooking stays).
                replaced = current_heat_demand + gas_base_load * sim["heat_pump_hot_water_share_of_base_load"]["value"]
                eff = _heating_efficiency(p.heating) if p.heating in _cfg("efficiencies.yaml")["heating_systems"] else 1.0
            elif heating_fuel == "electricity":
                replaced = current_heat_demand  # storage / direct electric space heating
                eff = _heating_efficiency(p.heating)
            else:
                replaced, eff = 0.0, 1.0  # unmetered heating fuel: nothing metered to replace
            useful_heat = replaced * eff
            elec_added = useful_heat / hp["efficiency"]
            if heating_fuel == "gas":
                delta_gas = replaced
                delta_elec = -elec_added  # negative saving: electricity use rises
            else:
                delta_elec = replaced - elec_added  # electric resistance heat replaced by SCOP-times-better heat
            # Higher SCOP -> less electricity added -> larger net saving; the band follows the SCOP range.
            band = (replaced - useful_heat / hp["efficiency_low"], replaced - useful_heat / hp["efficiency_high"])
            current_heat_demand = elec_added  # heating is now delivered by heat-pump electricity

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
        low, high = min(low, tot_kwh), max(high, tot_kwh)  # the band always contains the point estimate
        tot_low, tot_high = tot_low + low, tot_high + high
        carbon_cut = calculate_carbon_kg(delta_elec, delta_gas)
        cost_range = RETROFIT_CATALOGUE[r]["indicative_cost"]

        bill_gbp = delta_elec * prices["electricity"] + delta_gas * prices["gas"]
        capped = r == "heat_pump" and insulate_first and "loft_insulation" not in ordered_retrofits
        suitability, payback = _suitability(bill_gbp, carbon_cut, cost_range, capped)

        recs.append(RetrofitRec(
            retrofit=r,
            suitability=suitability,  # type: ignore
            reason=REASONS[r] + (" " + METHOD_NOTE[method].format(model=model_label) if method else ""),
            marginal_saving=FuelDelta(elec_kwh=round(delta_elec, 1), gas_kwh=round(delta_gas, 1),
                                      total_kwh=round(tot_kwh, 1), low=round(low, 1), high=round(high, 1)),
            carbon_kg=round(carbon_cut, 1),
            indicative_cost=cost_range,
            bill_saving_gbp=round(bill_gbp, 0),
            payback_years=None if payback is None else round(payback, 1),
        ))
    rank_recommendations(recs)

    package_kwh = tot_saved_elec + tot_saved_gas
    package_saving = FuelDelta(elec_kwh=round(tot_saved_elec, 1), gas_kwh=round(tot_saved_gas, 1),
                               total_kwh=round(package_kwh, 1), low=round(tot_low, 1), high=round(tot_high, 1))
    package_carbon = calculate_carbon_kg(tot_saved_elec, tot_saved_gas)

    elements_before = build_home_elements(p)
    elements_after = build_home_elements(apply_retrofits_to_property(p, ordered_retrofits))
    hlc = lambda els: round(sum(e.heat_loss_w_per_k for e in els), 1) if els else None
    version = model.version + ("_fallback" if fallback else "") + ("+mode1" if forecast_used else "")
    # Calibration k = this home's use / the model's expected use for it (heating fuel), when it has bills.
    k = None
    if not fallback and heating_fuel in ("gas", "electricity") and not baseline_flags:
        expected = pred_now.gas_kwh if heating_fuel == "gas" else pred_now.elec_kwh
        actual = base_gas if heating_fuel == "gas" else base_elec
        k = round(float(actual / expected), 3) if np.isfinite(expected) and expected > 0 else None

    # The model's 80% band for this home's use, shown when its use is estimated rather than billed.
    bands = {}
    if not fallback and baseline_flags:
        for f in ("elec", "gas"):
            if f"{f}_estimated_from_similar_homes" in baseline_flags and f"{f}_p10" in pred_now:
                lo, hi = pred_now[f"{f}_p10"], pred_now[f"{f}_p90"]
                if np.isfinite(lo) and np.isfinite(hi):
                    bands.update({f"{f}_p10": round(float(lo), 1), f"{f}_p90": round(float(hi), 1)})

    energy_before = EnergyBreakdown(**bands,
        elec_kwh=round(base_elec, 1), gas_kwh=round(base_gas, 1),
        heat_demand_kwh=round(gas_heat_demand + elec_heat_demand, 1),
        base_load_kwh=round(gas_base_load + elec_base_load, 1),
        hlc_w_per_k=hlc(elements_before), calibration_k=k, calibrated=k is not None, model_version=version,
        elements=elements_before,
    )
    energy_after = EnergyBreakdown(
        elec_kwh=round(current_elec, 1), gas_kwh=round(current_gas, 1),
        heat_demand_kwh=round(current_heat_demand, 1),
        base_load_kwh=round(gas_base_load + elec_base_load, 1),
        hlc_w_per_k=hlc(elements_after), calibration_k=k, calibrated=k is not None, model_version=version,
        elements=elements_after,
    )
    return energy_before, energy_after, recs, package_saving, package_carbon
