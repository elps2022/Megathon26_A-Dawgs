"""Heat-loss physics: the grey-box energy model's step 1 and the physics-only baseline (ARCHITECTURE 3.5).

One vectorised implementation serves three callers, so they can never disagree:
  - NEED homes (training / evaluation): bands -> fabric via age-band defaults (need_fabric)
  - EPC homes and Property objects (deployment, what-ifs): their own fabric fields (property_fabric)
  - the engine's per-element heat loss for the house diagram (elements)

  HLC (W/K) = sum(U x A) over roof, walls, windows, floor + 0.33 x ACH x volume
  space heat (kWh/yr) = HLC x HDD_typical x 24 / 1000          (base 15.5 C already allows for gains)
  hot water (kWh/yr)  = SAP Appendix J (occupancy from floor area)
  lighting + appliances (kWh/yr electricity) = SAP Appendix L
Fuel = useful heat / heating efficiency (config/efficiencies.yaml; NEED gas homes use a mixed boiler efficiency).
Every constant lives in config/physics.yaml, geometry.yaml, u_values.yaml, efficiencies.yaml or region_hdd.csv.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"
ELEMENTS = ("roof", "wall", "window", "floor")
GAS_HEATING = {"gas_condensing", "gas_non_condensing"}
ELECTRIC_HEATING = {"storage_heaters", "direct_electric", "heat_pump"}
NEED_AGE = {1: "pre-1930", 2: "1930-1972", 3: "1973-1999", 4: "2000+"}
FABRIC_COLS = ["property_type", "age_band", "floor_area_m2", "has_own_roof", "loft_insulation_mm", "wall_type",
               "wall_insulation", "glazing"]


@lru_cache(maxsize=None)
def cfg(name: str) -> dict:
    return yaml.safe_load((CONFIG / name).read_text())


@lru_cache(maxsize=1)
def region_hdd() -> dict[str, float]:
    r = pd.read_csv(CONFIG / "region_hdd.csv", comment="#")
    return dict(zip(r.region_code, r.typical_hdd))


def _fill_defaults(f: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Unknown fabric fields -> the age band's as-built default. Returns (fabric, list of filled field names)."""
    defaults = cfg("physics.yaml")["age_band_defaults"]
    filled = pd.Series([[] for _ in range(len(f))], index=f.index)
    for col in ("wall_type", "wall_insulation", "loft_insulation_mm", "glazing"):
        unknown = f[col].isna() | f[col].isin(["unknown", ""])
        if col == "loft_insulation_mm":
            unknown &= f.has_own_roof.astype(bool)   # no roof of its own: not applicable, not unknown
        d = f.age_band.map(lambda a: defaults.get(a, defaults["1930-1972"])[col])
        f.loc[unknown, col] = d[unknown]
        for i in f.index[unknown]:
            filled[i].append(col)
    return f, filled


def need_fabric(x: pd.DataFrame) -> pd.DataFrame:
    """NEED parity features (need_data.features) -> fabric inputs, from age-band defaults."""
    p = cfg("physics.yaml")
    f = pd.DataFrame(index=x.index)
    f["property_type"] = x.prop_type.astype(str)
    f["age_band"] = x.age_band.map(NEED_AGE)
    f["floor_area_m2"] = x.floor_area_band.map(p["need_floor_area_midpoint_m2"]).astype(float)
    f["has_own_roof"] = (f.property_type != "Flat") | p["need_flats_have_own_roof"]
    for col in ("wall_type", "wall_insulation", "loft_insulation_mm", "glazing"):
        f[col] = np.nan
    f = f.astype({"wall_type": object, "wall_insulation": object, "glazing": object})
    f, _ = _fill_defaults(f)
    if "cavity_installed" in x:
        f.loc[(x.cavity_installed == 1) & (f.wall_type == "cavity"), "wall_insulation"] = p["cavity_installed_wall_insulation"]
    f["region_code"] = x.region.astype(str)
    return f


def property_fabric(props: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """EPC homes / Property rows -> fabric inputs from their own fields; unknowns -> age default (flagged)."""
    codes = cfg("epc_mapping.yaml")["need_region_codes"]
    f = props[FABRIC_COLS].copy()
    f = f.astype({"wall_type": object, "wall_insulation": object, "glazing": object})
    f["loft_insulation_mm"] = pd.to_numeric(f.loft_insulation_mm, errors="coerce")
    f["floor_area_m2"] = pd.to_numeric(f.floor_area_m2, errors="coerce")
    f["has_own_roof"] = f.has_own_roof.fillna(True).astype(bool)
    f, filled = _fill_defaults(f)
    f["region_code"] = props.region.map(codes)
    return f, filled


def envelope(f: pd.DataFrame) -> pd.DataFrame:
    """Per-element area (m2), U (W/m2K) and loss (W/K), ventilation loss and HLC. NaN where floor area unknown."""
    geo, u = cfg("geometry.yaml"), cfg("u_values.yaml")
    p = cfg("physics.yaml")
    area = f.floor_area_m2.astype(float)
    is_flat = f.property_type.astype(str).str.lower().str.contains("flat")
    storeys = np.where(is_flat, geo["storeys"]["flat"], geo["storeys"]["house"])
    footprint = area / storeys
    roof_area = np.where(f.has_own_roof.astype(bool), footprint, 0.0)
    window_area = geo["glazing_ratio_of_floor_area"] * area
    wall_area = np.maximum(4.0 * np.sqrt(footprint) * geo["storey_height_m"] * storeys - window_area,
                           geo["min_net_wall_area_m2"])
    t = geo["loft_thresholds_mm"]
    mm = f.loft_insulation_mm.astype(float)
    u_roof = np.where(~f.has_own_roof.astype(bool) | (mm >= t["standard"]), u["roof"]["standard"],
                      np.where(mm >= t["partial"], u["roof"]["partial"], u["roof"]["uninsulated"]))
    ins = f.wall_insulation.map({"full": "insulated", "partial": "partial"}).fillna("uninsulated")
    u_wall = (np.where(f.wall_type == "solid", "solid_", "cavity_") + ins).map(u["wall"]).astype(float)
    u_win = f.glazing.map(u["window"]).astype(float)
    out = pd.DataFrame(index=f.index)
    for name, a, uv in (("roof", roof_area, u_roof), ("wall", wall_area, u_wall), ("window", window_area, u_win),
                        ("floor", footprint, u["floor"]["standard"])):
        out[f"{name}_area_m2"] = a
        out[f"{name}_u"] = uv
        out[f"{name}_w_per_k"] = np.asarray(a, dtype=float) * np.asarray(uv, dtype=float)
    out["ventilation_w_per_k"] = p["ventilation_w_per_m3k"] * p["air_changes_per_hour"] * area * geo["storey_height_m"]
    out["hlc_w_per_k"] = out[[f"{e}_w_per_k" for e in ELEMENTS]].sum(axis=1, min_count=4) + out.ventilation_w_per_k
    return out


def occupancy(area: pd.Series) -> pd.Series:
    """SAP Appendix J occupancy from total floor area."""
    a = area.astype(float) - 13.9
    n = 1 + 1.76 * (1 - np.exp(-0.000349 * a ** 2)) + 0.0013 * a
    return n.where(a > 0, 1.0)


def demands(f: pd.DataFrame) -> pd.DataFrame:
    """Useful annual demands (kWh/yr): space heat, hot water, and lighting + appliances electricity."""
    p = cfg("physics.yaml")
    env = envelope(f)
    hdd = f.region_code.map(region_hdd()).astype(float)
    area = f.floor_area_m2.astype(float)
    n = occupancy(area)
    hw = p["hot_water"]
    litres = hw["litres_per_person_per_day"] * n + hw["litres_fixed_per_day"]
    hot_water = 4.18 * litres * hw["mean_temperature_rise_k"] / 3600 * 365 * (1 + hw["distribution_loss_share"])
    ap = p["appliances"]
    tfa_n = (area * n) ** ap["exponent"]
    env["space_heat_kwh"] = env.hlc_w_per_k * hdd * 24 / 1000
    env["hot_water_kwh"] = hot_water
    env["lights_appliances_kwh"] = (ap["appliance_coefficient"] + ap["lighting_coefficient"] * ap["lighting_low_energy_factor"]) * tfa_n
    env["typical_hdd"] = hdd
    return env


def need_physics(x: pd.DataFrame) -> pd.DataFrame:
    """Physics-only annual gas and electricity for NEED rows (need_data features). Gas-heated homes burn gas for
    space heat + hot water at the mixed boiler efficiency. NEED doesn't say how non-gas homes heat, so their
    physics electricity is lighting + appliances only (the residual model learns the rest)."""
    d = demands(need_fabric(x))
    eff = cfg("physics.yaml")["need_gas_boiler_efficiency"]
    gas_heated = (x.gas_heated == 1) if "gas_heated" in x else pd.Series(True, index=x.index)
    d["gas_kwh"] = np.where(gas_heated, (d.space_heat_kwh + d.hot_water_kwh) / eff, 0.0)
    d["elec_kwh"] = d.lights_appliances_kwh
    d["gas_kwh_per_hdd"] = np.where(gas_heated, d.hlc_w_per_k * 24 / 1000 / eff, np.nan)
    return d


def property_physics(props: pd.DataFrame) -> pd.DataFrame:
    """Physics-only annual gas and electricity for EPC homes / Property rows, from their own fabric and heating.
    Electric heating adds (space heat + hot water) / efficiency to electricity; oil, LPG and district heat
    aren't metered, so they add nothing to either meter. Returns per-element losses too (house diagram)."""
    f, filled = property_fabric(props)
    d = demands(f)
    systems = cfg("efficiencies.yaml")["heating_systems"]
    heating = props.heating.astype(str)
    eff = heating.map(lambda h: systems.get(h, {}).get("efficiency", np.nan)).astype(float)
    useful = d.space_heat_kwh + d.hot_water_kwh
    gas_heat = heating.isin(GAS_HEATING) & props.gas_meter.astype(bool)
    elec_heat = heating.isin(ELECTRIC_HEATING)
    d["heating_fuel_kwh"] = useful / eff
    d["gas_kwh"] = np.where(gas_heat, d.heating_fuel_kwh, 0.0)
    d["elec_kwh"] = d.lights_appliances_kwh + np.where(elec_heat, d.heating_fuel_kwh, 0.0)
    d["heating_fuel_kwh_per_hdd"] = np.where(gas_heat | elec_heat, d.hlc_w_per_k * 24 / 1000 / eff, np.nan)
    d["fabric_defaulted"] = filled.map(",".join)
    return d


def elements(env_row: pd.Series) -> list[dict]:
    """Per-element heat loss for one home (Element fields), from an envelope() row."""
    total = sum(env_row[f"{e}_w_per_k"] for e in ELEMENTS) or 1.0
    return [{"name": e, "area_m2": round(float(env_row[f"{e}_area_m2"]), 1), "u_value": float(env_row[f"{e}_u"]),
             "heat_loss_w_per_k": round(float(env_row[f"{e}_w_per_k"]), 2),
             "share": round(float(env_row[f"{e}_w_per_k"]) / total, 3)} for e in ELEMENTS]
