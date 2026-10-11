"""Interim synthetic dataset: EPC-like homes in Leeds, Cornwall and Southwark with monthly per-fuel bills,
retrofit events, sales, customers, loans and customer-level splits.

Reads only config/ (no raw data needed), so every teammate can regenerate the same data:

    uv run python -m greenhome.data.generate --seed 42

Outputs go to data/generated/. Files under data/generated/_hidden/ are ground truth for evaluation only and
must never be used as model features.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from greenhome.data.annualise import annualise_dataset

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"
OUT = ROOT / "data" / "generated"

AGE_LABELS = {"1": "pre-1930", "2": "1930-1972", "3": "1973-1999", "4": "2000+"}
HOUSE_TYPES = {"Detached", "Semi detached", "End terrace", "Mid terrace", "Bungalow"}
GAS_HEATING = {"gas_condensing", "gas_non_condensing"}
ELECTRIC_HEATING = {"storage_heaters", "direct_electric", "heat_pump"}


def load_config() -> tuple[dict, dict, dict, pd.DataFrame, pd.DataFrame]:
    syn = yaml.safe_load((CONFIG / "synthetic.yaml").read_text())
    need = yaml.safe_load((CONFIG / "need_calibration.yaml").read_text())
    price = yaml.safe_load((CONFIG / "price_calibration.yaml").read_text())
    hdd = pd.read_csv(CONFIG / "degree_days.csv", comment="#")
    hpi = pd.read_csv(CONFIG / "hpi_index.csv", comment="#", parse_dates=["date"])
    return syn, need, price, hdd, hpi


def pick(rng: np.random.Generator, dist: dict) -> str:
    keys = list(dist)
    p = np.array([dist[k] for k in keys], dtype=float)
    return keys[rng.choice(len(keys), p=p / p.sum())]


def need_predict(model: dict, ptype: str, area_band: str, age_band: str, epc_band: str) -> float:
    e = model["effects"]
    return math.exp(model["intercept"] + e["PROP_TYPE"][ptype] + e["FLOOR_AREA_BAND"][area_band]
                    + e["PROP_AGE_BAND"][age_band] + e["EPC"][epc_band])


def area_band(area: float) -> str:
    return "1" if area <= 50 else "2" if area <= 100 else "3" if area <= 150 else "4" if area <= 200 else "5"


def need_epc(band: str) -> str:
    return {"A": "A/B", "B": "A/B", "F": "F/G", "G": "F/G"}.get(band, band)


def sap_band(sap: float, bands: dict) -> str:
    return next(b for b, lo in bands.items() if sap >= lo)


def make_home(rng: np.random.Generator, syn: dict, la: str) -> dict:
    las, fab = syn["local_authorities"], syn["fabric"]
    ptype = pick(rng, las[la]["property_type"])
    age = pick(rng, las[la]["age_band"])
    area = float(np.clip(syn["floor_area_m2"]["median"][ptype] * math.exp(rng.normal(0, syn["floor_area_m2"]["log_sd"])), 25, 400))

    heat_dist = dict(syn["heating"]["Cornwall" if la == "Cornwall" else "gas_grid"])
    if ptype == "Flat":
        for k in ("storage_heaters", "direct_electric"):
            heat_dist[k] *= syn["heating"]["flat_electric_uplift"]
    heating = pick(rng, heat_dist)

    wall_type = "solid" if rng.random() < fab["solid_wall_share_by_age"][age] else "cavity"
    if age == "4":
        wall_ins = "full"
    elif rng.random() < fab["partial_share"]:
        wall_ins = "partial"
    else:
        filled = fab["cavity_filled_share"] if wall_type == "cavity" else fab["solid_insulated_share"]
        wall_ins = "full" if rng.random() < filled else "none"

    own_roof = ptype in HOUSE_TYPES or rng.random() < fab["flat_with_own_roof"]
    loft = None
    if own_roof:
        loft = int(rng.choice(fab["loft_mm"]["values"], p=fab["loft_mm"]["probs"]))
        if age == "4":
            loft = max(loft, 270)
    glazing = pick(rng, fab["glazing"])
    solar = 0.0
    if ptype in HOUSE_TYPES and rng.random() < fab["solar_share_houses"][la]:
        solar = round(float(rng.uniform(*fab["solar_kwp_range"])), 1)
    smart = pick(rng, syn["smart_controls"]["shares"])
    return dict(local_authority=la, property_type=ptype, age_code=age, floor_area_m2=round(area, 1),
                heating=heating, wall_type=wall_type, wall_insulation=wall_ins, has_own_roof=own_roof,
                loft_insulation_mm=loft, glazing=glazing, solar_kwp=solar, smart_controls=smart)


def true_sap(rng: np.random.Generator, h: dict, s: dict) -> float:
    pts = s["base_by_age"][h["age_code"]] + s["wall"][h["wall_insulation"]] + s["glazing"][h["glazing"]]
    pts += s["heating"][h["heating"]] + s["type"][h["property_type"]] + s["solar_points_per_kwp"] * h["solar_kwp"]
    if h["loft_insulation_mm"] is not None:
        pts += s["loft_max_points"] * min(h["loft_insulation_mm"], 270) / 270 - s["loft_max_points"] / 2
    return float(np.clip(pts + rng.normal(0, s["noise_sd"]), 1, 100))


class EnergyWorld:
    """Hidden data-generating process for one home's bills. Kept independent of the grey-box physics model."""

    def __init__(self, syn: dict, need: dict, hdd: pd.DataFrame):
        self.c, self.solar = syn["consumption"], syn["solar"]
        self.smart_mult = syn["smart_controls"]["heating_multiplier"]
        self.need = need
        normal = hdd[(hdd.year >= 2015) & (hdd.year <= 2024)].groupby(["local_authority", "year"]).hdd.sum()
        self.normal_hdd = normal.groupby("local_authority").mean().to_dict()
        self.hdd = hdd.set_index(["local_authority", "year", "month"]).hdd.to_dict()
        # Typical year (year key 0): mean monthly HDD over 2015-2024, used for true annual effects.
        typical = hdd[(hdd.year >= 2015) & (hdd.year <= 2024)].groupby(["local_authority", "month"]).hdd.mean()
        self.hdd.update({(la, 0, m): v for (la, m), v in typical.items()})

    def demand(self, h: dict, lat: dict) -> dict:
        """Typical-weather annual demand components (kWh), before fuel conversion."""
        args = (h["property_type"], area_band(h["floor_area_m2"]), h["age_code"], need_epc(h["true_band"]))
        gas_equiv = need_predict(self.need["gas_kwh_gas_heated"], *args)
        appliances = need_predict(self.need["elec_kwh_gas_heated"], *args) * lat["occupancy"] * lat["idio_elec"]
        base = gas_equiv * self.c["gas_base_share"] * lat["occupancy"] * lat["idio_gas"]
        heat = (gas_equiv * (1 - self.c["gas_base_share"]) * lat["behaviour"] * lat["idio_gas"]
                * self.smart_mult[h["smart_controls"]] * h["heat_mult"])
        return {"base": base, "heat": heat, "appliances": appliances}

    def month(self, h: dict, lat: dict, year: int, month: int, rng: np.random.Generator | None) -> tuple[float, float]:
        """(grid import kWh, gas kWh) for one month."""
        d = self.demand(h, lat)
        w = self.hdd[(h["local_authority"], year, month)] / self.normal_hdd[h["local_authority"]]
        heat_m, base_m = d["heat"] * w, d["base"] / 12
        amp = self.c["elec_seasonal_amplitude"]
        elec = d["appliances"] / 12 * (1 + amp * math.cos(2 * math.pi * (month - 1) / 12))
        gas = 0.0
        eff_gas = self.c["boiler_efficiency_gas"]
        if h["heating"] in GAS_HEATING:
            gas = heat_m + base_m
        elif h["heating"] in ELECTRIC_HEATING:
            eff = h["scop"] if h["heating"] == "heat_pump" else self.c["electric_heating_efficiency"][h["heating"]]
            elec += (heat_m + base_m) * eff_gas / eff
        if h["solar_kwp"] > 0:
            gen = h["solar_kwp"] * self.solar["yield_kwh_per_kwp"][h["local_authority"]] * self.solar["monthly_share"][month - 1]
            elec -= min(self.solar["self_consumption"] * gen, 0.9 * elec)
        if rng is not None:
            sd = self.c["meter_noise_sd"]
            elec *= math.exp(rng.normal(0, sd))
            gas *= math.exp(rng.normal(0, sd))
        return elec, gas

    def annual(self, h: dict, lat: dict) -> tuple[float, float]:
        """Typical-year (grid import, gas) without meter noise, used for true retrofit effects."""
        tot_e = tot_g = 0.0
        for m in range(1, 13):
            e, g = self.month(h, lat, 0, m, None)
            tot_e, tot_g = tot_e + e, tot_g + g
        return tot_e, tot_g


def candidate_retrofits(h: dict) -> list[str]:
    c = []
    if h["has_own_roof"] and h["loft_insulation_mm"] is not None and h["loft_insulation_mm"] < 150:
        c.append("loft_insulation")
    if h["wall_insulation"] == "none":
        c.append("cavity_wall_insulation" if h["wall_type"] == "cavity" else "solid_wall_insulation")
    if h["glazing"] in ("single", "partial", "double_old"):
        c.append("glazing_upgrade")
    if h["smart_controls"] in ("none", "programmer"):
        c.append("smart_thermostat")
    if h["heating"] in GAS_HEATING and h["wall_insulation"] != "none":
        c.append("heat_pump")
    if h["property_type"] in HOUSE_TYPES and h["solar_kwp"] == 0:
        c.append("solar_pv")
    return c


def apply_retrofit(rng: np.random.Generator, h: dict, retrofit: str, syn: dict, need: dict) -> dict:
    h = dict(h)
    r = syn["retrofit_events"]
    base_share = syn["consumption"]["gas_base_share"]
    if retrofit in ("loft_insulation", "cavity_wall_insulation", "solid_wall_insulation", "glazing_upgrade"):
        if retrofit in need["retrofit_effects"]:  # NEED effect is on total gas; rescale to the heating part
            mean = need["retrofit_effects"][retrofit]["median_fractional_change"] / (1 - base_share)
        else:
            mean = r["assumed_effects"][retrofit]
        eff = min(rng.normal(mean, r["effect_sd_fraction"] * abs(mean) + 0.02), 0.02)
        h["heat_mult"] *= 1 + eff
        if retrofit == "loft_insulation":
            h["loft_insulation_mm"] = 270
        elif retrofit == "glazing_upgrade":
            h["glazing"] = "double_new"
        else:
            h["wall_insulation"] = "full"
    elif retrofit == "smart_thermostat":
        h["smart_controls"] = "smart_thermostat"
    elif retrofit == "heat_pump":
        h["heating"] = "heat_pump"
        h["scop"] = float(rng.uniform(*syn["consumption"]["electric_heating_efficiency"]["heat_pump_scop"]))
    elif retrofit == "solar_pv":
        h["solar_kwp"] = round(float(rng.uniform(*syn["fabric"]["solar_kwp_range"])), 1)
    return h


def generate(seed: int = 42) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    syn, need, price, hdd, hpi = load_config()
    world = EnergyWorld(syn, need, hdd)
    c, miss = syn["consumption"], syn["missingness"]
    gas_sd = math.sqrt(max(need["gas_kwh_gas_heated"]["resid_sd"] ** 2 - c["occupancy_log_sd"] ** 2 - c["behaviour_log_sd"] ** 2, 0.01))
    elec_sd = math.sqrt(max(need["elec_kwh_gas_heated"]["resid_sd"] ** 2 - c["occupancy_log_sd"] ** 2, 0.01))
    months = pd.date_range(syn["series_start"], periods=syn["n_months"], freq="MS")

    n = syn["n_properties"]
    las = syn["local_authorities"]
    la_names = [k for k in las if k != "source"]
    la_draw = rng.choice(la_names, size=n, p=[las[k]["share"] for k in la_names])

    props, latents, monthly, events, truth = [], [], [], [], []
    for i in range(n):
        pid = f"P{i + 1:05d}"
        h = make_home(rng, syn, str(la_draw[i]))
        h["heat_mult"] = 1.0
        h["scop"] = float(rng.uniform(*c["electric_heating_efficiency"]["heat_pump_scop"]))
        sap = round(true_sap(rng, h, syn["sap"]))
        h["true_band"] = sap_band(sap, syn["sap"]["bands"])
        lat = {"occupancy": math.exp(rng.normal(0, c["occupancy_log_sd"])),
               "behaviour": math.exp(rng.normal(0, c["behaviour_log_sd"])),
               "idio_gas": math.exp(rng.normal(0, gas_sd)), "idio_elec": math.exp(rng.normal(0, elec_sd))}

        # Optional retrofit event part-way through the series.
        install_idx, h_after, retrofit = None, h, None
        cands = candidate_retrofits(h)
        if cands and rng.random() < syn["retrofit_events"]["share_of_homes"]:
            retrofit = str(rng.choice(cands))
            install_idx = int(rng.integers(*syn["retrofit_events"]["install_month_range"]))
            h_after = apply_retrofit(rng, h, retrofit, syn, need)
            e0, g0 = world.annual(h, lat)
            e1, g1 = world.annual(h_after, lat)
            events.append({"property_id": pid, "retrofit": retrofit, "install_month": months[install_idx].date()})
            truth.append({"property_id": pid, "retrofit": retrofit, "true_elec_change_kwh_yr": round(e1 - e0, 1),
                          "true_gas_change_kwh_yr": round(g1 - g0, 1)})

        # Monthly bills, with missingness.
        drop = set()
        if rng.random() < miss["homes_with_missing_months"]:
            k = int(rng.integers(1, miss["max_missing_months"] + 1))
            start = int(rng.integers(0, len(months) - k + 1))
            drop = set(range(start, start + k)) if rng.random() < 0.6 else set(rng.choice(len(months), k, replace=False).tolist())
        has_gas_meter = h["heating"] in GAS_HEATING
        gas_unmatched = has_gas_meter and rng.random() < miss["gas_meter_unmatched"]
        for j, m in enumerate(months):
            if j in drop:
                continue
            state = h_after if install_idx is not None and j >= install_idx else h
            e, g = world.month(state, lat, m.year, m.month, rng)
            gas_val = np.nan if gas_unmatched else (round(g, 1) if state["heating"] in GAS_HEATING else 0.0)
            monthly.append({"property_id": pid, "month": m.date(), "elec_import_kwh": round(e, 1), "gas_kwh": gas_val})

        # Observed EPC-like record (state at series start), with missing and conflicting fields.
        band, sap_obs, conflict = h["true_band"], round(sap), False
        if rng.random() < miss["no_epc"]:
            band, sap_obs = None, None
        elif band in ("D", "E", "F", "G") and rng.random() < miss["conflicting_fields"] * 2:
            band, sap_obs, conflict = "A", int(rng.integers(92, 97)), True
        area_obs = None if rng.random() < miss["floor_area_missing"] else h["floor_area_m2"]
        loft_obs = None if rng.random() < miss["loft_missing"] else h["loft_insulation_mm"]
        props.append({
            "property_id": pid, "local_authority": h["local_authority"], "region": hdd.loc[hdd.local_authority == h["local_authority"], "region"].iat[0],
            "property_type": h["property_type"], "age_band": AGE_LABELS[h["age_code"]], "floor_area_m2": area_obs,
            "bedrooms": int(np.clip(round(h["floor_area_m2"] / 28), 1, 6)), "energy_rating": band, "sap_score": sap_obs,
            "heating": h["heating"], "wall_type": h["wall_type"], "wall_insulation": h["wall_insulation"],
            "has_own_roof": h["has_own_roof"], "loft_insulation_mm": loft_obs, "glazing": h["glazing"],
            "solar_kwp": h["solar_kwp"], "smart_controls": h["smart_controls"], "gas_meter": has_gas_meter,
            "data_flags": "",
        })
        latents.append({"property_id": pid, **{k: round(v, 4) for k, v in lat.items()}, "true_sap": round(sap, 1),
                        "true_band": h["true_band"], "conflict_injected": conflict, "gas_meter_unmatched": gas_unmatched,
                        "true_floor_area_m2": h["floor_area_m2"], "heat_pump_scop": round(h["scop"], 2)})

    properties = pd.DataFrame(props)
    hidden = pd.DataFrame(latents)

    # Sales and current valuation (hedonic on real Price Paid levels, indexed by real UK HPI).
    s = syn["sales"]
    hpi = hpi.set_index(["local_authority", "date"]).hpi
    # Re-base from the years the Price Paid medians pool (config/price_calibration.yaml) to each sale / valuation date
    ref = {la: hpi[la][pd.to_datetime(hpi[la].index).year.isin(price["years"])].mean() for la in la_names}
    latest = hpi.index.get_level_values("date").max()
    sale_months = pd.date_range(*s["date_range"], freq="MS")
    quality = np.exp(rng.normal(0, s["quality_log_sd"], n))
    hidden["value_quality"] = quality.round(4)
    sales, values = [], []
    for i, p in properties.iterrows():
        ppd_t = s["type_to_ppd"][p.property_type]
        true_area = hidden.at[i, "true_floor_area_m2"]
        level = (price["median_price_by_type"][p.local_authority][ppd_t]
                 * (true_area / s["reference_area"][ppd_t]) ** s["area_elasticity"]
                 * (1 + s["rating_premium"][hidden.at[i, "true_band"]]) * quality[i])
        values.append(round(level * hpi[(p.local_authority, latest)] / ref[p.local_authority]
                            * math.exp(rng.normal(0, s["valuation_noise_log_sd"])), -3))
        if rng.random() < s["share_with_sale"]:
            for d in sorted(rng.choice(sale_months, 1 if rng.random() < 0.7 else 2, replace=False)):
                d = pd.Timestamp(d)
                sales.append({"property_id": p.property_id, "sale_date": d.date(),
                              "price": round(level * hpi[(p.local_authority, d)] / ref[p.local_authority]
                                             * math.exp(rng.normal(0, s["sale_noise_log_sd"])), -2)})
    properties["current_value_estimate"] = values

    # Customers (a few landlords hold several homes) and customer-level splits.
    cust = syn["customers"]
    n_cust = int(n * cust["customers_per_property"])
    owner = np.concatenate([np.arange(n_cust), rng.integers(0, n_cust, n - n_cust)])
    rng.shuffle(owner)
    properties.insert(1, "customer_id", [f"C{o + 1:05d}" for o in owner])
    cust_ids = np.array(sorted(properties.customer_id.unique()))
    split = rng.choice(list(cust["split"]), size=len(cust_ids), p=list(cust["split"].values()))
    splits = pd.DataFrame({"customer_id": cust_ids, "split": split})

    loans = properties[["property_id", "customer_id", "current_value_estimate"]].copy()
    loans = loans[rng.random(n) < syn["loans"]["share_mortgaged"]]
    loans["ltv"] = rng.uniform(*syn["loans"]["ltv_range"], len(loans)).round(3)
    loans["outstanding_balance"] = (loans.current_value_estimate * loans.ltv).round(-2)
    loans = loans.drop(columns="current_value_estimate")

    return {
        "properties": properties, "monthly_energy": pd.DataFrame(monthly), "retrofit_events": pd.DataFrame(events),
        "sales": pd.DataFrame(sales), "loans": loans, "splits": splits,
        "_hidden/latents": hidden, "_hidden/retrofit_truth": pd.DataFrame(truth),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    tables = generate(args.seed)
    # Annualise the monthly bills (ARCHITECTURE 3.2) so every consumer sees each home's own annual use + flags.
    tables["properties"] = annualise_dataset(tables["properties"], tables["monthly_energy"])
    (args.out / "_hidden").mkdir(parents=True, exist_ok=True)
    for name, df in tables.items():
        df.to_csv(args.out / f"{name}.csv", index=False)
        print(f"{name:24s} {len(df):>8,} rows")


if __name__ == "__main__":
    main()
