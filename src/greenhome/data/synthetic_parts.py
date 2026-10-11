"""Synthetic parts for the real EPC property book (ARCHITECTURE 3.1): monthly bills, customer_id,
smart-controls level and LTV. These are the only fields with no public source; real EPC fields are never
altered. Seeded, and every rule lives in config/synthetic.yaml with a source.

    uv run python -m greenhome.data.synthetic_parts --seed 42      # after `make epc`

Bills follow the same hidden process as the interim generator (generate.EnergyWorld): NEED log-linear demand
for the home's cell x hidden occupancy / behaviour latents x smart-control multiplier, spread over the series
with real regional HDD; electric heating via efficiency or SCOP; solar via self-consumption. It is vectorised
here because the book has ~540k homes. Homes the NEED model can't place (unknown type, age, floor area or
heating) get no bills, so they exercise the "unavailable" path.

Outputs (gitignored), data/processed/real/:
  properties.parquet           real homes + annualised bills, customer_id, smart controls (incl. postcode)
  properties_share.parquet     same without postcode, safe to share on Drive
  monthly_energy/part-*.parquet
  loans.parquet, splits.parquet
  _hidden/latents.parquet      evaluation only, never features
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from greenhome.data.annualise import annualise_dataset
from greenhome.data.generate import ELECTRIC_HEATING, GAS_HEATING, area_band, need_epc
from greenhome.models.expected_use import AGE_CODES

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "config"
SRC = ROOT / "data" / "processed" / "epc_properties.parquet"
OUT = ROOT / "data" / "processed" / "real"
SUMMARY = ROOT / "eval" / "results" / "real_synthetic_summary.json"
CHUNK = 50_000


def load_config() -> tuple[dict, dict, pd.DataFrame]:
    syn = yaml.safe_load((CONFIG / "synthetic.yaml").read_text())
    need = yaml.safe_load((CONFIG / "need_calibration.yaml").read_text())
    hdd = pd.read_csv(CONFIG / "degree_days.csv", comment="#")
    return syn, need, hdd


def assign_smart_controls(x: pd.DataFrame, rng: np.random.Generator, syn: dict) -> pd.DataFrame:
    """EPC records none / programmer / zone control but has no smart-thermostat field. Upgrade a share of
    programmer homes to smart_thermostat so the book matches the documented market share."""
    target = syn["smart_controls"]["shares"]["smart_thermostat"]
    prog = x.smart_controls.eq("programmer").to_numpy()
    up = prog & (rng.random(len(x)) < min(1.0, target / max(prog.mean(), 1e-9)))
    x.loc[up, "smart_controls"] = "smart_thermostat"
    x.loc[up, "data_flags"] = x.loc[up, "data_flags"].str.replace(
        "smart_controls_from_epc", "smart_controls_synthetic", regex=False)
    return x


def assign_customers(x: pd.DataFrame, rng: np.random.Generator, syn: dict) -> pd.DataFrame:
    """One customer per home, except a share of private-rented homes grouped into landlord portfolios
    within the same local authority."""
    lp = syn["real_homes"]["landlord_portfolios"]
    lo, hi = lp["homes_per_landlord"]
    owner = np.arange(len(x))
    rented = x.tenure.eq("rented (private)").to_numpy()
    la = x.local_authority.to_numpy()
    for name in np.unique(la):
        idx = np.flatnonzero(rented & (la == name))
        sel = rng.permutation(idx[rng.random(len(idx)) < lp["share_of_private_rented"]])
        i = 0
        while i < len(sel):
            size = int(rng.integers(lo, hi + 1))
            owner[sel[i:i + size]] = sel[i]
            i += size
    _, inv = np.unique(owner, return_inverse=True)
    ids = rng.permutation(inv.max() + 1)[inv] + 1  # shuffle so IDs don't encode LA or order
    x["customer_id"] = [f"C{i:07d}" for i in ids]
    return x


def draw_latents(n: int, need: dict, syn: dict, rng: np.random.Generator) -> pd.DataFrame:
    """Hidden per-home factors; the NEED residual variance is split across them (as in generate.py)."""
    c = syn["consumption"]
    gas_sd = math.sqrt(max(need["gas_kwh_gas_heated"]["resid_sd"] ** 2 - c["occupancy_log_sd"] ** 2
                           - c["behaviour_log_sd"] ** 2, 0.01))
    elec_sd = math.sqrt(max(need["elec_kwh_gas_heated"]["resid_sd"] ** 2 - c["occupancy_log_sd"] ** 2, 0.01))
    return pd.DataFrame({
        "occupancy": np.exp(rng.normal(0, c["occupancy_log_sd"], n)),
        "behaviour": np.exp(rng.normal(0, c["behaviour_log_sd"], n)),
        "idio_gas": np.exp(rng.normal(0, gas_sd, n)),
        "idio_elec": np.exp(rng.normal(0, elec_sd, n)),
        "heat_pump_scop": rng.uniform(*c["electric_heating_efficiency"]["heat_pump_scop"], n),
    })


class BillWorld:
    """Vectorised generate.EnergyWorld: same config, same equations, many homes at once."""

    def __init__(self, syn: dict, need: dict, hdd: pd.DataFrame, months: pd.DatetimeIndex):
        self.c, self.solar, self.need = syn["consumption"], syn["solar"], need
        self.smart_mult = syn["smart_controls"]["heating_multiplier"]
        normal = (hdd[(hdd.year >= 2015) & (hdd.year <= 2024)].groupby(["local_authority", "year"]).hdd.sum()
                  .groupby("local_authority").mean())
        h = hdd.set_index(["local_authority", "year", "month"]).hdd
        self.weight = {la: np.array([h[(la, m.year, m.month)] for m in months]) / normal[la] for la in normal.index}
        month_num = np.array([m.month for m in months])
        self.seasonal = 1 + self.c["elec_seasonal_amplitude"] * np.cos(2 * np.pi * (month_num - 1) / 12)
        self.solar_share = np.array(self.solar["monthly_share"])[month_num - 1]

    def _need_kwh(self, model: dict, cell: pd.DataFrame) -> np.ndarray:
        e = model["effects"]
        log_kwh = (model["intercept"] + cell.ptype.map(e["PROP_TYPE"]) + cell.area.map(e["FLOOR_AREA_BAND"])
                   + cell.age.map(e["PROP_AGE_BAND"]) + cell.epc.map(e["EPC"]))
        return np.exp(log_kwh.to_numpy(dtype=float))

    def monthly(self, x: pd.DataFrame, lat: pd.DataFrame, rng: np.random.Generator | None):
        """(elec_import, gas, placeable): arrays shaped (homes, months); placeable is False where the NEED
        model can't place the home or its heating is unknown."""
        cell = pd.DataFrame({
            "ptype": x.property_type.to_numpy(),
            "area": [area_band(a) if pd.notna(a) else None for a in x.floor_area_m2],
            "age": x.age_band.map(AGE_CODES).to_numpy(),
            "epc": [need_epc(b) if isinstance(b, str) else None for b in x.energy_rating],
        })
        occ, beh = lat.occupancy.to_numpy(), lat.behaviour.to_numpy()
        gas_equiv = self._need_kwh(self.need["gas_kwh_gas_heated"], cell)
        appliances = self._need_kwh(self.need["elec_kwh_gas_heated"], cell) * occ * lat.idio_elec.to_numpy()
        base = gas_equiv * self.c["gas_base_share"] * occ * lat.idio_gas.to_numpy()
        heat = (gas_equiv * (1 - self.c["gas_base_share"]) * beh * lat.idio_gas.to_numpy()
                * x.smart_controls.map(self.smart_mult).to_numpy(dtype=float))

        w = np.stack([self.weight[la] for la in x.local_authority])
        heat_m, base_m = heat[:, None] * w, (base / 12)[:, None]
        elec = (appliances / 12)[:, None] * self.seasonal
        heating = x.heating.to_numpy()
        gas_heated, electric = np.isin(heating, list(GAS_HEATING)), np.isin(heating, list(ELECTRIC_HEATING))
        # A gas meter in a home not heated by gas carries the base load (hot water + cooking) on gas rather
        # than a fabricated 0. Never happens in the interim generator, where gas meter <=> gas heating.
        base_on_gas = x.gas_meter.to_numpy(dtype=bool) & ~gas_heated
        gas = np.where(gas_heated[:, None], heat_m + base_m, np.where(base_on_gas[:, None], base_m, 0.0))
        e = self.c["electric_heating_efficiency"]
        eff = np.where(heating == "heat_pump", lat.heat_pump_scop.to_numpy(),
                       np.where(heating == "storage_heaters", e["storage_heaters"], e["direct_electric"]))
        elec_base = np.where(base_on_gas[:, None], 0.0, base_m)
        elec = elec + np.where(electric[:, None], (heat_m + elec_base) * self.c["boiler_efficiency_gas"] / eff[:, None], 0.0)
        gen = (x.solar_kwp.to_numpy() * x.local_authority.map(self.solar["yield_kwh_per_kwp"]).to_numpy())[:, None]
        elec = elec - np.minimum(self.solar["self_consumption"] * gen * self.solar_share, 0.9 * elec)
        if rng is not None:
            sd = self.c["meter_noise_sd"]
            elec = elec * np.exp(rng.normal(0, sd, elec.shape))
            gas = gas * np.exp(rng.normal(0, sd, gas.shape))
        placeable = ~np.isnan(gas_equiv) & ~np.isnan(appliances) & (heating != "unknown")
        return elec, gas, placeable


def missing_months(n_homes: int, n_months: int, miss: dict, rng: np.random.Generator) -> np.ndarray:
    """Boolean (homes, months) mask of dropped readings, as in generate.py (60% contiguous gaps)."""
    mask = np.zeros((n_homes, n_months), dtype=bool)
    for i in np.flatnonzero(rng.random(n_homes) < miss["homes_with_missing_months"]):
        k = int(rng.integers(1, miss["max_missing_months"] + 1))
        if rng.random() < 0.6:
            s = int(rng.integers(0, n_months - k + 1))
            mask[i, s:s + k] = True
        else:
            mask[i, rng.choice(n_months, k, replace=False)] = True
    return mask


def make_loans(x: pd.DataFrame, rng: np.random.Generator, syn: dict) -> pd.DataFrame:
    share = syn["real_homes"]["mortgaged_share_by_tenure"]
    mortgaged = rng.random(len(x)) < x.tenure.map(share).fillna(share["unknown"]).to_numpy()
    loans = x.loc[mortgaged, ["property_id", "customer_id"]].copy()
    loans["ltv"] = rng.uniform(*syn["loans"]["ltv_range"], mortgaged.sum()).round(3)
    # The balance needs a current valuation (Price Paid + HPI step): unavailable until then, never guessed.
    loans["outstanding_balance"] = (loans.ltv * x.loc[mortgaged, "current_value_estimate"]).round(-2)
    return loans


def build(seed: int = 42) -> dict[str, pd.DataFrame]:
    syn, need, hdd = load_config()
    rng = np.random.default_rng(seed)
    x = pd.read_parquet(SRC)
    x = assign_smart_controls(x, rng, syn)
    x = assign_customers(x, rng, syn)
    lat = draw_latents(len(x), need, syn, rng)

    months = pd.date_range(syn["series_start"], periods=syn["n_months"], freq="MS")
    world, miss = BillWorld(syn, need, hdd, months), syn["missingness"]
    props, monthly_parts = [], []
    lat["gas_meter_unmatched"], lat["missing_months"], lat["has_bills"] = False, 0, False
    for k, start in enumerate(range(0, len(x), CHUNK)):
        xc, lc = x.iloc[start:start + CHUNK].copy(), lat.iloc[start:start + CHUNK]
        crng = np.random.default_rng([seed, k])
        elec, gas, ok = world.monthly(xc, lc, crng)
        unmatched = np.isin(xc.heating, list(GAS_HEATING)) & (crng.random(len(xc)) < miss["gas_meter_unmatched"])
        gas[unmatched] = np.nan
        drop = missing_months(len(xc), len(months), miss, crng)
        ii, jj = np.nonzero(~drop & ok[:, None])
        part = pd.DataFrame({"property_id": xc.property_id.to_numpy()[ii], "month": months[jj].date,
                             "elec_import_kwh": elec[ii, jj].round(1), "gas_kwh": gas[ii, jj].round(1)})
        no_bills = ~ok
        xc.loc[no_bills, "data_flags"] = xc.loc[no_bills, "data_flags"].map(
            lambda f: f"{f},no_synthetic_bills_unknown_cell" if f else "no_synthetic_bills_unknown_cell")
        props.append(annualise_dataset(xc, part))  # Person B's annualisation, unchanged
        monthly_parts.append(part)
        idx = lat.index[start:start + CHUNK]
        lat.loc[idx, "gas_meter_unmatched"] = unmatched
        lat.loc[idx, "missing_months"] = (drop & ok[:, None]).sum(axis=1)
        lat.loc[idx, "has_bills"] = ok
    properties = pd.concat(props)
    lat.insert(0, "property_id", x.property_id.to_numpy())

    split = syn["customers"]["split"]
    customers = np.sort(properties.customer_id.unique())
    splits = pd.DataFrame({"customer_id": customers,
                           "split": rng.choice(list(split), size=len(customers), p=list(split.values()))})
    return {"properties": properties, "monthly_parts": monthly_parts, "loans": make_loans(properties, rng, syn),
            "splits": splits, "latents": lat}


def summarise(t: dict) -> dict:
    p = t["properties"]
    per_customer = p.customer_id.value_counts()
    gas_heated = p.heating.isin(GAS_HEATING)
    return {
        "homes": len(p), "homes_with_bills": int(t["latents"].has_bills.sum()),
        "monthly_rows": int(sum(len(m) for m in t["monthly_parts"])),
        "annual_elec_available_share": round(float(p.annual_elec_kwh.notna().mean()), 4),
        "annual_gas_available_share_gas_heated": round(float(p.loc[gas_heated, "annual_gas_kwh"].notna().mean()), 4),
        "median_annual_kwh_by_heating": {h: {f: (None if pd.isna(v := g[f"annual_{f}_kwh"].median()) else round(float(v)))
                                                 for f in ("elec", "gas")}
                                             for h, g in p.groupby("heating")},
        "customers": len(per_customer), "multi_home_customers": int((per_customer > 1).sum()),
        "homes_in_multi_home_customers": int(per_customer[per_customer > 1].sum()),
        "smart_controls": p.smart_controls.value_counts(normalize=True).round(4).to_dict(),
        "loans": len(t["loans"]), "split_customers": t["splits"].split.value_counts().to_dict(),
    }


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    t = build(args.seed)
    (OUT / "monthly_energy").mkdir(parents=True, exist_ok=True)
    (OUT / "_hidden").mkdir(exist_ok=True)
    for old in (OUT / "monthly_energy").glob("part-*.parquet"):
        old.unlink()
    for k, part in enumerate(t["monthly_parts"]):
        part.to_parquet(OUT / "monthly_energy" / f"part-{k:03d}.parquet", index=False)
    t["properties"].to_parquet(OUT / "properties.parquet", index=False)
    t["properties"].drop(columns=["postcode"]).to_parquet(OUT / "properties_share.parquet", index=False)
    t["loans"].to_parquet(OUT / "loans.parquet", index=False)
    t["splits"].to_parquet(OUT / "splits.parquet", index=False)
    t["latents"].to_parquet(OUT / "_hidden" / "latents.parquet", index=False)
    summary = summarise(t)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
