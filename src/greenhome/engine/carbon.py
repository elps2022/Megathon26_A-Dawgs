import csv
from pathlib import Path

_CFG = Path(__file__).resolve().parents[3] / "config" / "emission_factors.csv"

def load_factors(path=_CFG) -> dict:
    with open(path) as f:
        return {r["fuel"]: float(r["kg_co2e_per_kwh"]) for r in csv.DictReader(f)}

def carbon_saved_kg(elec_saved_kwh: float, gas_saved_kwh: float, factors: dict | None = None) -> float:
    f = factors or load_factors()
    return elec_saved_kwh * f["electricity"] + gas_saved_kwh * f["gas"]
