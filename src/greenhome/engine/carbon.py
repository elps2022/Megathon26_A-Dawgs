import csv
from pathlib import Path

_CFG = Path(__file__).resolve().parents[3] / "config" / "emission_factors.csv"

def load_factors(path=_CFG) -> dict:
    with open(path) as f:
        return {r["fuel"]: float(r["kg_co2e_per_kwh"]) for r in csv.DictReader(f)}

def carbon_saved_kg(elec_saved_kwh: float, gas_saved_kwh: float, factors: dict | None = None) -> float:
    f = factors or load_factors()
    return elec_saved_kwh * f["electricity"] + gas_saved_kwh * f["gas"]

# Convenience aliases and defaults for Person B engine
try:
    DEFAULT_FACTORS = load_factors()
    DEFAULT_EF_ELEC = DEFAULT_FACTORS["electricity"]
    DEFAULT_EF_GAS = DEFAULT_FACTORS["gas"]
except Exception:
    DEFAULT_EF_ELEC = 0.20733
    DEFAULT_EF_GAS = 0.20275

calculate_carbon_kg = carbon_saved_kg

