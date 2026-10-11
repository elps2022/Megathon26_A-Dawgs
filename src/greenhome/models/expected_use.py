"""Expected annual use for similar homes, from the NEED log-linear model (config/need_calibration.yaml).

Used when a home has no usable bills (ARCHITECTURE 3.4/3.5): the estimate is flagged, never silent. Returns
None for a fuel when the home's description is too incomplete to place it in a NEED cell, so callers can report
"unavailable" instead of inventing a number. Interim until the tuned V1/V2 energy models are wired in.
"""

from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path

import yaml

from greenhome.schema import Property

_CFG = Path(__file__).resolve().parents[3] / "config" / "need_calibration.yaml"
AGE_CODES = {"pre-1930": "1", "1930-1972": "2", "1973-1999": "3", "2000+": "4"}
GAS_HEATING = {"gas_condensing", "gas_non_condensing"}


@lru_cache(maxsize=1)
def _need() -> dict:
    return yaml.safe_load(_CFG.read_text())


def _cell(p: Property) -> tuple[str, str, str, str] | None:
    if p.floor_area_m2 is None or p.energy_rating is None or p.age_band not in AGE_CODES:
        return None
    a = p.floor_area_m2
    area_band = "1" if a <= 50 else "2" if a <= 100 else "3" if a <= 150 else "4" if a <= 200 else "5"
    epc = {"A": "A/B", "B": "A/B", "F": "F/G", "G": "F/G"}.get(p.energy_rating.upper(), p.energy_rating.upper())
    return p.property_type, area_band, AGE_CODES[p.age_band], epc


def _predict(model: dict, cell: tuple[str, str, str, str], smear: bool = True) -> float | None:
    """Expected (mean) annual kWh for the cell: exp(log prediction) is the typical home, which sits below the
    average because use is skewed, so it is scaled by the model's smearing factor (Duan) to keep savings,
    carbon and portfolio totals unbiased."""
    e = model["effects"]
    try:
        log_kwh = (model["intercept"] + e["PROP_TYPE"][cell[0]] + e["FLOOR_AREA_BAND"][cell[1]]
                   + e["PROP_AGE_BAND"][cell[2]] + e["EPC"][cell[3]])
    except KeyError:  # property type or band the NEED model has never seen
        return None
    return round(math.exp(log_kwh) * (model["smearing_factor"] if smear else 1.0), 1)


def expected_annual_kwh(p: Property, statistic: str = "mean") -> dict[str, float | None]:
    """{"elec": kWh | None, "gas": kWh | None}. Gas is 0.0 (a real zero) for homes without a gas supply.
    statistic="mean" (default) is the expected use for totals and savings; "median" is the typical similar
    home, which the score compares a home against (ratio 1.0 = typical)."""
    smear = statistic == "mean"
    cell = _cell(p)
    need = _need()
    gas_heated = p.gas_meter and p.heating in GAS_HEATING
    elec = None if cell is None else _predict(
        need["elec_kwh_gas_heated" if gas_heated else "elec_kwh_not_gas_heated"], cell, smear)
    if not p.gas_meter:
        gas = 0.0
    else:
        gas = None if cell is None else _predict(need["gas_kwh_gas_heated"], cell, smear)
    return {"elec": elec, "gas": gas}
