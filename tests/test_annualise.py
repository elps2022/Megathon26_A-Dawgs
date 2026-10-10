"""Tests for annualise.py."""

from greenhome.data.annualise import annualise_series, annualise_dataset
import pandas as pd


def test_annualise_full_year() -> None:
    # 12 months uniform elec 300 kWh/mo = 3600 total
    months = list(range(1, 13))
    vals = [300.0] * 12
    annual, qual = annualise_series(months, vals, "elec")
    assert qual == "good"
    assert round(annual) == 3600


def test_annualise_winter_gas_scale() -> None:
    # Winter months (Dec, Jan, Feb) only: total 6000 kWh
    # Should scale properly to a whole year, not just sum
    months = [12, 1, 2]
    vals = [2000.0, 2500.0, 1500.0]
    annual, qual = annualise_series(months, vals, "gas")
    assert qual == "low_coverage"  # < 6 months
    # Share of Dec+Jan+Feb = 0.145 + 0.165 + 0.145 = 0.455
    # 6000 / 0.455 ≈ 13186 kWh
    assert annual > 10000.0


def test_annualise_missing_gas() -> None:
    annual, qual = annualise_series([], [], "gas")
    assert annual is None
    assert qual == "unavailable"
