"""Tests for schema.py contract."""

from pathlib import Path
import pandas as pd
import pytest
from greenhome.schema import (
    Element,
    EnergyBreakdown,
    FuelDelta,
    HomeState,
    Infeasible,
    PortfolioPropertySummary,
    PortfolioSummary,
    Property,
    PropertyReport,
    RetrofitRec,
    ScoreResult,
    SubScore,
)


def test_property_instantiation_minimal() -> None:
    prop = Property(
        property_id="P00001",
        customer_id="C00001",
        local_authority="Leeds",
        region="Yorkshire and The Humber",
        property_type="Semi detached",
        age_band="1930-1972",
        heating="gas_condensing",
        wall_type="cavity",
        glazing="double_new",
    )
    assert prop.property_id == "P00001"
    assert prop.solar_kwp == 0.0
    assert prop.smart_controls == "none"
    assert prop.has_own_roof is True
    assert prop.gas_meter is True


def test_property_from_generated_csv() -> None:
    csv_path = Path("data/generated/properties.csv")
    if not csv_path.exists():
        pytest.skip("Generated properties.csv not found")

    df = pd.read_csv(csv_path).head(100)
    for _, row in df.iterrows():
        d = row.to_dict()
        # Clean NaNs
        clean_d = {k: (None if pd.isna(v) else v) for k, v in d.items()}
        if clean_d.get("data_flags"):
            clean_d["data_flags"] = [f.strip() for f in str(clean_d["data_flags"]).split(",") if f.strip()]
        else:
            clean_d["data_flags"] = []
        prop = Property(**clean_d)
        assert prop.property_id.startswith("P")


def test_full_report_serialization() -> None:
    prop = Property(
        property_id="P10001",
        customer_id="C00100",
        local_authority="Southwark",
        region="London",
        property_type="Terraced",
        age_band="pre-1930",
        floor_area_m2=90.0,
        bedrooms=3,
        energy_rating="E",
        sap_score=46.0,
        heating="gas_non_condensing",
        wall_type="solid",
        wall_insulation="none",
        loft_insulation_mm=50.0,
        glazing="single",
        solar_kwp=0.0,
        smart_controls="none",
        current_value_estimate=450000.0,
        annual_elec_kwh=3200.0,
        annual_gas_kwh=14000.0,
    )

    subscores = [
        SubScore(component="Energy efficiency", weight=0.25, score=46.0, contribution=11.5),
        SubScore(component="Consumption intensity", weight=0.20, score=35.0, contribution=7.0),
        SubScore(component="Insulation", weight=0.15, score=25.0, contribution=3.75),
        SubScore(component="Heating", weight=0.15, score=35.0, contribution=5.25),
        SubScore(component="Glazing", weight=0.10, score=0.0, contribution=0.0),
        SubScore(component="Renewables", weight=0.10, score=0.0, contribution=0.0),
        SubScore(component="Smart controls", weight=0.05, score=0.0, contribution=0.0),
    ]

    score_before = ScoreResult(
        total=28,
        classification="Very poor",
        components=subscores,
    )
    score_after = ScoreResult(
        total=68,
        classification="Good",
        components=subscores,
    )

    rec = RetrofitRec(
        retrofit="loft_insulation_300mm",
        suitability="High",
        reason="Minimal existing loft insulation (50mm). High heat loss.",
        marginal_saving=FuelDelta(elec_kwh=0.0, gas_kwh=2100.0, total_kwh=2100.0, low=1800.0, high=2400.0),
        carbon_kg=384.3,
        indicative_cost=(600.0, 1100.0),
    )

    infeasible = Infeasible(
        retrofit="cavity_wall_insulation",
        rule="Property has solid masonry walls.",
    )

    energy_before = EnergyBreakdown(
        elec_kwh=3200.0,
        gas_kwh=14000.0,
        heat_demand_kwh=11900.0,
        base_load_kwh=2100.0,
        hlc_w_per_k=280.0,
        calibration_k=1.05,
        calibrated=True,
    )

    report = PropertyReport(
        property=prop,
        score=score_before,
        score_after=score_after,
        is_poor=True,
        key_issues=["single glazing", "uninsulated solid walls", "old non-condensing boiler"],
        improvement_potential_pct=42.5,
        recommendations=[rec],
        infeasible=[infeasible],
        package_saving=FuelDelta(elec_kwh=-600.0, gas_kwh=8500.0, total_kwh=7900.0, low=6800.0, high=8900.0),
        package_carbon_kg=1431.3,
        value={
            "baseline": 450000.0,
            "low": 463500.0,
            "high": 472500.0,
            "pct_low": 3.0,
            "pct_high": 5.0,
            "method": "LightGBM quantile hedonic pricing (p10-p90)",
            "caveat": "Indicative estimate based on observational market association, not a causal guarantee.",
        },
        energy_before=energy_before,
        energy_after=energy_before,
        assumptions=["Standard occupancy", "DESNZ 2024 emission factors"],
    )

    data = report.model_dump()
    assert data["is_poor"] is True
    assert len(data["recommendations"]) == 1
    assert data["value"]["pct_low"] == 3.0

    # Round trip
    reconstructed = PropertyReport.model_validate(data)
    assert reconstructed.property.property_id == "P10001"
