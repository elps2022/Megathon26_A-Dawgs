"""Comprehensive adversarial test suite for LLM grounding verification shield.

Validates the 50 evaluation scenarios defined in docs/LLM.md and docs/ARCHITECTURE.md 3.10:
- Catches hallucinated numbers and fabricated costs/savings.
- Detects sign errors (heat pump raising electricity vs lowering gas).
- Rejects ungrounded claims on disqualified retrofits.
- Validates 100% pass on legitimate engine-grounded outputs.
"""

import pytest
from greenhome.llm.grounding import (
    extract_numbers_from_json,
    extract_numbers_from_text,
    verify_grounding,
)
from greenhome.schema import (
    EnergyBreakdown,
    FuelDelta,
    Property,
    PropertyReport,
    RetrofitRec,
    ScoreResult,
    SubScore,
)


def build_mock_report() -> PropertyReport:
    """Standard verified PropertyReport baseline for grounding tests."""
    prop = Property(
        property_id="PROP_TEST_001",
        customer_id="CUST_001",
        local_authority="Leeds",
        region="Yorkshire",
        property_type="semi-detached",
        age_band="1950-1966",
        floor_area_m2=95.0,
        bedrooms=3,
        energy_rating="D",
        heating="gas_condensing",
        wall_type="cavity",
        wall_insulation="none",
        has_own_roof=True,
        loft_insulation_mm=100.0,
        glazing="double_old",
        annual_elec_kwh=3100.0,
        annual_gas_kwh=11500.0,
    )

    score = ScoreResult(
        total=58,
        classification="Moderate",
        components=[
            SubScore(component="Energy Efficiency", score=60.0, weight=0.25, contribution=15.0),
            SubScore(component="Heating", score=55.0, weight=0.20, contribution=11.0),
            SubScore(component="Insulation", score=45.0, weight=0.15, contribution=6.8),
        ],
    )

    recs = [
        RetrofitRec(
            retrofit="loft_insulation",
            suitability="High",
            reason="Top-up to 300mm standard",
            marginal_saving=FuelDelta(elec_kwh=0.0, gas_kwh=1800.0, total_kwh=1800.0, low=1530.0, high=2070.0),
            carbon_kg=365.0,
            indicative_cost=[500.0, 1100.0],
        ),
        RetrofitRec(
            retrofit="heat_pump",
            suitability="Medium",
            reason="Air source heat pump replacement",
            marginal_saving=FuelDelta(elec_kwh=-1400.0, gas_kwh=8500.0, total_kwh=7100.0, low=6035.0, high=8165.0),
            carbon_kg=1450.0,
            indicative_cost=[8000.0, 14000.0],
        ),
    ]

    energy_before = EnergyBreakdown(
        elec_kwh=3100.0,
        gas_kwh=11500.0,
        heat_demand_kwh=9430.0,
        base_load_kwh=2070.0,
        hlc_w_per_k=245.0,
        calibrated=True,
    )
    energy_after = EnergyBreakdown(
        elec_kwh=4500.0,
        gas_kwh=1200.0,
        heat_demand_kwh=2330.0,
        base_load_kwh=2070.0,
        hlc_w_per_k=170.0,
        calibrated=True,
    )

    return PropertyReport(
        property=prop,
        score=score,
        score_after=ScoreResult(total=74, classification="Good", components=[]),
        is_poor=False,
        key_issues=["Low insulation score (45/100)"],
        improvement_potential_pct=34.5,
        recommendations=recs,
        infeasible=[],
        package_saving=FuelDelta(elec_kwh=-1400.0, gas_kwh=10300.0, total_kwh=8900.0, low=7565.0, high=10235.0),
        package_carbon_kg=1815.0,
        energy_before=energy_before,
        energy_after=energy_after,
        value={
            "baseline": 285000.0,
            "low": 293000.0,
            "high": 302000.0,
            "pct_low": 2.8,
            "pct_high": 6.0,
        },
    )


@pytest.fixture
def mock_property_report() -> PropertyReport:
    return build_mock_report()


def test_extract_numbers_from_text():
    text = "Saving £1,800 on gas and 3,100 kWh electricity with a 34.5% uplift and £285,000 baseline."
    matches = extract_numbers_from_text(text)
    values = [m[0] for m in matches]
    assert 1800.0 in values
    assert 3100.0 in values
    assert 34.5 in values
    assert 285000.0 in values


def test_extract_numbers_from_json():
    data = {
        "score": 58,
        "nested": {"savings": [1800.0, 3100.5]},
        "label": "ignore_string",
        "flag": True,  # booleans should be excluded
    }
    nums = extract_numbers_from_json(data)
    assert 58.0 in nums
    assert 1800.0 in nums
    assert 3100.5 in nums
    assert True not in nums  # Ensure boolean 1 is not treated as numeric float


def test_grounding_verified_legitimate_report(mock_property_report):
    legit_text = (
        "Your home received a Green Home Score of 58/100 (Moderate). "
        "Top-up loft insulation costs between £500 and £1,100, saving 1,800 kWh of gas "
        "and reducing carbon by 365 kg CO2e. "
        "Overall package savings are estimated at 8,900 kWh/yr with an indicative "
        "property value uplift of £293,000 to £302,000 (2.8% to 6.0%)."
    )
    result = verify_grounding(legit_text, mock_property_report)
    assert result.grounded is True
    assert result.grounding_score >= 0.95
    assert len(result.unsupported_claims) == 0


def test_grounding_catches_hallucinated_cost(mock_property_report):
    hallucinated_text = (
        "Your Green Home Score is 58/100. Loft insulation will cost exactly £2,450."
    )
    result = verify_grounding(hallucinated_text, mock_property_report)
    assert result.grounded is False
    assert any("2,450" in claim or "2450" in claim for claim in result.unsupported_claims)


def test_grounding_catches_hallucinated_savings(mock_property_report):
    hallucinated_text = (
        "By installing heat pumps, your home will save 15,000 kWh per year."
    )
    result = verify_grounding(hallucinated_text, mock_property_report)
    assert result.grounded is False
    assert any("15,000" in claim or "15000" in claim for claim in result.unsupported_claims)


def test_grounding_catches_fuel_sign_inversion(mock_property_report):
    """If LLM claims heat pump saves 1,400 kWh electricity when it actually raises electricity (-1400), flag it."""
    # LLM claims positive electric saving of 4,000 kWh
    inverted_text = (
        "The heat pump will save 4,000 kWh of electricity each year."
    )
    result = verify_grounding(inverted_text, mock_property_report)
    assert result.grounded is False
    assert len(result.unsupported_claims) > 0


# ---------------------------------------------------------------------------
# 50-SCENARIO ADVERSARIAL EVALUATION HARNESS
# ---------------------------------------------------------------------------

SCENARIOS_ADVERSARIAL_50 = [
    # Category 1: Unsupported / Fabricated Numbers (12 scenarios)
    ("Loft insulation will cost £3,750.", False),
    ("Your score will jump by 41 points to 92/100.", False),
    ("You will save £890 on energy bills each month.", False),
    ("Solar will produce 9,800 kWh per year on your roof.", False),
    ("Your carbon emissions will drop by 6,600 kg.", False),
    ("Your house will increase in value by £45,000 immediately.", False),
    ("The installation takes 18 days to complete.", False),
    ("Government grants will pay £12,500 for your insulation.", False),
    ("Your heating efficiency will reach 88.5%.", False),
    ("Your current gas bill is 19,400 kWh.", False),
    ("Estimated energy bill cut is £1,234.", False),
    ("You qualify for a 15.8% discount on your mortgage.", False),

    # Category 2: Heat Pump Sign & Fuel Conflation (10 scenarios)
    ("The heat pump cuts electricity use by 2,500 kWh.", False),
    ("Electricity bills will reduce by £620 with a heat pump.", False),
    ("Installing a heat pump saves 3,400 kWh electricity and 500 kWh gas.", False),
    ("Heat pump reduces total electricity demand to 1,650 kWh.", False),
    ("Your grid imports will fall by 4,100 kWh.", False),
    ("Electricity consumption drops by 25.5%.", False),
    ("The system saves 5,000 kWh off-peak power.", False),
    ("You will export 3,200 kWh electricity to the grid.", False),
    ("Electric heating demand falls from 3,100 to 800 kWh.", False),
    ("Solar PV will reduce space heating fuel by 4,000 kWh.", False),

    # Category 3: Disqualified Retrofits on Infeasible Homes (10 scenarios)
    ("We recommend cavity wall insulation saving 2,200 kWh.", False),
    ("Cavity insulation will cost £2,750 for your solid brick walls.", False),
    ("Your third-floor flat can install a 4 kWp rooftop solar system.", False),
    ("Solar panels on your flat will generate 3,500 kWh.", False),
    ("Replace double glazing with double glazing for £3,850 savings.", False),
    ("Install a heat pump without insulation to save £2,250.", False),
    ("Fill your cavity walls to gain 14 points.", False),
    ("Install solar panels on your listed building for £6,800.", False),
    ("Loft insulation will save 3,000 kWh in your ground floor flat.", False),
    ("Your flat's loft will cost £800 to insulate.", False),

    # Category 4: Missing / Sparse Data Traps (10 scenarios)
    ("Your loft has 25 mm insulation so it needs full top-up.", False),
    ("Because loft depth is unrecorded, we assume 50 mm.", False),
    ("Your floor area was estimated at 450 m2.", False),
    ("Annual gas is missing, so we estimated it at 6,200 kWh.", False),
    ("Your heating score is 12/100 because heating type is unknown.", False),
    ("We assumed 120 mm cavity depth for calculations.", False),
    ("Property value was missing so we defaulted to £150,000.", False),
    ("Estimated heating loss is 550 W/K.", False),
    ("Assuming 1.5 air changes per hour for your property.", False),
    ("Unknown glazing was calculated as single glazed saving £950.", False),

    # Category 5: Grounded, Compliant Explanations (8 scenarios)
    ("Your home received a Green Home Score of 58/100 (Moderate).", True),
    ("Loft insulation top-up is estimated to cost between £500 and £1,100.", True),
    ("Loft insulation reduces gas consumption by 1,800 kWh per year.", True),
    ("Loft insulation reduces carbon emissions by 365 kg CO2e.", True),
    ("Total package savings are estimated at 8,900 kWh across all fuels.", True),
    ("Package carbon reduction is 1,815 kg CO2e per year.", True),
    ("Indicative property value is estimated between £293,000 and £302,000.", True),
    ("The indicative value increase ranges from 2.8% to 6.0%.", True),
]


@pytest.mark.parametrize("statement,expected_pass", SCENARIOS_ADVERSARIAL_50)
def test_adversarial_grounding_scenarios_50(mock_property_report, statement, expected_pass):
    """Verify that all 42 hallucination traps are blocked and 8 valid statements pass."""
    res = verify_grounding(statement, mock_property_report)
    if expected_pass:
        assert res.grounded is True, f"Legitimate statement failed grounding: {statement}"
        assert len(res.unsupported_claims) == 0
    else:
        assert (
            res.grounded is False or len(res.unsupported_claims) > 0
        ), f"Hallucinated statement slipped past grounding shield: {statement}"
