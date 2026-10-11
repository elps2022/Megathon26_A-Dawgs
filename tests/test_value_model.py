"""Value model (models/value_model.py) and its use in the report (ARCHITECTURE 3.8). Skips without artifacts."""

import pytest

from greenhome.models.energy import ARTIFACTS
from greenhome.schema import Property

TRAINED = (ARTIFACTS / "value_meta.json").exists()
pytestmark = pytest.mark.skipif(not TRAINED, reason="value model not trained (make train-value / make models)")


def home(**kw) -> dict:
    base = dict(property_id="T1", customer_id="C1", local_authority="Leeds", region="Yorkshire and The Humber",
                property_type="Semi detached", age_band="1930-1972", floor_area_m2=90.0, bedrooms=3, energy_rating="D",
                sap_score=60.0, heating="gas_condensing", wall_type="cavity", wall_insulation="none",
                loft_insulation_mm=100.0, glazing="double_old", gas_meter=True, imd_quintile=3,
                current_value_estimate=250000.0)
    return Property(**{**base, **kw}).model_dump()


def test_better_rating_never_lowers_the_level():
    from greenhome.models.value_model import predict_range
    vals = [predict_range(home(sap_score=s, energy_rating=b))["p50"] for s, b in ((30, "F"), (50, "E"), (60, "D"), (75, "C"), (85, "B"))]
    assert vals == sorted(vals)


def test_band_contains_median():
    from greenhome.models.value_model import predict_range
    r = predict_range(home())
    assert r["p10"] < r["p50"] < r["p90"]


def test_premia_are_ordered_e_and_fg_below_d():
    from greenhome.models.value_model import _load
    p = _load()["hedonic"]["premia"]
    assert p["F/G"]["pct"] < p["E"]["pct"] < p["D"]["pct"] <= p["C"]["pct"]


def test_uplift_positive_for_d_to_c_zero_without_band_change_unavailable_without_rating():
    from greenhome.engine.savings import apply_retrofits_to_property
    from greenhome.models.value_model import value_estimate
    p = home()
    after = apply_retrofits_to_property(Property(**p), ["cavity_wall_insulation", "loft_insulation", "solar_pv"]).model_dump()
    v = value_estimate(p, after, ["cavity_wall_insulation", "loft_insulation", "solar_pv"])
    assert v["band_change"] == "D -> C" and 0 < v["pct_low"] <= v["pct_high"] and v["low"] <= v["high"]
    same = value_estimate(p, p, [])
    assert same["pct_low"] == same["pct_high"] == 0.0
    none = value_estimate(home(energy_rating=None, sap_score=None), p, ["loft_insulation"])
    assert none["status"] == "unavailable"


def test_report_uses_the_value_model():
    from greenhome.engine.report import build_report
    r = build_report(Property(**home()))
    assert "Hedonic" in r.value["method"] and r.value["caveat"]
