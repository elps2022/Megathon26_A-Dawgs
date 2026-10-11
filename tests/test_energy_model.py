"""EnergyModel interface (models/energy.py) and its use in the savings simulator (ARCHITECTURE 3.5, 3.6).

Physics-only and fallback paths need no trained artifacts; V1/V2 tests skip when `make train` hasn't run."""

import numpy as np
import pandas as pd
import pytest

from greenhome.engine import savings
from greenhome.models.energy import ARTIFACTS, MODELS, ModelUnavailable, load_model, physics_elasticity
from greenhome.models.need_data import epc_features
from greenhome.schema import Property

TRAINED = (ARTIFACTS / "meta.json").exists()
needs_artifacts = pytest.mark.skipif(not TRAINED, reason="energy model artifacts not trained (make train)")


def home(**kw) -> Property:
    base = dict(property_id="T1", customer_id="C1", local_authority="Leeds", region="Yorkshire and The Humber",
                property_type="Semi detached", age_band="1930-1972", floor_area_m2=90.0, energy_rating="D",
                heating="gas_condensing", wall_type="cavity", wall_insulation="none", loft_insulation_mm=100.0,
                glazing="double_old", gas_meter=True, imd_quintile=3, cavity_filled_retrofit=False,
                annual_elec_kwh=3000.0, annual_gas_kwh=12000.0)
    return Property(**{**base, **kw})


def frame(*props: Property) -> pd.DataFrame:
    return pd.DataFrame([p.model_dump() for p in props])


def test_physics_model_runs_without_artifacts():
    out = load_model("physics").predict_frame(frame(home(), home(heating="storage_heaters", gas_meter=False)))
    assert np.isfinite(out.elec_kwh).all()
    assert np.isfinite(out.gas_kwh[0]) and out.gas_kwh[0] > 0
    assert np.isnan(out.gas_kwh[1])  # not gas heated: the NEED gas model doesn't cover it


def test_unknown_model_name():
    with pytest.raises(ValueError):
        load_model("nope")


def test_unknown_cavity_flag_stays_unknown():
    x = epc_features(frame(home(cavity_filled_retrofit=None), home(cavity_filled_retrofit=True)))
    assert np.isnan(x.cavity_installed[0]) and x.cavity_installed[1] == 1


def test_elasticity_band_contains_pooled_value():
    e = physics_elasticity()
    assert 0 < e["low"] <= e["beta"] <= e["high"] <= 1


def _package(model_name, p=None, retrofits=("loft_insulation", "cavity_wall_insulation", "double_glazing")):
    return savings.simulate_package_savings(p or home(), list(retrofits), model_name)


@pytest.mark.parametrize("name", list(MODELS))
def test_fabric_savings_positive_and_additive(name):
    before, after, recs, pkg, _ = _package(name)
    assert all(r.marginal_saving.total_kwh > 0 for r in recs)
    assert sum(r.marginal_saving.total_kwh for r in recs) == pytest.approx(pkg.total_kwh, abs=0.1 * len(recs))
    for r in recs:
        assert r.marginal_saving.low <= r.marginal_saving.total_kwh <= r.marginal_saving.high
    assert after.gas_kwh < before.gas_kwh
    assert after.elec_kwh == pytest.approx(before.elec_kwh)  # fabric in a gas home doesn't touch electricity


def test_missing_artifacts_fall_back_to_physics(monkeypatch, tmp_path):
    from greenhome.models import energy
    load_model.cache_clear()
    monkeypatch.setattr(energy, "ARTIFACTS", tmp_path)
    monkeypatch.setitem(energy.MODELS, "greybox", lambda: energy.GreyboxModel(tmp_path))
    try:
        before, _, recs, _, _ = _package("greybox")
        assert before.model_version == "physics_fallback"
        assert all("physics" in r.reason for r in recs)
    finally:
        load_model.cache_clear()


def test_electric_heating_saves_electricity_only():
    p = home(heating="storage_heaters", gas_meter=False, annual_gas_kwh=None, annual_elec_kwh=9000.0)
    before, after, recs, pkg, _ = _package("greybox", p, ["loft_insulation"])
    assert pkg.gas_kwh == 0 and pkg.elec_kwh > 0
    assert "physics" in recs[0].reason


@needs_artifacts
def test_v1_retrofit_is_physics_change_to_the_elasticity():
    m = load_model("greybox")
    p0, p1 = home(), savings.apply_retrofits_to_property(home(), ["loft_insulation"])
    out = m.predict_frame(frame(p0, p1))
    phys = out.physics_space_heat_kwh + out.physics_hot_water_kwh
    beta = m.meta["gas"]["physics_elasticity"]
    assert out.gas_kwh[1] / out.gas_kwh[0] == pytest.approx((phys[1] / phys[0]) ** beta, rel=1e-6)
    assert beta == physics_elasticity()["beta"]  # deployment and evaluation use the same value


@needs_artifacts
@pytest.mark.parametrize("retrofit", ["loft_insulation", "cavity_wall_insulation", "solid_wall_insulation", "double_glazing"])
def test_v1_insulation_never_raises_gas(retrofit):
    p0 = home(wall_type="solid") if retrofit == "solid_wall_insulation" else home()
    p1 = savings.apply_retrofits_to_property(p0, [retrofit])
    gas = load_model("greybox").predict_frame(frame(p0, p1)).gas_kwh
    assert gas[1] < gas[0]


@needs_artifacts
def test_predict_breakdown_and_unknown_floor_area():
    b = load_model("greybox").predict(home())
    assert b.gas_kwh > 0 and b.elec_kwh > 0 and len(b.elements) == 4
    assert b.heat_demand_kwh + b.base_load_kwh == pytest.approx(b.gas_kwh + b.elec_kwh, abs=0.2)
    with pytest.raises(ModelUnavailable):
        load_model("greybox").predict(home(floor_area_m2=None))


@needs_artifacts
def test_v1_band_brackets_the_prediction():
    out = load_model("greybox").predict_frame(frame(home(), home(heating="storage_heaters", gas_meter=False)))
    for fuel in ("gas", "elec"):
        r = out.iloc[0]
        assert r[f"{fuel}_p10"] < r[f"{fuel}_kwh"] < r[f"{fuel}_p90"]
    assert np.isnan(out.gas_p10[1]) and np.isnan(out.gas_p90[1])
    b = load_model("greybox").predict(home())
    assert b.gas_p10 < b.gas_kwh < b.gas_p90 and b.elec_p10 < b.elec_kwh < b.elec_p90


@needs_artifacts
def test_missing_deprivation_is_neutral_not_most_deprived():
    out = load_model("greybox").predict_frame(frame(home(imd_quintile=None), home(imd_quintile=3), home(imd_quintile=1)))
    assert out.gas_kwh[0] == pytest.approx(out.gas_kwh[1])
    assert out.gas_kwh[0] != pytest.approx(out.gas_kwh[2])


@pytest.mark.parametrize("name", ["greybox", "physics"])
def test_cooker_only_gas_is_base_load_not_a_heating_estimate(name):
    gas_heated = savings.baseline_energy(home(annual_gas_kwh=None), name)[1]
    cooker = savings.baseline_energy(home(heating="storage_heaters", annual_gas_kwh=None), name)[1]
    assert 0 < cooker < 0.3 * gas_heated


@needs_artifacts
def test_estimated_use_carries_the_model_band():
    before = savings.simulate_package_savings(home(annual_gas_kwh=None, annual_elec_kwh=None), ["loft_insulation"])[0]
    assert before.gas_p10 < before.gas_kwh < before.gas_p90 and before.elec_p10 < before.elec_kwh < before.elec_p90
    billed = savings.simulate_package_savings(home(), ["loft_insulation"])[0]
    assert billed.gas_p10 is None  # billed use is a measurement, not a model estimate


@needs_artifacts
def test_mode1_forecast_is_near_the_bill_and_sets_the_baseline():
    f = load_model("greybox").forecast_frame(frame(home(), home(annual_gas_kwh=None)))
    assert 0.7 < f.gas_kwh[0] / 12000 < 1.3 and 0.7 < f.elec_kwh[0] / 3000 < 1.3
    assert np.isnan(f.gas_kwh[1])  # no gas bill: nothing to forecast from
    before = savings.simulate_package_savings(home(), ["loft_insulation"])[0]
    assert before.model_version == "greybox_v1+mode1" and before.gas_kwh == pytest.approx(f.gas_kwh[0], abs=0.1)


def test_score_reference_compares_gas_meter_homes_with_gas_heated_homes():
    from greenhome.models.energy import similar_home_kwh
    hp = home(heating="heat_pump").model_dump()
    gas_home = similar_home_kwh(home().model_dump(), "median", score_reference=True)["gas"]
    assert similar_home_kwh(hp, "median", score_reference=True)["gas"] == pytest.approx(gas_home)
    assert similar_home_kwh(hp, "median")["gas"] < 0.3 * gas_home   # actual-use estimate: cooking / hot water only


def test_heat_pump_and_solar_never_lower_the_score():
    from greenhome.engine.report import build_report
    r0 = build_report(home())
    r1 = build_report(home(), retrofits=["heat_pump", "solar_pv"])
    assert r1.score_after.total >= r0.score.total


@needs_artifacts
def test_detector_flags_heavy_users_not_big_homes():
    from greenhome.engine.detector import similar_homes_check
    from greenhome.engine.report import build_report
    typical = similar_homes_check(home())
    assert typical["status"] == "checked" and typical["flagged"] == []
    heavy = similar_homes_check(home(annual_gas_kwh=60000.0))
    assert heavy["flagged"] == ["gas"] and heavy["fuels"]["gas"]["ratio_to_similar_median"] > 2
    big = similar_homes_check(home(floor_area_m2=250.0, property_type="Detached", annual_gas_kwh=22000.0))
    assert "gas" not in big["flagged"]          # big, but normal for a big home
    assert any("similar homes" in i for i in build_report(home(annual_gas_kwh=60000.0)).key_issues)


@needs_artifacts
def test_detector_unavailable_without_bills_or_for_cooker_gas():
    from greenhome.engine.detector import similar_homes_check
    c = similar_homes_check(home(annual_gas_kwh=None, annual_elec_kwh=None))
    assert c["status"] == "unavailable" and c["flagged"] == []
    cooker = similar_homes_check(home(heating="storage_heaters"))
    assert cooker["fuels"]["gas"]["status"] == "unavailable"
