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
