import numpy as np
import pandas as pd

from greenhome.models.physics import envelope, need_fabric, need_physics, property_fabric, property_physics


def _home(**kw):
    row = dict(property_type="Semi detached", age_band="1930-1972", floor_area_m2=100.0, has_own_roof=True,
               loft_insulation_mm=100.0, wall_type="cavity", wall_insulation="none", glazing="double_old",
               region="Yorkshire and The Humber", heating="gas_condensing", gas_meter=True)
    row.update(kw)
    return pd.DataFrame([row])


def test_hlc_matches_hand_calculation():
    env = envelope(property_fabric(_home())[0]).iloc[0]
    # house: 2 storeys -> footprint 50 m2; perimeter 4*sqrt(50); window 0.2*100 = 20 m2; height 2.6 m
    wall = 4 * np.sqrt(50) * 2.6 * 2 - 20
    expected = 50 * 0.65 + wall * 1.6 + 20 * 2.8 + 50 * 0.70 + 0.33 * 0.8 * 100 * 2.6
    assert abs(env.hlc_w_per_k - expected) < 1e-6


def test_more_insulation_never_raises_gas():
    base = property_physics(_home()).gas_kwh.iat[0]
    assert property_physics(_home(loft_insulation_mm=300.0)).gas_kwh.iat[0] < base
    assert property_physics(_home(wall_insulation="full")).gas_kwh.iat[0] < base
    assert property_physics(_home(glazing="triple")).gas_kwh.iat[0] < base


def test_heat_pump_moves_heating_from_gas_to_less_electricity():
    gas = property_physics(_home()).iloc[0]
    hp = property_physics(_home(heating="heat_pump", gas_meter=False)).iloc[0]
    assert hp.gas_kwh == 0 and hp.elec_kwh > gas.elec_kwh
    assert hp.elec_kwh - gas.elec_kwh < gas.gas_kwh    # SCOP > 1: less energy than the boiler burned


def test_unknown_fabric_uses_age_default_and_is_flagged():
    d = property_physics(_home(wall_type="unknown", glazing=None, loft_insulation_mm=None)).iloc[0]
    assert np.isfinite(d.gas_kwh) and set(d.fabric_defaulted.split(",")) == {"wall_type", "loft_insulation_mm", "glazing"}


def test_need_and_epc_paths_agree_for_the_same_described_home():
    x = pd.DataFrame([{"prop_type": "Semi detached", "age_band": 2.0, "floor_area_band": 2.0, "region": "E12000003",
                       "cavity_installed": 0, "gas_heated": 1}])
    need = envelope(need_fabric(x)).hlc_w_per_k.iat[0]
    epc = envelope(property_fabric(_home(floor_area_m2=75.0))[0]).hlc_w_per_k.iat[0]
    assert abs(need - epc) < 1e-6
    assert need_physics(x).gas_kwh.iat[0] > 0
