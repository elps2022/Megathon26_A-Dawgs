import pytest
from greenhome.score.score import load_mappings, score_home, classify, _interp

M = load_mappings()


def test_weights_sum_to_one():
    assert sum(M["weights"].values()) == pytest.approx(1.0)


def test_seven_components():
    assert len(M["weights"]) == 7


@pytest.mark.parametrize("total,expected", [
    (100, "Highly efficient"), (80, "Highly efficient"), (79, "Good"),
    (65, "Good"), (64, "Moderate"), (50, "Moderate"),
    (49, "Poor"), (30, "Poor"), (29, "Very poor"), (0, "Very poor"),
])
def test_band_boundaries(total, expected):
    assert classify(total, M) == expected


def _strictly_decreasing(values):
    return all(a > b for a, b in zip(values, values[1:]))


def test_rating_monotonic():
    t = M["energy_efficiency"]["rating_midpoints"]
    assert _strictly_decreasing([t[k] for k in "abcdefg"])


def test_heating_monotonic():
    t = M["heating"]["scores"]
    order = ["heat_pump", "district", "gas_condensing", "gas_non_condensing",
             "storage_heaters", "oil", "direct_electric"]
    assert _strictly_decreasing([t[k] for k in order])


def test_glazing_monotonic():
    t = M["glazing"]["scores"]
    order = ["triple", "double_new", "double_old", "partial", "single"]
    assert _strictly_decreasing([t[k] for k in order])


def test_smart_controls_monotonic():
    t = M["smart_controls"]["scores"]
    order = ["smart_trvs", "smart_thermostat", "programmer", "none"]
    assert _strictly_decreasing([t[k] for k in order])


def test_wall_monotonic():
    t = M["insulation"]["wall"]
    assert t["full"] > t["partial"] > t["none"]


def test_loft_depth_never_decreases():
    pts = M["insulation"]["loft_mm_points"]
    vals = [_interp(mm, pts) for mm in range(0, 400, 10)]
    assert all(b >= a for a, b in zip(vals, vals[1:]))


def test_lower_intensity_never_scores_lower():
    for key, top in (("points", 500), ("ratio_points", 3)):
        pts = M["consumption_intensity"][key]
        vals = [_interp(k / 10 if key == "ratio_points" else k, pts) for k in range(0, top * 10, 1 if key == "ratio_points" else 10)]
        assert all(b <= a for a, b in zip(vals, vals[1:]))


def _consumption_component(home):
    return next(c for c in score_home(home, M)["components"] if c["component"] == "consumption_intensity")


def _typical(home):
    from types import SimpleNamespace
    from greenhome.models.expected_use import expected_annual_kwh
    keys = ("property_type", "age_band", "floor_area_m2", "energy_rating", "heating", "gas_meter")
    e = expected_annual_kwh(SimpleNamespace(**{k: home[k] for k in keys}), statistic="median")
    return dict(home, annual_elec_kwh=e["elec"], annual_gas_kwh=e["gas"])


def test_typical_homes_score_the_same_whatever_their_size():
    small = _typical(dict(POOR, property_type="Flat", floor_area_m2=45, energy_rating="D"))
    big = _typical(dict(POOR, property_type="Detached", floor_area_m2=190, energy_rating="D"))
    s, b = _consumption_component(small), _consumption_component(big)
    assert s["score"] == pytest.approx(75, abs=0.5) and b["score"] == pytest.approx(75, abs=0.5)
    assert s["inputs_used"]["kwh_per_m2"] > b["inputs_used"]["kwh_per_m2"]   # raw kWh/m2 would have punished the flat


def test_using_more_than_similar_homes_scores_lower():
    base = _typical(dict(POOR, energy_rating="D"))
    more = dict(base, annual_gas_kwh=base["annual_gas_kwh"] * 1.5)
    assert _consumption_component(more)["score"] < _consumption_component(base)["score"]


def test_unmetered_heating_fuel_is_missing_not_perfect():
    oil = dict(POOR, heating="oil", gas_meter=False, annual_gas_kwh=0.0, annual_elec_kwh=2500,
               data_flags=["heating_fuel_unmetered"])
    c = _consumption_component(oil)
    assert c["score"] == M["default_score"] and "heating_fuel_unmetered" in c["flags"]


POOR = dict(energy_rating="F", sap_score=None, floor_area_m2=80,
            annual_elec_kwh=3000, annual_gas_kwh=20000, loft_insulation_mm=0,
            wall_insulation="none", heating="gas_non_condensing", glazing="single",
            solar_kwp=0, smart_controls="none", property_type="Semi detached", age_band="pre-1930", gas_meter=True)

GOOD = dict(energy_rating="A", sap_score=None, floor_area_m2=100,
            annual_elec_kwh=4000, annual_gas_kwh=0, loft_insulation_mm=300,
            wall_insulation="full", heating="heat_pump", glazing="triple",
            solar_kwp=4, smart_controls="smart_trvs", property_type="Detached", age_band="2000+", gas_meter=False)


def test_poor_home_is_very_poor():
    assert score_home(POOR, M)["classification"] == "Very poor"


def test_good_home_is_highly_efficient():
    assert score_home(GOOD, M)["total"] >= 80


def test_better_home_scores_higher():
    assert score_home(GOOD, M)["total"] > score_home(POOR, M)["total"]


def test_breakdown_sums_to_total():
    r = score_home(POOR, M)
    assert round(sum(c["contribution"] for c in r["components"])) == pytest.approx(r["total"], abs=1)


def test_missing_data_is_flagged_not_zero():
    home = dict(GOOD, heating=None)
    r = score_home(home, M)
    heating = next(c for c in r["components"] if c["component"] == "heating")
    assert "missing_heating" in heating["flags"]
    assert heating["score"] == M["default_score"] and heating["score"] != 0


def test_no_solar_is_zero_not_missing():
    r = score_home(dict(GOOD, solar_kwp=0), M)
    ren = next(c for c in r["components"] if c["component"] == "renewables")
    assert ren["score"] == 0 and ren["flags"] == []
