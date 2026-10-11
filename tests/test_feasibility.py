from greenhome.engine.feasibility import assess, load_catalogue

CAT = load_catalogue()

BASE = dict(property_type="semi_detached", loft_insulation_mm=0, wall_insulation="none",
            wall_type="cavity", glazing="single", heating="gas_condensing",
            solar_kwp=0, smart_controls="none")


def blocked(p):
    return {x["retrofit"]: x["rule"] for x in assess(p, CAT)["infeasible"]}


def test_every_retrofit_has_stage_cost_and_source():
    for name, r in CAT["retrofits"].items():
        assert r["stage"] in CAT["stages"], name
        low, high = r["cost_gbp"]
        assert 0 < low < high, name
        assert r["source"], name


def test_cavity_walls_block_solid_wall_insulation_only():
    b = blocked(BASE)
    assert "solid_wall_insulation" in b and "cavity_wall_insulation" not in b


def test_solid_walls_block_cavity_insulation_only():
    b = blocked(dict(BASE, wall_type="solid"))
    assert "cavity_wall_insulation" in b and "solid_wall_insulation" not in b


def test_unknown_wall_type_blocks_both_and_flags_it():
    r = assess(dict(BASE, wall_type=None), CAT)
    names = [x["retrofit"] for x in r["infeasible"]]
    assert "cavity_wall_insulation" in names and "solid_wall_insulation" in names
    assert "unknown_wall_type" in r["flags"]["cavity_wall_insulation"]


def test_fully_insulated_walls_block_both_wall_upgrades():
    b = blocked(dict(BASE, wall_insulation="full"))
    assert "cavity_wall_insulation" in b and "solid_wall_insulation" in b


def test_existing_double_glazing_blocks_glazing():
    for g in ("double_old", "double_new", "triple"):
        assert "double_glazing" in blocked(dict(BASE, glazing=g))
    assert "double_glazing" not in blocked(dict(BASE, glazing="partial"))


def test_deep_loft_blocks_loft_insulation():
    assert "loft_insulation" in blocked(dict(BASE, loft_insulation_mm=300))
    assert "loft_insulation" not in blocked(dict(BASE, loft_insulation_mm=0))


def test_flat_blocks_loft_and_solar():
    b = blocked(dict(BASE, property_type="flat"))
    assert "loft_insulation" in b and "solar_pv" in b


def test_existing_solar_and_listed_building_block_solar():
    assert "solar_pv" in blocked(dict(BASE, solar_kwp=3))
    assert "solar_pv" in blocked(dict(BASE, listed_building=True))
    assert "solar_pv" in blocked(dict(BASE, roof_usable=False))


def test_existing_heat_pump_and_smart_thermostat_block_themselves():
    assert "heat_pump" in blocked(dict(BASE, heating="heat_pump"))
    assert "smart_thermostat" in blocked(dict(BASE, smart_controls="smart_thermostat"))


def test_heat_pump_on_uninsulated_home_is_flagged_insulate_first():
    r = assess(BASE, CAT)
    assert "heat_pump" in r["feasible"]
    assert "insulate_first" in r["flags"]["heat_pump"]


def test_feasible_list_is_in_stage_order():
    order = CAT["stages"]
    r = assess(BASE, CAT)["feasible"]
    idx = [order.index(CAT["retrofits"][n]["stage"]) for n in r]
    assert idx == sorted(idx)
    assert r.index("loft_insulation") < r.index("heat_pump") < r.index("solar_pv")


def test_suitability_is_monotone_and_capped():
    from greenhome.engine.savings import _suitability
    order = {"Low": 0, "Medium": 1, "High": 2}
    cost = (1000.0, 3000.0)                     # mid 2,000
    levels = [order[_suitability(g, 0.0, cost, False)[0]] for g in (-50, 0, 50, 100, 150, 250, 500)]
    assert levels == sorted(levels)             # more bill saving never lowers suitability
    assert _suitability(250.0, 0.0, cost, False) == ("High", 8.0)
    assert _suitability(-50.0, 0.0, cost, False) == ("Low", None)          # a higher bill never pays back
    assert _suitability(-50.0, 2000.0, cost, False)[0] == "Medium"         # a big carbon cut keeps it Medium
    assert _suitability(250.0, 0.0, cost, True)[0] == "Medium"             # 'insulate first' caps a heat pump


def test_rank_orders_by_suitability_then_payback():
    from greenhome.engine.savings import rank_recommendations
    from greenhome.schema import FuelDelta, RetrofitRec
    z = FuelDelta(elec_kwh=0, gas_kwh=0, total_kwh=0, low=0, high=0)
    rec = lambda name, s, pb, c: RetrofitRec(retrofit=name, suitability=s, reason="", marginal_saving=z,
                                              carbon_kg=c, indicative_cost=(1, 2), payback_years=pb)
    recs = [rec("a", "Medium", 20.0, 10), rec("b", "High", 9.0, 5), rec("c", "High", 3.0, 1), rec("d", "Low", None, 50)]
    rank_recommendations(recs)
    assert [r.rank for r in recs] == [3, 2, 1, 4]
    assert [r.retrofit for r in recs] == ["a", "b", "c", "d"]   # list order (simulation order) is untouched
