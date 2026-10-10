from greenhome.engine.carbon import carbon_saved_kg, load_factors

F = load_factors()

def test_zero():
    assert carbon_saved_kg(0, 0, F) == 0

def test_heat_pump_electricity_penalty_counts():
    assert carbon_saved_kg(-2000, 10000, F) < carbon_saved_kg(0, 10000, F)

def test_factors_present_and_positive():
    assert F["electricity"] > 0 and F["gas"] > 0
