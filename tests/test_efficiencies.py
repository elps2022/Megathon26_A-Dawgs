from pathlib import Path
import yaml

CFG = Path(__file__).resolve().parents[1] / "config" / "efficiencies.yaml"
SYSTEMS = yaml.safe_load(CFG.read_text())["heating_systems"]
SCORE_KEYS = yaml.safe_load(
    (CFG.parent / "score_mappings.yaml").read_text()
)["heating"]["scores"].keys()


def test_every_scored_heating_type_has_an_efficiency():
    assert set(SCORE_KEYS) == set(SYSTEMS)


def test_every_row_has_source_fuel_and_positive_efficiency():
    for name, row in SYSTEMS.items():
        assert row["source"], name
        assert row["fuel"], name
        assert row["efficiency"] > 0, name


def test_only_the_heat_pump_beats_100_percent():
    for name, row in SYSTEMS.items():
        if name != "heat_pump":
            assert row["efficiency"] <= 1.0, name
    assert SYSTEMS["heat_pump"]["efficiency"] > 1.0


def test_heat_pump_range_brackets_central_value():
    hp = SYSTEMS["heat_pump"]
    assert hp["efficiency_low"] <= hp["efficiency"] <= hp["efficiency_high"]
