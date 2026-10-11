import numpy as np
import pandas as pd

from greenhome.data.degree_day_fit import fit, load_hdd


def _home(heating="gas_condensing", gas_meter=True):
    return pd.DataFrame([{"property_id": "H1", "local_authority": "Leeds", "heating": heating, "gas_meter": gas_meter}])


def _bills(base, slope, months, fuel="gas"):
    hdd, _ = load_hdd()
    h = hdd[hdd.local_authority == "Leeds"].set_index(["year", "month"]).hdd
    rows = [{"property_id": "H1", "month": f"{m.year}-{m.month:02d}-01",
             "gas_kwh": np.nan, "elec_import_kwh": 200.0} for m in months]
    for r, m in zip(rows, months):
        r["gas_kwh" if fuel == "gas" else "elec_import_kwh"] = base + slope * h[(m.year, m.month)]
    return pd.DataFrame(rows)


def test_recovers_base_and_slope_and_normalises_to_typical_year():
    months = pd.date_range("2024-10-01", periods=24, freq="MS")
    out = fit(_home(), _bills(150.0, 3.0, months)).iloc[0]
    _, typical = load_hdd()
    assert abs(out.base_kwh_per_month - 150) < 0.1 and abs(out.slope_kwh_per_hdd - 3) < 1e-3
    assert out.fit_ok and abs(out.annual_heating_fuel_kwh_normalised - (12 * 150 + 3 * typical["Leeds"])) < 1


def test_missing_months_drop_out_and_too_few_months_is_not_ok():
    months = pd.date_range("2024-10-01", periods=24, freq="MS")
    gappy = _bills(150.0, 3.0, months).drop(index=[3, 4, 5, 6])          # a winter gap
    assert abs(fit(_home(), gappy).iloc[0].slope_kwh_per_hdd - 3) < 1e-3
    few = fit(_home(), _bills(150.0, 3.0, months[:4])).iloc[0]
    assert not few.fit_ok and np.isnan(few.annual_heating_fuel_kwh_normalised)


def test_electric_heating_fits_electricity_and_oil_is_not_fitted():
    months = pd.date_range("2024-10-01", periods=24, freq="MS")
    e = fit(_home("storage_heaters", False), _bills(100.0, 2.0, months, fuel="elec")).iloc[0]
    assert e.heating_fuel == "elec" and abs(e.slope_kwh_per_hdd - 2) < 1e-3
    assert fit(_home("oil", False), _bills(100.0, 2.0, months, fuel="elec")).empty
