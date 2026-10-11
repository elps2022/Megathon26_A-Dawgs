import numpy as np
import pandas as pd

from greenhome.models.need_data import features


def _need_rows(**overrides):
    row = dict(PROP_TYPE="Semi detached", PROP_AGE_BAND=2, FLOOR_AREA_BAND=2, CONSERVATORY_FLAG=np.nan,
               COUNCIL_TAX_BAND="C", IMD_BAND_ENG=3.0, IMD_BAND_WALES=np.nan, REGION="E12000003", EPC="D",
               LI_FLAG=0, LI_DATE=np.nan, CWI_FLAG=0, CWI_DATE=np.nan, PV_FLAG=0, PV_DATE=np.nan, MAIN_HEAT_FUEL=1)
    row.update(overrides)
    return pd.DataFrame([row])


def test_install_after_target_year_is_not_a_feature():
    x = features(_need_rows(LI_FLAG=1, LI_DATE=2024.0), year=2023)
    assert x.loft_installed.iat[0] == 0 and np.isnan(x.years_since_loft.iat[0])


def test_install_before_target_year_counts():
    x = features(_need_rows(CWI_FLAG=1, CWI_DATE=2015.0), year=2023)
    assert x.cavity_installed.iat[0] == 1 and x.years_since_cavity.iat[0] == 8


def test_council_tax_band_excluded_and_no_epc_is_nan():
    x = features(_need_rows(EPC="No EPC"))
    assert "council_tax_band" not in x.columns
    assert np.isnan(x.epc_band.iat[0]) and x.has_epc.iat[0] == 0


def _epc_home(**overrides):
    row = dict(property_type="Semi detached", age_band="1930-1972", floor_area_m2=85.0, energy_rating="D",
               region="Yorkshire and The Humber", imd_quintile=3, heating="gas_condensing", gas_meter=True,
               cavity_filled_retrofit=True, solar_kwp=0.0)
    row.update(overrides)
    return pd.DataFrame([row])


def test_epc_features_match_need_features_exactly():
    from greenhome.models.need_data import PARITY_FEATURES, epc_features
    need = features(_need_rows(CWI_FLAG=1, CWI_DATE=2015.0))[PARITY_FEATURES]
    epc = epc_features(_epc_home())
    assert list(epc.columns) == list(need.columns)
    for c in PARITY_FEATURES:
        assert str(epc[c].dtype) == str(need[c].dtype) or (epc[c].dtype.kind in "if" and need[c].dtype.kind in "if"), c
        if isinstance(need[c].dtype, pd.CategoricalDtype):
            assert list(epc[c].cat.categories) == list(need[c].cat.categories), c
    # The same home described by both sources gets the same feature values
    pd.testing.assert_frame_equal(epc.reset_index(drop=True), need.reset_index(drop=True), check_dtype=False)


def test_epc_features_unknowns_stay_missing():
    from greenhome.models.need_data import epc_features
    x = epc_features(_epc_home(property_type="unknown", floor_area_m2=None, energy_rating=None))
    assert pd.isna(x.prop_type.iat[0]) and np.isnan(x.floor_area_band.iat[0])
    assert np.isnan(x.epc_band.iat[0]) and x.has_epc.iat[0] == 0
    assert epc_features(_epc_home(floor_area_m2=50.0)).floor_area_band.iat[0] == 1  # NEED band edges: <=50
    assert epc_features(_epc_home(floor_area_m2=50.5)).floor_area_band.iat[0] == 2


def test_expected_use_is_smeared_to_the_mean():
    import math
    import yaml
    from greenhome.models import expected_use as eu
    need = yaml.safe_load(eu._CFG.read_text())
    for key in ("gas_kwh_gas_heated", "elec_kwh_gas_heated", "elec_kwh_not_gas_heated"):
        assert need[key]["smearing_factor"] > 1.0          # log-model outputs sit below the mean
    m, cell = need["gas_kwh_gas_heated"], ("Semi detached", "2", "2", "D")
    e = m["effects"]
    typical = math.exp(m["intercept"] + e["PROP_TYPE"][cell[0]] + e["FLOOR_AREA_BAND"][cell[1]]
                       + e["PROP_AGE_BAND"][cell[2]] + e["EPC"][cell[3]])
    assert abs(eu._predict(m, cell) - round(typical * m["smearing_factor"], 1)) < 1e-6
