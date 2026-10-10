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
