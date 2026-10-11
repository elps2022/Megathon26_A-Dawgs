"""Per-home degree-day regression (PRISM) on monthly bills: weather-normalised annual use and the home's
own heating response (ARCHITECTURE 3.5 step 2, stage 2 feature engineering).

For each home and heating fuel: monthly kWh = base + slope x HDD_month (regional Met Office HDD), fitted by
ordinary least squares over the months observed. Missing months simply drop out of the fit.
  normalised annual = 12 x base + slope x HDD_typical_year   (typical = mean annual HDD over normal_years)
A fit is used only with >= min_months, slope > 0, base >= 0 and R^2 >= min_r2 (config/annualisation.yaml);
otherwise the home keeps its annualised figure and is flagged weather_normalisation_unavailable, with the
reason in fit_issue. Electrically heated homes with solar are not fitted: solar lowers summer grid import,
which bills alone can't separate from weather, so the heating slope would be overstated.
The slope is also V1's "actual heating response" for its calibration factor k = actual / physics slope.

Heating fuel per home: gas for gas-heated homes, electricity for electrically heated homes. Other fuels
(oil, LPG, district) aren't metered, so there is nothing to fit. Note: synthetic bills are generated from the
same HDD, so the fits look near perfect on them; never report that as accuracy.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
CFG = yaml.safe_load((ROOT / "config" / "annualisation.yaml").read_text())["weather_normalisation"]
GAS_HEATING = {"gas_condensing", "gas_non_condensing"}
ELECTRIC_HEATING = {"storage_heaters", "direct_electric", "heat_pump"}


def load_hdd() -> tuple[pd.DataFrame, pd.Series]:
    hdd = pd.read_csv(ROOT / "config" / "degree_days.csv", comment="#")
    lo, hi = CFG["normal_years"]
    typical = hdd[hdd.year.between(lo, hi)].groupby(["local_authority", "year"]).hdd.sum().groupby("local_authority").mean()
    return hdd[["local_authority", "year", "month", "hdd"]], typical


def heating_fuel(props: pd.DataFrame) -> pd.Series:
    """"gas" / "elec" for the metered heating fuel, else None."""
    gas = props.heating.isin(GAS_HEATING) & props.gas_meter.astype(bool)
    elec = props.heating.isin(ELECTRIC_HEATING)
    return pd.Series(np.where(gas, "gas", np.where(elec, "elec", None)), index=props.index)


def fit(props: pd.DataFrame, monthly: pd.DataFrame) -> pd.DataFrame:
    """One row per home: heating fuel, months used, base (kWh/month), slope (kWh/HDD), R^2, normalised annual."""
    hdd, typical = load_hdd()
    fuel = heating_fuel(props).rename("fuel")
    solar_elec = (fuel == "elec") & (props.solar_kwp.fillna(0) > 0)
    m = monthly.merge(props[["property_id", "local_authority"]].assign(fuel=fuel.to_numpy()), on="property_id")
    m = m[m.fuel.notna()]
    month = pd.to_datetime(m.month)
    m = m.assign(year=month.dt.year, month_num=month.dt.month)
    m = m.merge(hdd.rename(columns={"month": "month_num"}), on=["local_authority", "year", "month_num"], how="left")
    m["y"] = np.where(m.fuel == "gas", m.gas_kwh, m.elec_import_kwh)
    m = m[m.y.notna() & m.hdd.notna()]
    g = m.assign(xy=m.hdd * m.y, xx=m.hdd ** 2, yy=m.y ** 2).groupby("property_id").agg(
        n=("y", "size"), sx=("hdd", "sum"), sy=("y", "sum"), sxy=("xy", "sum"), sxx=("xx", "sum"),
        syy=("yy", "sum"), fuel=("fuel", "first"), la=("local_authority", "first"))
    sxx_c = g.sxx - g.sx ** 2 / g.n
    sxy_c = g.sxy - g.sx * g.sy / g.n
    syy_c = g.syy - g.sy ** 2 / g.n
    slope = sxy_c / sxx_c.where(sxx_c > 0)
    base = (g.sy - slope * g.sx) / g.n
    r2 = (sxy_c ** 2 / (sxx_c * syy_c)).where((sxx_c > 0) & (syy_c > 0))
    confounded = g.index.isin(props.property_id[solar_elec])
    issue = np.select([confounded, g.n < CFG["min_months"], ~(slope > 0) | ~(base >= 0), ~(r2 >= CFG["min_r2"])],
                      ["solar_confounded", "too_few_months", "implausible_fit", "weak_fit"], default="")
    ok = issue == ""
    normalised = 12 * base + slope * g.la.map(typical)
    out = pd.DataFrame({"heating_fuel": g.fuel, "fit_months": g.n, "base_kwh_per_month": base.round(1),
                        "slope_kwh_per_hdd": slope.round(3), "fit_r2": r2.round(3), "fit_ok": ok, "fit_issue": issue,
                        "annual_heating_fuel_kwh_normalised": normalised.where(ok).round(1)})
    return out.reset_index()
