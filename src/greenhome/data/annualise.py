"""Annualise monthly energy consumption using seasonal weighting.

Defined in docs/ARCHITECTURE.md section 3.2.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import yaml

# Monthly share of a typical year's use, per fuel (sum to 1.0); profiles and thresholds with their source
# live in config/annualisation.yaml.
_CFG = yaml.safe_load((Path(__file__).resolve().parents[3] / "config" / "annualisation.yaml").read_text())
GAS_SEASONAL_PROFILE = {int(k): float(v) for k, v in _CFG["gas"].items()}
ELEC_SEASONAL_PROFILE = {int(k): float(v) for k, v in _CFG["elec"].items()}


def annualise_series(months: list[int], values: list[float], fuel: str) -> tuple[float | None, str]:
    """Annualise observed monthly usage using seasonal shares.

    Returns:
        (annual_kwh, quality_flag)
    """
    valid = [(m, v) for m, v in zip(months, values) if not pd.isna(v) and v is not None]
    if not valid:
        return None, "unavailable"

    profile = GAS_SEASONAL_PROFILE if fuel == "gas" else ELEC_SEASONAL_PROFILE
    total_observed = sum(v for _, v in valid)
    total_share = sum(profile.get(m, 1.0 / 12) for m, _ in valid)

    if total_share <= 0:
        return None, "unavailable"

    # Observed use / the share of a typical year it covers (works for any number of months, incl. 2+ years)
    annual_kwh = round(float(total_observed / total_share), 1)

    # Low quality flag if < 6 distinct calendar months or seasonal coverage < 40%
    distinct_months = len(set(m for m, _ in valid))
    unique_share = sum(profile.get(m, 1.0 / 12) for m in set(m for m, _ in valid))

    if distinct_months < _CFG["low_coverage_if_fewer_distinct_months"] or unique_share < _CFG["low_coverage_if_share_below"]:
        return annual_kwh, "low_coverage"

    return annual_kwh, "good"


def annualise_dataset(
    properties_df: pd.DataFrame,
    monthly_df: pd.DataFrame,
) -> pd.DataFrame:
    """Compute annual_elec_kwh, annual_gas_kwh, and quality data_flags for all properties."""
    df_props = properties_df.copy()
    monthly = monthly_df.copy()
    monthly["month_num"] = pd.to_datetime(monthly["month"]).dt.month

    # Per-year history for mode 1: the latest 12 months of each home's series and the 12 before, each
    # annualised on its own; kept only when that year has good coverage (else None -> the 1-year model is used).
    monthly["_date"] = pd.to_datetime(monthly["month"])
    end = monthly.groupby("property_id")["_date"].transform("max")
    age = (end.dt.year - monthly["_date"].dt.year) * 12 + (end.dt.month - monthly["_date"].dt.month)
    monthly["_year"] = np.where(age < 12, "recent", np.where(age < 24, "prior", ""))
    history: dict[tuple, float | None] = {}
    for (pid, yr), g in monthly[monthly["_year"] != ""].groupby(["property_id", "_year"]):
        for fuel, col in (("elec", "elec_import_kwh"), ("gas", "gas_kwh")):
            kwh, qual = annualise_series(g["month_num"].tolist(), g[col].tolist(), fuel)
            history[(pid, yr, fuel)] = kwh if qual == "good" else None

    # Group by property
    grouped = monthly.groupby("property_id")

    elec_res = {}
    gas_res = {}

    for pid, group in grouped:
        months = group["month_num"].tolist()

        # Elec
        elec_vals = group["elec_import_kwh"].tolist()
        e_kwh, e_qual = annualise_series(months, elec_vals, "elec")
        elec_res[pid] = (e_kwh, e_qual)

        # Gas
        gas_vals = group["gas_kwh"].tolist()
        g_kwh, g_qual = annualise_series(months, gas_vals, "gas")
        gas_res[pid] = (g_kwh, g_qual)

    # Map back to properties
    elec_kwhs = []
    gas_kwhs = []
    flags_list = []

    for idx, row in df_props.iterrows():
        pid = row["property_id"]
        e_kwh, e_qual = elec_res.get(pid, (None, "unavailable"))
        g_kwh, g_qual = gas_res.get(pid, (None, "unavailable"))

        # Check existing flags
        flags = [f.strip() for f in str(row.get("data_flags", "")).split(",") if f.strip() and f != "nan"]

        if e_qual != "good":
            flags.append(f"elec_{e_qual}")
        if g_qual != "good" and row.get("gas_meter", True):
            flags.append(f"gas_{g_qual}")

        # If property has no gas meter, annual gas is 0.0
        if not row.get("gas_meter", True):
            g_kwh = 0.0

        elec_kwhs.append(e_kwh)
        gas_kwhs.append(g_kwh)
        flags_list.append(",".join(flags))

    df_props["annual_elec_kwh"] = elec_kwhs
    df_props["annual_gas_kwh"] = gas_kwhs
    df_props["data_flags"] = flags_list
    for yr in ("recent", "prior"):
        for fuel in ("elec", "gas"):
            vals = [history.get((pid, yr, fuel)) for pid in df_props["property_id"]]
            if fuel == "gas":   # no gas meter: no gas history (the gas model isn't used for these homes)
                vals = [v if bool(m) else None for v, m in zip(vals, df_props.get("gas_meter", pd.Series(True, index=df_props.index)))]
            df_props[f"{yr}_{fuel}_kwh"] = vals

    return df_props
