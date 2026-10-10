"""Annualise monthly energy consumption using seasonal weighting.

Defined in docs/ARCHITECTURE.md section 3.2.
"""

from pathlib import Path
import numpy as np
import pandas as pd

# Monthly weights representing typical UK seasonal consumption share (sum to 1.0)
# Gas: strong winter peak (heating). Elec: mild seasonal variation.
GAS_SEASONAL_PROFILE = {
    1: 0.165, 2: 0.145, 3: 0.125, 4: 0.080, 5: 0.045, 6: 0.025,
    7: 0.020, 8: 0.020, 9: 0.035, 10: 0.075, 11: 0.120, 12: 0.145,
}

ELEC_SEASONAL_PROFILE = {
    1: 0.105, 2: 0.095, 3: 0.090, 4: 0.080, 5: 0.075, 6: 0.070,
    7: 0.070, 8: 0.070, 9: 0.075, 10: 0.085, 11: 0.090, 12: 0.095,
}


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

    # Normalize if multiple years observed (total_share > 1.0)
    # Annualisation scale: observed / (total_share_per_year)
    years_represented = len(valid) / 12.0
    if total_share <= 0:
        return None, "unavailable"

    # Scale to single typical 12-month annual figure
    annual_estimate = total_observed / total_share * (total_share / max(years_represented, 1.0))
    annual_kwh = round(float(total_observed / total_share), 1)

    # Low quality flag if < 6 distinct calendar months or seasonal coverage < 40%
    distinct_months = len(set(m for m, _ in valid))
    unique_share = sum(profile.get(m, 1.0 / 12) for m in set(m for m, _ in valid))

    if distinct_months < 6 or unique_share < 0.40:
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

    return df_props
