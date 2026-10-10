"""Lender portfolio aggregator for Lloyds mortgage book.

Implements PCAF Scope 3 Category 15 financed emissions accounting.
Defined in docs/ARCHITECTURE.md section 3.12.
"""

from pathlib import Path
import pandas as pd
from greenhome.engine.carbon import DEFAULT_EF_ELEC, DEFAULT_EF_GAS
from greenhome.engine.report import build_report
from greenhome.schema import PortfolioPropertySummary, PortfolioSummary, Property


def build_portfolio(
    properties_df: pd.DataFrame,
    loans_df: pd.DataFrame | None = None,
    max_sample: int = 500,
) -> PortfolioSummary:
    """Aggregate PropertyReports into lender mortgage book summary."""
    df = properties_df.head(max_sample).copy()

    # Merge loans if provided
    if loans_df is not None:
        loans_sub = loans_df[["property_id", "ltv", "outstanding_balance"]].drop_duplicates("property_id")
        df = df.merge(loans_sub, on="property_id", how="left")
    else:
        df["ltv"] = 0.65
        df["outstanding_balance"] = df["current_value_estimate"].fillna(300000.0) * 0.65

    summaries: list[PortfolioPropertySummary] = []
    total_carbon_kg = 0.0
    total_financed_carbon_kg = 0.0
    potential_carbon_cut_kg = 0.0
    scores = []
    poor_count = 0

    for _, row in df.iterrows():
        d = row.to_dict()
        clean_d = {k: (None if pd.isna(v) else v) for k, v in d.items()}
        flags_val = clean_d.get("data_flags")
        clean_d["data_flags"] = [f.strip() for f in str(flags_val).split(",") if f.strip() and f != "nan"] if flags_val else []

        # Remove loan merge cols before creating Property
        ltv = clean_d.pop("ltv", 0.65) or 0.65
        out_bal = clean_d.pop("outstanding_balance", 200000.0) or 200000.0

        p = Property(**clean_d)
        report = build_report(p)

        scores.append(report.score.total)
        if report.is_poor:
            poor_count += 1

        # Calculate annual operational carbon
        annual_elec = p.annual_elec_kwh or 3200.0
        annual_gas = p.annual_gas_kwh if p.annual_gas_kwh is not None else (12000.0 if p.gas_meter else 0.0)
        home_carbon_kg = annual_elec * DEFAULT_EF_ELEC + annual_gas * DEFAULT_EF_GAS

        # PCAF financed carbon = (outstanding balance / property value) * home carbon
        prop_val = p.current_value_estimate or 300000.0
        attribution_factor = min(max(out_bal / max(prop_val, 10000.0), 0.0), 1.0)
        financed_carbon_kg = home_carbon_kg * attribution_factor

        total_carbon_kg += home_carbon_kg
        total_financed_carbon_kg += financed_carbon_kg
        potential_carbon_cut_kg += report.package_carbon_kg

        summaries.append(
            PortfolioPropertySummary(
                property_id=p.property_id,
                customer_id=p.customer_id,
                local_authority=p.local_authority,
                score=report.score.total,
                classification=report.score.classification,
                is_poor=report.is_poor,
                annual_carbon_kg=round(home_carbon_kg, 1),
                financed_emissions_kg=round(financed_carbon_kg, 1),
                improvement_potential_pct=report.improvement_potential_pct,
                ltv=round(ltv, 3),
                outstanding_balance=round(out_bal, 2),
            )
        )

    # Sort target properties by improvement potential
    targets = sorted(
        summaries,
        key=lambda s: (s.improvement_potential_pct or 0.0),
        reverse=True,
    )[:20]

    n = len(summaries)
    return PortfolioSummary(
        total_properties=n,
        poor_count=poor_count,
        poor_share=round(poor_count / max(n, 1), 3),
        avg_score=round(sum(scores) / max(n, 1), 1) if scores else 0.0,
        total_emissions_tonnes=round(total_carbon_kg / 1000.0, 1),
        total_financed_emissions_tonnes=round(total_financed_carbon_kg / 1000.0, 1),
        potential_carbon_reduction_tonnes=round(potential_carbon_cut_kg / 1000.0, 1),
        high_potential_targets=targets,
    )
