"""Lender portfolio aggregator for Lloyds mortgage book.

Implements PCAF Scope 3 Category 15 financed emissions accounting.
Defined in docs/ARCHITECTURE.md section 3.12.
"""

from pathlib import Path
import pandas as pd
from greenhome.engine.carbon import DEFAULT_EF_ELEC, DEFAULT_EF_GAS
from greenhome.engine.report import build_report
from greenhome.schema import PortfolioPropertySummary, PortfolioSummary, Property


def _clean(d: dict) -> dict:
    out = {k: (None if pd.isna(v) else v) for k, v in d.items() if k in Property.model_fields}
    flags = out.get("data_flags")
    out["data_flags"] = [f.strip() for f in str(flags).split(",") if f.strip() and f != "nan"] if flags else []
    return out


def _warm_energy_cache(df: pd.DataFrame) -> None:
    """Predict every home (and its gas-heated comparison home, and its bill forecast) in one batch, so the
    per-home reports below hit the model cache instead of running the model one row at a time."""
    from greenhome.models.energy import ModelUnavailable, load_model
    try:
        model = load_model("greybox")
    except ModelUnavailable:
        return
    from greenhome.engine.feasibility import RETROFIT_CATALOGUE, check_feasibility
    from greenhome.engine.savings import apply_retrofits_to_property
    rows = df.drop(columns=[c for c in ("ltv", "outstanding_balance") if c in df]).copy()
    model.forecast_frame(rows)
    states = []
    for _, row in rows.iterrows():
        try:
            p = Property(**_clean(row.to_dict()))
        except Exception:
            continue
        feasible = sorted(check_feasibility(p, [])[0], key=lambda r: RETROFIT_CATALOGUE[r]["priority"])
        for i in range(len(feasible) + 1):          # the home, then each cumulative package step
            d = apply_retrofits_to_property(p, feasible[:i]).model_dump()
            states += [d, {**d, "heating": "gas_condensing"}]   # + its gas-heated comparison home (score)
    if states:
        model.predict_frame(pd.DataFrame(states))


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
    else:  # no loan book: financed emissions are unavailable, not guessed
        df["ltv"] = None
        df["outstanding_balance"] = None

    _warm_energy_cache(df)

    summaries: list[PortfolioPropertySummary] = []
    total_carbon_kg = 0.0
    total_financed_carbon_kg = 0.0
    potential_carbon_cut_kg = 0.0
    scores = []
    poor_count = 0
    energy_unavailable = 0
    total_recs_cost = 0.0
    homes_with_recs = 0

    for _, row in df.iterrows():
        d = row.to_dict()
        clean_d = {k: (None if pd.isna(v) else v) for k, v in d.items()}
        flags_val = clean_d.get("data_flags")
        clean_d["data_flags"] = [f.strip() for f in str(flags_val).split(",") if f.strip() and f != "nan"] if flags_val else []

        # Remove loan merge cols before creating Property
        ltv = clean_d.pop("ltv", None)
        out_bal = clean_d.pop("outstanding_balance", None)

        p = Property(**clean_d)
        report = build_report(p)

        scores.append(report.score.total)
        if report.is_poor:
            poor_count += 1

        # Track average recommendation costs dynamically across homes
        if report.recommendations:
            home_recs_cost = sum((r.indicative_cost[0] + r.indicative_cost[1]) / 2.0 for r in report.recommendations)
            total_recs_cost += home_recs_cost
            homes_with_recs += 1

        # Annual operational carbon from the report's baseline energy (bills, or flagged similar-home estimate)
        home_carbon_kg = None
        if report.energy_before is not None:
            home_carbon_kg = (report.energy_before.elec_kwh * DEFAULT_EF_ELEC
                              + report.energy_before.gas_kwh * DEFAULT_EF_GAS)
            total_carbon_kg += home_carbon_kg
        else:
            energy_unavailable += 1

        # PCAF financed carbon = (outstanding balance / property value) * home carbon
        # Under PCAF mortgage standard, LTV directly equals the loan attribution factor
        financed_carbon_kg = None
        if home_carbon_kg is not None:
            attribution_factor = None
            if out_bal is not None and p.current_value_estimate and p.current_value_estimate > 0:
                attribution_factor = min(max(out_bal / p.current_value_estimate, 0.0), 1.0)
            elif ltv is not None and ltv > 0:
                attribution_factor = min(max(ltv, 0.0), 1.0)

            if attribution_factor is not None:
                financed_carbon_kg = home_carbon_kg * attribution_factor
                total_financed_carbon_kg += financed_carbon_kg
        potential_carbon_cut_kg += report.package_carbon_kg or 0.0  # unavailable homes add no claimed cut

        summaries.append(
            PortfolioPropertySummary(
                property_id=p.property_id,
                customer_id=p.customer_id,
                local_authority=p.local_authority,
                score=report.score.total,
                classification=report.score.classification,
                is_poor=report.is_poor,
                annual_carbon_kg=None if home_carbon_kg is None else round(home_carbon_kg, 1),
                financed_emissions_kg=None if financed_carbon_kg is None else round(financed_carbon_kg, 1),
                improvement_potential_pct=report.improvement_potential_pct,
                ltv=None if ltv is None else round(ltv, 3),
                outstanding_balance=None if out_bal is None else round(out_bal, 2),
            )
        )

    # Sort target properties by improvement potential
    targets = sorted(
        summaries,
        key=lambda s: (s.improvement_potential_pct or 0.0),
        reverse=True,
    )[:20]

    n = len(summaries)
    avg_recs_cost = round(total_recs_cost / max(homes_with_recs, 1), 0) if homes_with_recs else 8500.0

    return PortfolioSummary(
        total_properties=n,
        poor_count=poor_count,
        poor_share=round(poor_count / max(n, 1), 3),
        avg_score=round(sum(scores) / max(n, 1), 1) if scores else 0.0,
        total_emissions_tonnes=round(total_carbon_kg / 1000.0, 1),
        total_financed_emissions_tonnes=round(total_financed_carbon_kg / 1000.0, 1),
        potential_carbon_reduction_tonnes=round(potential_carbon_cut_kg / 1000.0, 1),
        energy_unavailable_count=energy_unavailable,
        high_potential_targets=targets,
        avg_retrofit_cost=avg_recs_cost,
    )

