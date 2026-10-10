"""current_value_estimate for the real EPC homes: last sale price indexed to today with UK HPI (ARCHITECTURE 3.1).

value = last sale price x HPI(type, LA, latest month) / HPI(type, LA, sale month), with the HPI index for the
sale's own property type (detached / semi / terraced / flat) in its local authority. Homes with no matched sale
since 2018 stay unavailable (NaN) until the value model (3.8) predicts them; nothing is imputed here.
The historical sale `price` stays in sales.parquet, separate from this estimate.

    uv run python -m greenhome.data.valuation      (after make real and make sales)

Updates data/processed/real/{properties,properties_share,loans}.parquet in place (recomputed from scratch on
every run) and writes data/processed/real/valuations.parquet plus eval/results/valuation_summary.json.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "data" / "processed" / "real"
HPI_PATH = ROOT / "data" / "raw" / "hpi" / "UK-HPI-full-file-2026-07.csv"
HPI_SOURCE = ("UK House Price Index July 2026, local-authority index by property type "
              "(https://www.gov.uk/government/statistical-data-sets/uk-house-price-index-data-downloads-july-2026)")
SUMMARY = ROOT / "eval" / "results" / "valuation_summary.json"
HPI_BY_PPD_TYPE = {"D": "DetachedIndex", "S": "SemiDetachedIndex", "T": "TerracedIndex", "F": "FlatIndex"}
FLAG = "value_hpi_indexed_last_sale"


def load_hpi() -> pd.DataFrame:
    h = pd.read_csv(HPI_PATH, usecols=["Date", "RegionName", *HPI_BY_PPD_TYPE.values()])
    h["Date"] = pd.to_datetime(h.Date, format="%d/%m/%Y")
    return h.rename(columns={"RegionName": "local_authority", "Date": "month"})


def value_homes(sales: pd.DataFrame, hpi: pd.DataFrame) -> tuple[pd.DataFrame, pd.Timestamp]:
    h = hpi[hpi.local_authority.isin(sales.local_authority.dropna().unique())]
    latest = h.month.max()
    long = h.melt(id_vars=["local_authority", "month"], var_name="index_col", value_name="hpi")
    last = sales.sort_values("sale_date").drop_duplicates("property_id", keep="last").copy()
    last["month"] = last.sale_date.dt.to_period("M").dt.to_timestamp()
    last["index_col"] = last.ppd_property_type.map(HPI_BY_PPD_TYPE)
    v = last.merge(long, on=["local_authority", "month", "index_col"], how="left")
    now = long[long.month == latest].rename(columns={"hpi": "hpi_now"}).drop(columns="month")
    v = v.merge(now, on=["local_authority", "index_col"], how="left")
    v["current_value_estimate"] = (v.price * v.hpi_now / v.hpi).round(-3)
    cols = ["property_id", "sale_date", "price", "ppd_property_type", "new_build", "hpi", "hpi_now",
            "current_value_estimate"]
    return v[cols].rename(columns={"sale_date": "last_sale_date", "price": "last_sale_price"}), latest


def apply(props: pd.DataFrame, values: pd.DataFrame) -> pd.DataFrame:
    p = props.drop(columns="current_value_estimate").merge(
        values[["property_id", "current_value_estimate"]], on="property_id", how="left")
    flags = p.data_flags.fillna("").str.split(",").map(lambda f: [x for x in f if x and x != FLAG])
    p["data_flags"] = [",".join(f + [FLAG] if v == v else f)  # v == v: not NaN
                       for f, v in zip(flags, p.current_value_estimate)]
    return p[props.columns]


def main() -> None:
    sales = pd.read_parquet(REAL / "sales.parquet")
    # The book's local authority (sales rows without a pre-sale EPC have no features of their own)
    props = pd.read_parquet(REAL / "properties.parquet")
    sales = sales.drop(columns="local_authority").merge(props[["property_id", "local_authority"]], on="property_id")
    values, latest = value_homes(sales, load_hpi())
    values = values[values.current_value_estimate.notna()]

    props = apply(props, values)
    props.to_parquet(REAL / "properties.parquet", index=False)
    share = pd.read_parquet(REAL / "properties_share.parquet")
    apply(share, values).to_parquet(REAL / "properties_share.parquet", index=False)
    loans = pd.read_parquet(REAL / "loans.parquet").drop(columns="outstanding_balance").merge(
        props[["property_id", "current_value_estimate"]], on="property_id", how="left")
    loans["outstanding_balance"] = (loans.ltv * loans.current_value_estimate).round(-2)  # same rule as make_loans
    loans.drop(columns="current_value_estimate").to_parquet(REAL / "loans.parquet", index=False)
    values.to_parquet(REAL / "valuations.parquet", index=False)

    v = props.current_value_estimate
    summary = {
        "source": HPI_SOURCE, "valuation_month": str(latest.date()),
        "method": "last matched Price Paid sale (2018-2025) x HPI(type, LA, valuation month) / HPI(type, LA, sale month)",
        "homes": len(props), "homes_valued": int(v.notna().sum()), "share_valued": round(float(v.notna().mean()), 4),
        "median_value_by_local_authority": props.groupby("local_authority").current_value_estimate.median().round(-3).to_dict(),
        "median_index_ratio": round(float((values.hpi_now / values.hpi).median()), 4),
        "valued_from_new_build_sale": int((values.new_build == "Y").sum()),
        "loans_with_balance": int(loans.outstanding_balance.notna().sum()), "loans": len(loans),
    }
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
