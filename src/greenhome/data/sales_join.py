"""Land Registry Price Paid sales -> real EPC homes (ARCHITECTURE 3.1 join rules, 3.8 value model).

Matches each sale to one EPC home (UPRN) in the same postcode by its address numbers, then attaches the
features of the certificate lodged on or before the sale date (never a later one: that could describe
post-sale works). The match rate and a precision proxy (PPD vs EPC property type agreement) are reported.

Match tiers, tried in order; a sale is matched only when exactly one UPRN fits the tier:
  1 numbers_ordered  same postcode, same sequence of house / flat numbers (SAON then PAON vs EPC address)
  2 numbers_any      same postcode, same numbers in any order ("21, Flat 2" vs "Flat 2", "21")
  3 name             same postcode, sale has no numbers: its house name (and flat name) appear in the EPC address

    uv run python -m greenhome.data.sales_join      (after make epc)

Outputs (gitignored; raw-derived, no address or postcode):
  data/processed/real/sales.parquet        one row per matched sale, historical price kept separate from
                                           current_value_estimate
  eval/results/sales_join_summary.json     match rates by tier and local authority (committed)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from greenhome.data.calibrate import PPD_COLS
from greenhome.data.epc_ingest import CFG, EPC_COLS, EPC_PATH, map_certificates

ROOT = Path(__file__).resolve().parents[3]
PPD_DIR = ROOT / "data" / "raw" / "ppd"
BOOK = ROOT / "data" / "processed" / "epc_properties.parquet"
OUT = ROOT / "data" / "processed" / "real" / "sales.parquet"
SUMMARY = ROOT / "eval" / "results" / "sales_join_summary.json"

ADDRESS_COLS = ["address1", "address2", "address3"]
UNIT_WORDS = r"\b(FLAT|FLATS|APARTMENT|APARTMENTS|APPARTMENT|APT|UNIT|NO)\b"
# PPD property type -> EPC/NEED types it is consistent with (diagnostic only, used as a precision proxy)
TYPE_AGREES = {"F": {"Flat"}, "D": {"Detached", "Bungalow"}, "S": {"Semi detached", "Bungalow"},
               "T": {"Mid terrace", "End terrace", "Bungalow"}}


def norm(s: pd.Series) -> pd.Series:
    """Uppercase words; punctuation and unit words (FLAT, APARTMENT...) removed; "21 A" -> "21A"."""
    s = s.fillna("").astype(str).str.upper()
    s = s.str.replace(r"[^A-Z0-9 ]", " ", regex=True).str.replace(UNIT_WORDS, " ", regex=True)
    s = s.str.replace(r"\b(\d+) ([A-Z])\b", r"\1\2", regex=True)
    return s.str.replace(r"\s+", " ", regex=True).str.strip()


def numbers(s: pd.Series) -> pd.Series:
    """Ordered house / flat number tokens as a key ("2|21"); empty when the address has none."""
    return s.str.findall(r"\b\d+[A-Z]?\b").map("|".join)


def property_id(uprn: pd.Series) -> pd.Series:
    return uprn.map(lambda u: "E" + hashlib.sha1(str(int(u)).encode()).hexdigest()[:10])  # same as epc_ingest


def load_sales() -> tuple[pd.DataFrame, dict]:
    ppd = pd.concat([pd.read_csv(f, header=None, names=PPD_COLS, dtype=str)
                     for f in sorted(PPD_DIR.glob("pp-*.csv"))], ignore_index=True)
    las = [la.upper() for la in CFG["local_authorities"]]
    counts = {"ppd_rows": len(ppd)}
    ppd = ppd[ppd.district.isin(las)]
    counts["in_local_authorities"] = len(ppd)
    ppd = ppd[ppd.ppd_cat == "A"]   # B = repossessions, bulk / company sales: not market prices of one home
    counts["category_a"] = len(ppd)
    ppd = ppd[ppd.ptype != "O"]     # O = other (commercial, land, mixed)
    counts["residential"] = len(ppd)
    ppd = ppd[ppd.postcode.notna()].copy()
    counts["with_postcode"] = len(ppd)
    ppd["sale_id"] = ppd.tid.str.strip("{}")
    ppd["sale_date"] = pd.to_datetime(ppd.date.str[:10])
    ppd["price"] = ppd.price.astype(float)
    ppd["postcode"] = ppd.postcode.str.upper().str.strip()
    ppd["key"] = norm(ppd.saon.fillna("") + " " + ppd.paon.fillna(""))
    ppd["nums"] = numbers(ppd.key)
    return ppd, counts


def load_certificates() -> pd.DataFrame:
    d = pd.read_csv(EPC_PATH, usecols=EPC_COLS + ADDRESS_COLS, low_memory=False)
    d = d[d.uprn.notna() & (d.property_type != "Park home")].copy()
    d["uprn"] = d.uprn.astype("int64")
    d["lodgement_date"] = pd.to_datetime(d.lodgement_date, errors="coerce")
    d["postcode"] = d.postcode.str.upper().str.strip()
    d["addr"] = norm(d[ADDRESS_COLS].fillna("").agg(" ".join, axis=1))
    d["nums"] = numbers(d.addr)
    return d


def unique_match(pairs: pd.DataFrame) -> pd.DataFrame:
    """Sales whose candidate pairs point at exactly one UPRN -> (sale_id, uprn)."""
    n = pairs.groupby("sale_id").uprn.nunique()
    one = n.index[n == 1]
    return pairs[pairs.sale_id.isin(one)].drop_duplicates("sale_id")[["sale_id", "uprn"]]


def match(sales: pd.DataFrame, certs: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    addr = certs[["uprn", "postcode", "addr", "nums"]].drop_duplicates()
    addr = addr.assign(nums_sorted=addr.nums.str.split("|").map(lambda t: "|".join(sorted(t))))
    s = sales[["sale_id", "postcode", "key", "nums"]]
    s = s.assign(nums_sorted=s.nums.str.split("|").map(lambda t: "|".join(sorted(t))))
    found, ambiguous, tiers = [], set(), {}
    left = s[s.postcode.isin(set(addr.postcode))]
    no_epc_postcode = len(s) - len(left)

    def tier(name: str, pairs: pd.DataFrame) -> None:
        nonlocal left
        m = unique_match(pairs)
        ambiguous.update(set(pairs.sale_id) - set(m.sale_id))
        tiers[name] = len(m)
        found.append(m.assign(match_tier=name))
        left = left[~left.sale_id.isin(m.sale_id)]

    with_nums = lambda df: df[df.nums != ""]
    tier("numbers_ordered", with_nums(left).merge(addr, on=["postcode", "nums"])[["sale_id", "uprn"]])
    tier("numbers_any", with_nums(left).merge(addr, on=["postcode", "nums_sorted"])[["sale_id", "uprn"]])
    named = left[(left.nums == "") & (left.key != "")].merge(addr[["uprn", "postcode", "addr"]], on="postcode")
    hit = [bool(re.search(rf"\b{re.escape(k)}\b", a)) for k, a in zip(named.key, named.addr)]
    tier("name", named[hit][["sale_id", "uprn"]])

    m = pd.concat(found, ignore_index=True)
    ambiguous -= set(m.sale_id)
    stats = {"no_epc_in_postcode": no_epc_postcode, "ambiguous": len(ambiguous), "by_tier": tiers}
    return m, stats


def pre_sale_features(matched: pd.DataFrame, certs: pd.DataFrame) -> pd.DataFrame:
    """Attach the latest certificate lodged on or before each sale date (leakage rule, ARCHITECTURE 3.1)."""
    c = certs[certs.lodgement_date.notna()].sort_values("lodgement_date")
    m = matched.sort_values("sale_date")
    j = pd.merge_asof(m, c[["uprn", "lodgement_date"]].assign(epc_lodgement_date=c.lodgement_date),
                      left_on="sale_date", right_on="lodgement_date", by="uprn", direction="backward")
    rows = j[j.epc_lodgement_date.notna()]
    cert = c.drop_duplicates(["uprn", "lodgement_date"], keep="last").set_index(["uprn", "lodgement_date"])
    picked = cert.loc[list(zip(rows.uprn, rows.epc_lodgement_date))].reset_index()
    feats = map_certificates(picked)[["local_authority", "region", "property_type", "build_year", "age_band",
                                      "floor_area_m2", "energy_rating", "sap_score", "bedrooms"]]
    feats.index = rows.index
    out = j.drop(columns=["lodgement_date"]).join(feats)
    out["has_pre_sale_epc"] = out.epc_lodgement_date.notna()
    return out


def join() -> tuple[pd.DataFrame, dict]:
    sales, counts = load_sales()
    certs = load_certificates()
    m, stats = match(sales, certs)
    m = m.merge(sales[["sale_id", "sale_date", "price", "ptype", "new_build", "tenure", "district"]], on="sale_id")
    m["property_id"] = property_id(m.uprn)
    in_book = set(pd.read_parquet(BOOK, columns=["property_id"]).property_id)
    not_in_book = int((~m.property_id.isin(in_book)).sum())  # UPRN whose latest certificate isn't in the book
    m = m[m.property_id.isin(in_book)]
    out = pre_sale_features(m, certs)
    out = out.rename(columns={"ptype": "ppd_property_type", "tenure": "ppd_tenure"})
    agrees = [t in TYPE_AGREES.get(p, set()) for p, t in zip(out.ppd_property_type, out.property_type)]
    out["type_agrees"] = np.where(out.has_pre_sale_epc, agrees, np.nan)

    n = counts["with_postcode"]
    by_la = out.groupby("district").size() / sales.groupby("district").size()
    summary = {
        "source": f"HM Land Registry Price Paid {sales.sale_date.dt.year.min()}-{sales.sale_date.dt.year.max()} "
                  "joined to EPC certificates by postcode + address numbers",
        "sales_filtering": counts,
        "matched_sales": len(out),
        "match_rate": round(len(out) / n, 4),
        "match_rate_by_local_authority": {k.title(): round(float(v), 4) for k, v in by_la.items()},
        "matched_by_tier": {t: int((out.match_tier == t).sum()) for t in stats["by_tier"]},
        "unmatched": {"no_epc_in_postcode": stats["no_epc_in_postcode"], "ambiguous": stats["ambiguous"],
                      "home_not_in_book": not_in_book,
                      "no_address_fit": n - sum(stats["by_tier"].values()) - stats["no_epc_in_postcode"] - stats["ambiguous"]},
        "with_pre_sale_epc": int(out.has_pre_sale_epc.sum()),
        "type_agreement_by_tier": {t: round(float(g.type_agrees.mean()), 4)
                                   for t, g in out[out.has_pre_sale_epc].groupby("match_tier")},
        "homes_with_a_sale": int(out.property_id.nunique()),
        "share_of_book_with_a_sale": round(out.property_id.nunique() / len(in_book), 4),
    }
    cols = ["sale_id", "property_id", "sale_date", "price", "ppd_property_type", "new_build", "ppd_tenure",
            "match_tier", "has_pre_sale_epc", "epc_lodgement_date", "local_authority", "region", "property_type",
            "build_year", "age_band", "floor_area_m2", "energy_rating", "sap_score", "bedrooms", "type_agrees"]
    return out[cols].sort_values(["property_id", "sale_date"]).reset_index(drop=True), summary


def main() -> None:
    out, summary = join()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT, index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
