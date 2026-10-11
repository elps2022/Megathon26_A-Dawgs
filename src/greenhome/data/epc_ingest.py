"""EPC certificates -> real property book (ARCHITECTURE 3.1).

Keeps the latest certificate per UPRN, maps EPC fields to the Property schema vocabulary with the rules in
config/epc_mapping.yaml, and joins IMD 2019 quintile via ONSPD postcode -> LSOA 2011. Anything assumed or
unparseable is flagged in data_flags, never guessed silently.

    uv run python -m greenhome.data.epc_ingest

Outputs (gitignored; raw-derived):
  data/processed/epc_properties.parquet        full book, incl. postcode (internal joins only)
  data/processed/epc_properties_share.parquet  no postcode / UPRN, safe to share on Drive
  eval/results/epc_ingest_summary.json         aggregate counts and mapping coverage (committed)
"""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[3]
RAW = ROOT / "data" / "raw"
EPC_PATH = RAW / "epc" / "epc_domestic_leeds_cornwall_southwark.csv"
OUT = ROOT / "data" / "processed"
SUMMARY = ROOT / "eval" / "results" / "epc_ingest_summary.json"
CFG = yaml.safe_load((ROOT / "config" / "epc_mapping.yaml").read_text())

EPC_COLS = ["uprn", "postcode", "lodgement_date", "local_authority_label", "property_type", "built_form",
            "construction_age_band", "total_floor_area", "current_energy_rating", "current_energy_efficiency",
            "mainheat_description", "mainheat_energy_eff", "main_fuel", "mains_gas_flag", "walls_description",
            "roof_description", "windows_description", "glazed_type", "mainheatcont_description", "photo_supply",
            "number_habitable_rooms", "tenure"]


def first_match(text: pd.Series, rules: list) -> pd.Series:
    """Apply ordered [pattern, value] rules; first match wins; NaN where nothing matches."""
    out = pd.Series(np.nan, index=text.index, dtype=object)
    t = text.fillna("").str.lower()
    for pattern, value in rules:
        hit = out.isna() & t.str.contains(pattern, regex=True)
        out[hit] = value
    return out


def build_year(band: pd.Series) -> pd.Series:
    """Representative construction year: the band's upper year, or the year itself for new-build entries."""
    s = band.fillna("").astype(str)
    years = s.str.findall(r"(\d{4})")
    upper = years.map(lambda ys: int(ys[-1]) if ys else np.nan)
    upper[s.str.contains("before", case=False) & upper.notna()] = upper - 1
    return upper


def age_band(year: pd.Series) -> pd.Series:
    out = pd.Series(np.nan, index=year.index, dtype=object)
    for upper, label in CFG["age_band_upper_years"]:
        out[out.isna() & (year <= upper)] = label
    return out


def u_value(text: pd.Series) -> pd.Series:
    return pd.to_numeric(text.str.extract(r"transmittance\s+([\d.]+)", flags=re.I)[0], errors="coerce")


def map_certificates(d: pd.DataFrame) -> pd.DataFrame:
    flags = pd.Series([[] for _ in range(len(d))], index=d.index)

    def flag(mask: pd.Series, name: str) -> None:
        for i in d.index[mask.fillna(False)]:
            flags[i].append(name)

    x = pd.DataFrame(index=d.index)
    x["property_id"] = d.uprn.map(lambda u: "E" + hashlib.sha1(str(int(u)).encode()).hexdigest()[:10])
    x["local_authority"] = d.local_authority_label
    x["region"] = d.local_authority_label.map(CFG["local_authorities"])
    x["postcode"] = d.postcode.str.upper().str.strip()
    x["lodgement_date"] = pd.to_datetime(d.lodgement_date, errors="coerce")
    x["tenure"] = d.tenure.str.lower().replace({"unknown": np.nan})

    # Property type in NEED vocabulary
    ptype = d.property_type.map(CFG["property_type"])
    house = d.property_type == "House"
    ptype[house] = d.built_form[house].map(CFG["built_form_for_houses"])
    x["property_type"] = ptype
    flag(house & ptype.isna(), "unknown_built_form")

    year = build_year(d.construction_age_band)
    x["build_year"] = year
    x["age_band"] = age_band(year)
    flag(x.age_band.isna(), "unknown_age_band")

    lo, hi = CFG["floor_area_plausible_m2"]
    area = d.total_floor_area.where(d.total_floor_area.between(lo, hi))
    x["floor_area_m2"] = area
    flag(area.isna(), "implausible_floor_area")

    x["energy_rating"] = d.current_energy_rating.where(d.current_energy_rating.isin(list("ABCDEFG")))
    x["sap_score"] = d.current_energy_efficiency.astype(float)

    # Heating
    heating = first_match(d.mainheat_description, CFG["heating"])
    condensing = d.mainheat_energy_eff.isin(CFG["gas_boiler_condensing_if_mainheat_energy_eff"])
    heating[heating == "GAS"] = np.where(condensing[heating == "GAS"], "gas_condensing", "gas_non_condensing")
    x["heating"] = heating
    flag(heating.isna(), "unknown_heating")
    flag(heating.isin(CFG["unmetered_heating"]), "heating_fuel_unmetered")  # oil / LPG / district: not in bills
    gas_flag = d.mains_gas_flag.map({"Y": True, "N": False})
    inferred = gas_flag.isna()
    gas_flag[inferred] = d.main_fuel[inferred].fillna("").str.contains("mains gas", case=False)
    x["gas_meter"] = gas_flag.astype(bool)
    flag(inferred, "gas_meter_inferred")

    # Walls
    w = CFG["walls"]
    x["wall_type"] = first_match(d.walls_description, w["construction"])
    flag(x.wall_type.isna(), "unknown_wall_type")
    ins = first_match(d.walls_description, w["insulation"])
    wu = u_value(d.walls_description)
    ins[ins.isna() & wu.notna()] = np.where(wu[ins.isna() & wu.notna()] <= w["u_value_full_if_at_most"], "full", "partial")
    x["wall_insulation"] = ins
    x["cavity_filled_retrofit"] = d.walls_description.str.contains(w["retrofit_fill"], case=False, na=False)
    flag(ins.isna(), "unknown_wall_insulation")
    flag(d.walls_description.str.contains("assumed", case=False, na=False), "wall_insulation_assumed")

    # Roof / loft
    r = CFG["roof"]
    roof = d.roof_description.fillna("").str.lower()
    no_roof = roof.str.contains("|".join(r["no_own_roof"]))
    x["has_own_roof"] = ~no_roof
    mm = pd.to_numeric(roof.str.extract(r["mm_pattern"])[0], errors="coerce")
    phrase_mm = pd.to_numeric(first_match(roof, r["phrases"]), errors="coerce")
    ru = u_value(d.roof_description)
    u_mm = pd.Series(np.nan, index=d.index)
    for upper, val in r["u_value_mm"]:
        u_mm[u_mm.isna() & (ru <= upper)] = val
    loft = mm.fillna(u_mm).fillna(phrase_mm)
    loft[no_roof] = np.nan  # not applicable: no roof of its own
    x["loft_insulation_mm"] = loft
    flag(~no_roof & mm.isna() & loft.notna(), "loft_mm_assumed_from_description")
    flag(~no_roof & loft.isna(), "unknown_loft_insulation")

    # Glazing
    g = first_match(d.windows_description, CFG["glazing"])
    gt = d.glazed_type.fillna("").str.lower()
    dbl = g == "DOUBLE"
    is_new = gt.str.contains("|".join(CFG["glazed_type_new"]))
    is_old = gt.str.contains("|".join(CFG["glazed_type_old"]))
    unknown_date = dbl & ~is_new & ~is_old
    g[dbl] = np.where(is_new[dbl] | (unknown_date[dbl] & (year[dbl] >= CFG["double_unknown_date_new_if_built_from"])),
                      "double_new", "double_old")
    x["glazing"] = g
    flag(unknown_date, "glazing_install_date_unknown")
    flag(g.isna(), "unknown_glazing")

    x["smart_controls"] = first_match(d.mainheatcont_description, CFG["controls"])
    flag(pd.Series(True, index=d.index), "smart_controls_from_epc")

    # Solar: % of roof covered -> kWp
    s = CFG["solar"]
    storeys = x.property_type.map(s["storeys"]).fillna(s["storeys"]["default"])
    pct = d.photo_supply.fillna(0).clip(0, 100)
    x["solar_kwp"] = (pct / 100 * (area / storeys) * s["kwp_per_m2_covered"]).fillna(0).round(2)
    flag(d.photo_supply.isna(), "solar_unknown_assumed_none")

    rooms = d.number_habitable_rooms
    x["bedrooms"] = (rooms - CFG["bedrooms_from_habitable_rooms_minus"]).clip(1, 6).astype("Int64")
    flag(rooms.notna(), "bedrooms_estimated_from_rooms")

    # Property requires strings here; "unknown" is scored/handled as unknown downstream (already flagged above).
    for col in ("property_type", "heating", "wall_type", "glazing"):
        x[col] = x[col].fillna("unknown")
    x["data_flags"] = flags.map(lambda f: ",".join(f))
    return x


def join_imd(x: pd.DataFrame) -> pd.Series:
    onspd = pd.concat([pd.read_csv(f, usecols=["pcds", "lsoa11cd"], dtype=str)
                       for f in sorted((RAW / "onspd").glob("ONSPD_*_UK_*.csv"))])
    onspd["pcds"] = onspd.pcds.str.upper().str.strip()
    imd = pd.read_excel(next((RAW / "imd").glob("*IMD2019*.xlsx")), sheet_name=1)
    code_col = next(c for c in imd.columns if "LSOA code" in c)
    decile_col = next(c for c in imd.columns if "Decile" in c)
    imd = imd[[code_col, decile_col]].rename(columns={code_col: "lsoa11cd", decile_col: "imd_decile"})
    m = x[["postcode"]].merge(onspd, left_on="postcode", right_on="pcds", how="left").merge(imd, on="lsoa11cd", how="left")
    m.index = x.index
    return ((m.imd_decile + 1) // 2).astype("Int64")  # NEED uses quintiles: 1 = most deprived


def ingest() -> tuple[pd.DataFrame, dict]:
    d = pd.read_csv(EPC_PATH, usecols=EPC_COLS, low_memory=False)
    n_certs = len(d)
    d = d[d.uprn.notna()]
    n_with_uprn = len(d)
    d["lodgement_date"] = pd.to_datetime(d.lodgement_date, errors="coerce")
    d = d.sort_values("lodgement_date").drop_duplicates("uprn", keep="last")  # latest certificate per home
    park = d.property_type == "Park home"
    d = d[~park].reset_index(drop=True)

    x = map_certificates(d)
    x["imd_quintile"] = join_imd(x)
    x["annual_elec_kwh"] = np.nan   # filled by the synthetic-bills step (ARCHITECTURE 3.1)
    x["annual_gas_kwh"] = np.nan
    x["current_value_estimate"] = np.nan  # filled by the Price Paid + HPI step

    flag_counts = pd.Series([f for fl in x.data_flags for f in fl.split(",") if f]).value_counts()
    summary = {
        "source": "EPC domestic certificates, Leeds / Cornwall / Southwark (2012-01 to 2026-09)",
        "certificates": n_certs, "with_uprn": n_with_uprn, "homes_latest_certificate": int(len(d) + park.sum()),
        "park_homes_excluded": int(park.sum()), "homes_in_book": len(x),
        "by_local_authority": x.local_authority.value_counts().to_dict(),
        "mapped_share": {c: round(float(x[c].notna().mean()), 4) for c in
                         ["property_type", "age_band", "floor_area_m2", "energy_rating", "heating", "wall_type",
                          "wall_insulation", "glazing", "imd_quintile"]},
        "loft_known_share_own_roof": round(float(x.loc[x.has_own_roof, "loft_insulation_mm"].notna().mean()), 4),
        "value_counts": {c: {str(k): int(v) for k, v in x[c].value_counts(dropna=False).items()} for c in
                         ["property_type", "age_band", "heating", "wall_type", "wall_insulation", "glazing",
                          "smart_controls", "energy_rating", "imd_quintile"]},
        "with_solar": int((x.solar_kwp > 0).sum()),
        "flags": flag_counts.astype(int).to_dict(),
    }
    return x, summary


def main() -> None:
    x, summary = ingest()
    OUT.mkdir(parents=True, exist_ok=True)
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    x.to_parquet(OUT / "epc_properties.parquet", index=False)
    x.drop(columns=["postcode"]).to_parquet(OUT / "epc_properties_share.parquet", index=False)
    SUMMARY.write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: summary[k] for k in ["certificates", "with_uprn", "park_homes_excluded", "homes_in_book",
                                              "by_local_authority", "mapped_share", "loft_known_share_own_roof"]},
                     indent=2, default=str))


if __name__ == "__main__":
    main()
