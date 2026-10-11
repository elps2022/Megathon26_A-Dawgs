"""Value model (ARCHITECTURE 3.8): an indicative value range for every home and the rating-band uplift.

Target. Each real matched sale (data/data/sales_join: Price Paid joined to the certificate lodged BEFORE the sale)
is indexed to the valuation month with the official UK HPI for its type and local authority:
    target = log(price x HPI(type, LA, valuation month) / HPI(type, LA, sale month))
so the model learns what a home is worth from its description, the index carries the market over time, and a
prediction is directly today's value (deployment needs no HPI file). New builds are excluded (their premium
doesn't apply when valuing an existing home).

Models (fixed customer split, data/processed/real/splits.parquet, so all sales of a home sit on one side):
  - level: LightGBM on the description (LA, type, floor area, bedrooms, age, SAP score / band, deprivation),
    monotone in rating (better rating never lowers value), with conformalised p10 / p90 (CQR, as the energy
    model: quantile GBDTs on train, one correction from val).
  - this home: the home's own last sale indexed to today (what data/valuation.py does), and a blend of the two,
    weight fitted on val.
  - rating premium: a hedonic log-linear model (log value on LA x type, log area, age band, bedrooms,
    deprivation and EPC band) with POSTCODE-SECTOR fixed effects (homes compared with others on the same
    streets), standard errors clustered by sector. Without them the premium came out backwards (F/G +6.9% vs D):
    within an LA, poorly rated homes are often expensive central period houses and A/B homes newer flats in
    cheaper areas. Band coefficients = % premium with a CI. Uplift = premium(band after) - premium(band before)
    for the same home on the same date. Not a tree: a tree's before / after difference is step-shaped and often
    exactly 0. A repeat-sales estimate (the same home sold twice, re-rated in between) is reported as an upper
    bound: re-rated homes have often been renovated too. Postcodes come from the raw-derived property file
    (training only); deployment needs only the band premia.
  - rating after a retrofit: a linear model of SAP score on the fabric fields (loft depth, wall insulation,
    glazing, heating, controls, solar) with type / age / size / LA controls, fitted on the real certificates
    of train-split customers; a retrofit's SAP gain is the model's change when the field changes.
Every value number is an association in historical sales, not a causal effect, and is labelled so.

    uv run python -m greenhome.models.value_model      -> artifacts (value_*.txt, value_meta.json), eval/results/value_v1.json
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold

from greenhome.models.energy import ARTIFACTS, ModelUnavailable

ROOT = Path(__file__).resolve().parents[3]
REAL = ROOT / "data" / "processed" / "real"
RESULTS = ROOT / "eval" / "results"
META = "value_meta.json"
BANDS = list("ABCDEFG")
HEDONIC_BANDS = ["A/B", "C", "D", "E", "F/G"]          # A and G are rare: pooled, as NEED does
BAND_GROUP = {"A": "A/B", "B": "A/B", "C": "C", "D": "D", "E": "E", "F": "F/G", "G": "F/G"}
AGE = {"pre-1930": 1, "1930-1972": 2, "1973-1999": 3, "2000+": 4}
TYPES = ["Bungalow", "Detached", "End terrace", "Flat", "Mid terrace", "Semi detached"]
LAS = ["Cornwall", "Leeds", "Southwark"]
FEATURES = ["local_authority", "property_type", "floor_area_m2", "bedrooms", "age_band", "sap_score", "epc_band",
            "imd_quintile"]
PARAMS = dict(objective="huber", alpha=0.3, learning_rate=0.05, num_leaves=63, min_child_samples=50,
              feature_fraction=0.9, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0, n_estimators=4000,
              verbose=-1, random_state=42)
BAND = (0.1, 0.9)
CAVEAT = ("Indicative estimate based on associations between energy ratings and sale prices in historical sales "
          "in this area; not a valuation and not a guaranteed or causal increase.")


# ---------------------------------------------------------------- features

def features(df: pd.DataFrame) -> pd.DataFrame:
    """Model inputs for sales rows or Property rows (same columns, codes and dtypes)."""
    x = pd.DataFrame(index=df.index)
    x["local_authority"] = pd.Categorical(df.local_authority, categories=LAS)
    x["property_type"] = pd.Categorical(df.property_type.where(df.property_type.isin(TYPES)), categories=TYPES)
    x["floor_area_m2"] = pd.to_numeric(df.floor_area_m2, errors="coerce")
    x["bedrooms"] = pd.to_numeric(df.get("bedrooms"), errors="coerce").astype(float)
    x["age_band"] = df.age_band.map(AGE).astype(float)
    x["sap_score"] = pd.to_numeric(df.sap_score, errors="coerce")
    x["epc_band"] = df.energy_rating.map({b: i + 1 for i, b in enumerate(BANDS)}).astype(float)  # 1 = A ... 7 = G
    x["imd_quintile"] = pd.to_numeric(df.get("imd_quintile"), errors="coerce").astype(float)
    return x[FEATURES]


def monotone(cols: list[str]) -> list[int]:
    return [1 if c == "sap_score" else -1 if c == "epc_band" else 0 for c in cols]


def training_sales() -> pd.DataFrame:
    """Matched resales with a pre-sale certificate, indexed to the valuation month, with the customer split."""
    from greenhome.data.valuation import HPI_BY_PPD_TYPE, load_hpi
    s = pd.read_parquet(REAL / "sales.parquet")
    props = pd.read_parquet(REAL / "properties_share.parquet", columns=["property_id", "customer_id", "imd_quintile"])
    split = pd.read_parquet(REAL / "splits.parquet")
    s = s[s.has_pre_sale_epc & (s.new_build == "N")].merge(props, on="property_id").merge(split, on="customer_id")
    pc = pd.read_parquet(REAL / "properties.parquet", columns=["property_id", "postcode"])   # raw-derived, training only
    s = s.merge(pc, on="property_id", how="left")
    s["postcode_sector"] = s.postcode.str.upper().str.strip().str[:-2].str.strip()
    s = s.drop(columns="postcode")
    h = load_hpi()
    h = h[h.local_authority.isin(LAS)]
    latest = h.month.max()
    long = h.melt(id_vars=["local_authority", "month"], var_name="index_col", value_name="hpi")
    now = long[long.month == latest].drop(columns="month").rename(columns={"hpi": "hpi_now"})
    s["month"] = s.sale_date.dt.to_period("M").dt.to_timestamp()
    s["index_col"] = s.ppd_property_type.map(HPI_BY_PPD_TYPE)
    s = s.merge(long, on=["local_authority", "month", "index_col"], how="left").merge(now, on=["local_authority", "index_col"])
    s = s[s.hpi.notna() & (s.price > 10_000)]
    s["log_value_now"] = np.log(s.price * s.hpi_now / s.hpi)
    # This home: its previous sale (strictly earlier), indexed to now with the same index.
    s = s.sort_values(["property_id", "sale_date"])
    s["prior_log_value_now"] = s.groupby("property_id").log_value_now.shift(1)
    s.attrs["valuation_month"] = str(latest.date())
    return s.reset_index(drop=True)


# ---------------------------------------------------------------- models

def fit_level(x, y, objective: dict | None = None):
    es = np.random.default_rng(0).random(len(x)) < 0.15
    params = {**PARAMS, **(objective or {})}
    # LightGBM can't combine monotone constraints with the quantile objective: only the band edges go without them
    mono = {} if params["objective"] == "quantile" else {"monotone_constraints": monotone(list(x.columns))}
    m = lgb.LGBMRegressor(**params, **mono)
    m.fit(x[~es], y[~es], eval_X=(x[es],), eval_y=(y[es],), callbacks=[lgb.early_stopping(100, verbose=False)])
    return m


def hedonic_design(df: pd.DataFrame) -> pd.DataFrame:
    """LA x type, log area, age band, bedrooms (incl. unknown), deprivation (incl. unknown), EPC band (base D)."""
    x = pd.DataFrame(index=df.index)
    la_type = df.local_authority.astype(str) + "|" + df.property_type.astype(str)
    for c in sorted(f"{a}|{t}" for a in LAS for t in TYPES):
        x[f"cell={c}"] = (la_type == c).astype(float)
    x["log_area"] = np.log(pd.to_numeric(df.floor_area_m2, errors="coerce").clip(15, 600))
    for a in ("1930-1972", "1973-1999", "2000+"):
        x[f"age={a}"] = (df.age_band == a).astype(float)
    beds = pd.to_numeric(df.get("bedrooms"), errors="coerce").clip(upper=6)
    for b in (1, 3, 4, 5, 6):
        x[f"beds={b}"] = (beds == b).astype(float)
    x["beds=unknown"] = beds.isna().astype(float)
    imd = pd.to_numeric(df.get("imd_quintile"), errors="coerce")
    for q in (1, 2, 4, 5):
        x[f"imd={q}"] = (imd == q).astype(float)
    x["imd=unknown"] = imd.isna().astype(float)
    band = df.energy_rating.map(BAND_GROUP)
    for b in HEDONIC_BANDS:
        if b != "D":
            x[f"epc={b}"] = (band == b).astype(float)
    return x


def fe_ols(X: pd.DataFrame, y: pd.Series, groups: pd.Series) -> tuple[pd.Series, pd.DataFrame, float]:
    """OLS with group fixed effects (within transformation) and group-clustered covariance; returns the
    coefficients that vary within groups, their covariance and the within-group R2."""
    Xd = X - X.groupby(groups.to_numpy()).transform("mean")
    yd = y - y.groupby(groups.to_numpy()).transform("mean")
    keep = Xd.columns[Xd.abs().sum() > 1e-9]
    A, b = Xd[keep].to_numpy(), yd.to_numpy()
    inv = np.linalg.pinv(A.T @ A)
    beta = inv @ A.T @ b
    e = b - A @ beta
    meat = np.zeros((A.shape[1], A.shape[1]))
    for idx in pd.Series(np.arange(len(b))).groupby(groups.to_numpy()).groups.values():
        sc = A[np.asarray(idx)].T @ e[np.asarray(idx)]
        meat += np.outer(sc, sc)
    g, n, k = groups.nunique(), len(b), A.shape[1]
    cov = inv @ meat @ inv * (g / (g - 1)) * ((n - 1) / (n - k))
    return pd.Series(beta, index=keep), pd.DataFrame(cov, index=keep, columns=keep), float(1 - (e ** 2).sum() / (b ** 2).sum())


def repeat_sales_premium(s: pd.DataFrame) -> dict:
    """Same home sold twice with a newer certificate in between: change in indexed value per band gained."""
    order = {b: i for i, b in enumerate("GFEDCBA")}
    r = s.sort_values(["property_id", "sale_date"]).copy()
    g = r.groupby("property_id")
    r["prev_band"], r["prev_lv"], r["prev_lodg"] = g.energy_rating.shift(1), g.log_value_now.shift(1), g.epc_lodgement_date.shift(1)
    r = r[r.prev_band.notna() & (r.epc_lodgement_date != r.prev_lodg)]
    gained = (r.energy_rating.map(order) - r.prev_band.map(order)).to_numpy(float)
    A = np.column_stack([np.ones(len(r)), gained])
    beta, *_ = np.linalg.lstsq(A, (r.log_value_now - r.prev_lv).to_numpy(), rcond=None)
    e = (r.log_value_now - r.prev_lv).to_numpy() - A @ beta
    inv = np.linalg.inv(A.T @ A)
    se = float(np.sqrt((inv @ (A * e[:, None] ** 2).T @ A @ inv)[1, 1]))
    return {"pairs": len(r), "pct_per_band_gained": round(float(np.expm1(beta[1])) * 100, 2),
            "pct_95ci": [round(float(np.expm1(beta[1] - 1.96 * se)) * 100, 2), round(float(np.expm1(beta[1] + 1.96 * se)) * 100, 2)],
            "note": "upper bound: homes re-rated between sales have often been renovated as well"}


def ols_hc1(X: pd.DataFrame, y: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    """OLS without intercept (the LA x type cells span it) and HC1 robust covariance."""
    A = X.to_numpy()
    xtx_inv = np.linalg.pinv(A.T @ A)
    beta = xtx_inv @ A.T @ y
    e = y - A @ beta
    n, k = A.shape
    meat = (A * e[:, None] ** 2).T @ A
    cov = xtx_inv @ meat @ xtx_inv * n / (n - k)
    return pd.Series(beta, index=X.columns), pd.DataFrame(cov, index=X.columns, columns=X.columns)


def band_premia(beta: pd.Series, cov: pd.DataFrame) -> dict:
    out = {"D": {"log_premium": 0.0, "pct": 0.0, "pct_95ci": [0.0, 0.0]}}
    for b in HEDONIC_BANDS:
        if b == "D":
            continue
        c, se = beta[f"epc={b}"], float(np.sqrt(cov.loc[f"epc={b}", f"epc={b}"]))
        out[b] = {"log_premium": round(float(c), 5), "se": round(se, 5), "pct": round(float(np.expm1(c)) * 100, 2),
                  "pct_95ci": [round(float(np.expm1(c - 1.96 * se)) * 100, 2), round(float(np.expm1(c + 1.96 * se)) * 100, 2)]}
    return {b: out[b] for b in HEDONIC_BANDS}


# SAP after retrofit ---------------------------------------------------------

def sap_design(df: pd.DataFrame) -> pd.DataFrame:
    x = pd.DataFrame(index=df.index)
    for a in LAS:
        x[f"la={a}"] = (df.local_authority == a).astype(float)
    for t in TYPES:
        if t != "Semi detached":
            x[f"type={t}"] = (df.property_type == t).astype(float)
    for a in ("1930-1972", "1973-1999", "2000+"):
        x[f"age={a}"] = (df.age_band == a).astype(float)
    x["log_area"] = np.log(pd.to_numeric(df.floor_area_m2, errors="coerce").clip(15, 600)).fillna(np.log(80))
    mm = pd.to_numeric(df.loft_insulation_mm, errors="coerce")
    own = df.has_own_roof.fillna(True).astype(bool)
    for lo, hi, name in ((0, 50, "loft<50"), (50, 150, "loft50-149"), (150, 270, "loft150-269")):
        x[name] = (own & (mm >= lo) & (mm < hi)).astype(float)       # base: >= 270 mm or no own roof
    x["loft=unknown"] = (own & mm.isna()).astype(float)
    wall = df.wall_type.astype(str) + "|" + df.wall_insulation.fillna("unknown").astype(str)
    for w in ("cavity|none", "cavity|partial", "solid|none", "solid|partial", "solid|full"):
        x[f"wall={w}"] = (wall == w).astype(float)                    # base: cavity|full
    for g in ("single", "partial", "double_old", "triple"):
        x[f"glazing={g}"] = (df.glazing == g).astype(float)           # base: double_new
    for h in ("gas_non_condensing", "oil", "lpg", "heat_pump", "direct_electric", "storage_heaters", "district"):
        x[f"heating={h}"] = (df.heating == h).astype(float)           # base: gas_condensing
    for c in ("none", "smart_thermostat", "smart_trvs"):
        x[f"controls={c}"] = (df.smart_controls == c).astype(float)   # base: programmer
    x["solar"] = (pd.to_numeric(df.solar_kwp, errors="coerce").fillna(0) > 0).astype(float)
    return x


def fit_sap_model() -> dict:
    props = pd.read_parquet(REAL / "properties_share.parquet")
    split = pd.read_parquet(REAL / "splits.parquet")
    p = props.merge(split, on="customer_id")
    p = p[(p.split == "train") & p.sap_score.notna() & p.heating.ne("unknown")]
    X = sap_design(p)
    A = X.to_numpy()
    coef, *_ = np.linalg.lstsq(A, p.sap_score.to_numpy(), rcond=None)
    pred = A @ coef
    r2 = 1 - ((p.sap_score - pred) ** 2).sum() / ((p.sap_score - p.sap_score.mean()) ** 2).sum()
    return {"coef": dict(zip(X.columns, np.round(coef, 4))), "n": len(p), "r2": round(float(r2), 4)}


# ---------------------------------------------------------------- evaluation helpers

def vmetrics(y, p) -> dict:
    a, b = np.exp(y), np.exp(p)
    ape = np.abs(b - a) / a
    return {"mae_gbp": round(float(np.abs(b - a).mean()), -2), "median_ape_pct": round(float(np.median(ape)) * 100, 2),
            "within_10pct": round(float((ape <= 0.1).mean()), 4), "within_20pct": round(float((ape <= 0.2).mean()), 4),
            "r2_log": round(float(1 - ((y - p) ** 2).sum() / ((y - y.mean()) ** 2).sum()), 4)}


def baselines(tr: pd.DataFrame, ev: pd.DataFrame, premia: dict | None = None) -> dict[str, np.ndarray]:
    """No-model references: LA median x fixed rating premium (the spec's baseline) and LA x type median."""
    la = tr.groupby("local_authority").log_value_now.median()
    fixed = {"A/B": 0.06, "C": 0.03, "D": 0.0, "E": -0.03, "F/G": -0.06}   # interim ~3%/band (value_uplift.yaml)
    adj = ev.energy_rating.map(BAND_GROUP).map(fixed).fillna(0).to_numpy()
    cell = tr.groupby(["local_authority", "property_type"]).log_value_now.median()
    key = pd.MultiIndex.from_frame(ev[["local_authority", "property_type"]])
    lt = pd.Series(cell.reindex(key).to_numpy()).fillna(pd.Series(ev.local_authority.map(la).to_numpy())).to_numpy()
    return {"la_median_x_fixed_premium": ev.local_authority.map(la).to_numpy() + adj, "la_type_median": lt}


def cqr(xtr, ytr, xva, yva) -> tuple[list, float, float]:
    models = [fit_level(xtr, ytr, {"objective": "quantile", "alpha": a}) for a in BAND]
    lo, hi = (m.predict(xva) for m in models)
    scores = np.maximum(lo - yva, yva - hi)
    n = len(yva)
    q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * (BAND[1] - BAND[0])) / n)))
    return models, q, float(np.mean((yva >= lo - q) & (yva <= hi + q)))


# ---------------------------------------------------------------- train + validate

def train(path: Path = ARTIFACTS) -> dict:
    s = training_sales()
    x, y = features(s), s.log_value_now.to_numpy()
    tr, va = (s.split == "train").to_numpy(), (s.split == "val").to_numpy()
    out = {"valuation_month": s.attrs["valuation_month"], "n_sales": {k: int(v) for k, v in s.split.value_counts().items()},
           "cv": {}, "val": {}}

    # 5-fold CV on train: level GBDT vs hedonic vs baselines (grouped by property via the customer split already)
    folds: dict[str, list] = {}
    str_ = s[tr].reset_index(drop=True)
    xtr_all, ytr_all = x[tr].reset_index(drop=True), y[tr]
    groups = str_.property_id.astype("category").cat.codes.to_numpy()
    uniq = np.unique(groups)
    for k, (gtr, gev) in enumerate(KFold(5, shuffle=True, random_state=42).split(uniq)):
        itr, iev = np.isin(groups, uniq[gtr]), np.isin(groups, uniq[gev])
        m = fit_level(xtr_all[itr], ytr_all[itr])
        Xh = hedonic_design(str_)
        b, _ = ols_hc1(Xh[itr].fillna(0), ytr_all[itr])
        preds = {"gbdt_level": m.predict(xtr_all[iev]), "hedonic": Xh[iev].fillna(0).to_numpy() @ b.to_numpy(),
                 **baselines(str_[itr], str_[iev])}
        for name, p in preds.items():
            folds.setdefault(name, []).append(vmetrics(ytr_all[iev], p))
    for name, f in folds.items():
        out["cv"][name] = {k: round(float(np.mean([d[k] for d in f])), 4) for k in f[0]}
        out["cv"][name]["mape_sd"] = round(float(np.std([d["median_ape_pct"] for d in f])), 2)

    # Final fits on train; val for the bands, the blend weight and the report.
    level = fit_level(x[tr], y[tr])
    qmodels, q, cov = cqr(x[tr], y[tr], x[va], y[va])
    Xh = hedonic_design(s).fillna(0)
    beta, covm = ols_hc1(Xh[tr], y[tr])        # no location effects: level comparison only
    sec = tr & s.postcode_sector.notna().to_numpy()
    beta_fe, cov_fe, r2_within = fe_ols(Xh[sec], s.log_value_now[sec], s.postcode_sector[sec])
    premia = band_premia(beta_fe, cov_fe)
    out["hedonic_no_location_premia_vs_D"] = band_premia(beta, covm)
    out["hedonic_sector_fe"] = {"sales": int(sec.sum()), "sectors": int(s.postcode_sector[sec].nunique()),
                                "within_sector_r2": round(r2_within, 4)}
    out["repeat_sales_premium_train"] = repeat_sales_premium(s[tr])
    sv = s[va]
    p_level, p_hed = level.predict(x[va]), Xh[va].to_numpy() @ beta.to_numpy()
    out["val"] = {"gbdt_level": vmetrics(y[va], p_level), "hedonic": vmetrics(y[va], p_hed),
                  **{k: vmetrics(y[va], v) for k, v in baselines(s[tr], sv).items()}}
    lo, hi = (m.predict(x[va]) for m in qmodels)
    out["val_band"] = {"coverage": round(cov, 4), "conformal_q": round(q, 5),
                       "median_width_pct": round(float(np.median(np.exp(hi + q) - np.exp(lo - q)) / np.median(np.exp(p_level))) * 100, 1)}

    # This home: indexed previous sale, and a blend with the level model, weight fitted on val.
    hp = sv.prior_log_value_now.notna().to_numpy()
    prior = sv.prior_log_value_now.to_numpy()[hp]
    grid = np.round(np.linspace(0, 1, 21), 2)
    maes = [np.abs(np.exp(w * prior + (1 - w) * p_level[hp]) - np.exp(y[va][hp])).mean() for w in grid]
    w = float(grid[int(np.argmin(maes))])
    out["val_repeat_sales"] = {"n": int(hp.sum()), "indexed_last_sale": vmetrics(y[va][hp], prior),
                               "gbdt_level": vmetrics(y[va][hp], p_level[hp]),
                               "blend": vmetrics(y[va][hp], w * prior + (1 - w) * p_level[hp]), "blend_weight_last_sale": w}

    sap = fit_sap_model()
    out["hedonic_band_premia_vs_D"] = premia
    out["sap_model"] = {"n": sap["n"], "r2": sap["r2"]}

    path.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, m in (("level", level), ("q10", qmodels[0]), ("q90", qmodels[1])):
        files[name] = f"value_{name}.txt"
        m.booster_.save_model(str(path / files[name]), num_iteration=m.booster_.best_iteration)
    meta = {"version": "value_v1", "valuation_month": out["valuation_month"], "features": FEATURES, "files": files,
            "conformal_q": round(q, 5), "val_coverage": round(cov, 4), "blend_weight_last_sale": w,
            "hedonic": {"premia": premia, "model": "log-linear, postcode-sector fixed effects, sector-clustered SEs",
                        "cov_epc": cov_fe.loc[[c for c in cov_fe.index if c.startswith("epc=")],
                                              [c for c in cov_fe.columns if c.startswith("epc=")]].round(10).to_dict()},
            "sap_coef": sap["coef"], "trained_on": "real matched resales, train-split customers"}
    (path / META).write_text(json.dumps(meta, indent=2))
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "value_v1.json").write_text(json.dumps(out, indent=2))
    _load.cache_clear()
    return out


# ---------------------------------------------------------------- deployment

@lru_cache(maxsize=1)
def _load(path: Path = ARTIFACTS) -> dict:
    if not (path / META).exists():
        raise ModelUnavailable(f"No trained value model in {path} (make train-value, or make models SRC=...).")
    meta = json.loads((path / META).read_text())
    meta["boosters"] = {k: lgb.Booster(model_file=str(path / f)) for k, f in meta["files"].items()}
    return meta


def sap_after(p: dict, p_after: dict) -> float | None:
    """SAP score after a retrofit: current SAP + the SAP model's change between the two states, in 1-100."""
    if p.get("sap_score") is None:
        return None
    coef = pd.Series(_load()["sap_coef"])
    X = sap_design(pd.DataFrame([p, p_after]))[coef.index]
    delta = float((X.iloc[1] - X.iloc[0]) @ coef)
    return float(np.clip(p["sap_score"] + delta, 1, 100))


def sap_band(sap: float) -> str:
    for cut, b in ((92, "A"), (81, "B"), (69, "C"), (55, "D"), (39, "E"), (21, "F")):
        if sap >= cut:
            return b
    return "G"


def predict_range(p: dict) -> dict | None:
    """Indicative current value (p10 / p50 / p90, GBP at the valuation month) from the description, or None."""
    meta = _load()
    x = features(pd.DataFrame([p]))
    if x[["local_authority", "property_type", "floor_area_m2"]].isna().any(axis=None):
        return None
    b, q = meta["boosters"], meta["conformal_q"]
    p50, lo, hi = (float(b[k].predict(x)[0]) for k in ("level", "q10", "q90"))
    return {"p10": round(float(np.exp(lo - q)), -3), "p50": round(float(np.exp(p50)), -3),
            "p90": round(float(np.exp(hi + q)), -3), "valuation_month": meta["valuation_month"]}


def value_estimate(p: dict, p_after: dict, retrofits: list[str]) -> dict:
    """PropertyReport.value: baseline value, its range, and the indicative before / after rating-band uplift."""
    meta = _load()
    level = predict_range(p)
    w = meta["blend_weight_last_sale"]
    if p.get("current_value_estimate") is not None and level is not None:
        # Blend fitted on val: beats the indexed last sale alone on held-out repeat sales (eval/eval_value.py)
        baseline = float(np.exp(w * np.log(p["current_value_estimate"]) + (1 - w) * np.log(level["p50"])))
        method = f"last sale indexed with the official house price index ({w:.0%}) blended with the value model"
    elif p.get("current_value_estimate") is not None:
        baseline, method = float(p["current_value_estimate"]), "last sale indexed with the official house price index"
    elif level is not None:
        baseline, method = level["p50"], "value model (similar homes' sale prices, real matched sales)"
    else:
        baseline, method = None, None
    sap0, sap1 = p.get("sap_score"), sap_after(p, p_after) if retrofits else p.get("sap_score")
    band0 = p.get("energy_rating") or (sap_band(sap0) if sap0 is not None else None)
    band1 = sap_band(sap1) if sap1 is not None and retrofits else band0
    prem = meta["hedonic"]["premia"]
    g0, g1 = BAND_GROUP.get(band0), BAND_GROUP.get(band1)
    out = {"status": "estimated", "baseline": None if baseline is None else round(baseline, -3),
           "baseline_method": method, "level_range": level, "rating_before": band0, "rating_after": band1,
           "sap_before": sap0, "sap_after": None if sap1 is None else round(sap1, 1),
           "method": "Hedonic rating-band premium (log-linear model on real matched sales), same home, same date",
           "caveat": CAVEAT}
    if g0 is None or g1 is None:
        return {**out, "status": "unavailable", "low": None, "high": None, "pct_low": None, "pct_high": None,
                "caveat": "No energy rating, so the rating-related value change can't be estimated. " + CAVEAT}
    d = prem[g1]["log_premium"] - prem[g0]["log_premium"]
    cov = meta["hedonic"]["cov_epc"]
    var = lambda a, b: cov.get(f"epc={a}", {}).get(f"epc={b}", 0.0)
    se = float(np.sqrt(max(var(g1, g1) + var(g0, g0) - 2 * var(g1, g0), 0.0))) if g0 != g1 else 0.0
    pct = sorted((float(np.expm1(d - 1.96 * se)) * 100, float(np.expm1(d + 1.96 * se)) * 100))
    out.update({"pct_low": round(pct[0], 1), "pct_high": round(pct[1], 1), "band_change": f"{band0} -> {band1}"})
    if baseline is None:
        return {**out, "status": "unavailable", "low": None, "high": None,
                "caveat": "No current valuation for this property, so the value impact can't be estimated. " + CAVEAT}
    return {**out, "low": round(baseline * (1 + pct[0] / 100), -2), "high": round(baseline * (1 + pct[1] / 100), -2)}


def main() -> None:
    out = train()
    print(json.dumps({k: out[k] for k in ("n_sales", "val", "val_band", "val_repeat_sales", "hedonic_band_premia_vs_D",
                                          "hedonic_no_location_premia_vs_D", "hedonic_sector_fe",
                                          "repeat_sales_premium_train", "sap_model")}, indent=1))


if __name__ == "__main__":
    main()
