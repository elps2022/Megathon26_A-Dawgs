"""Savings-simulator checks against real installs and the catalogue (ARCHITECTURE 3.5 savings validity, 3.6).

    uv run python eval/eval_savings.py      -> eval/results/savings_checks.json

1. Solar: the simulator's rule (kWp x yield x self-consumption, capped at a share of import; config/simulator.yaml)
   vs the observed change in metered electricity for real NEED PV installs (data/need_labels.py), every split.
   NEED doesn't record kWp or location within the region, so the default kWp and yield are used.
2. Catalogue sanity bounds: on real homes, the share of model-based fabric savings (% of heating demand) that
   fall outside the catalogue's typical % range. Those ranges are sanity bounds, so outside != wrong;
   they are flagged, as ARCHITECTURE 3.5 asks.
3. V1 vs V2 on real installs (ARCHITECTURE 3.5 "both scored"): the shipped, frozen V2 GBDT's change when NEED's
   cavity_installed flag goes 0 -> 1 on the homes that really had cavity walls filled, next to V1 and the observed
   change, every split. V2 has no loft feature, so it predicts no change for loft installs (the engine then uses
   physics^beta). Post-test reporting on frozen models: nothing is fitted or tuned.
4. Synthetic mid-series retrofits (overview step 1): every retrofit event in data/generated (all seven measures,
   including heat pumps, controls and glazing, which real data can't check) vs the generator's hidden true annual
   change per fuel (_hidden/retrofit_truth.csv, labels only). The engine sees the home's description and its
   bills from before the install only, annualised with seasonal weights, and simulates that one retrofit with V1
   and with V2.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.engine.feasibility import RETROFIT_CATALOGUE  # noqa: E402
from greenhome.engine.report import build_report  # noqa: E402
from greenhome.data.annualise import annualise_series  # noqa: E402
from greenhome.engine.carbon import calculate_carbon_kg  # noqa: E402
from greenhome.engine.portfolio import _clean  # noqa: E402
from greenhome.engine.savings import FABRIC, EnergyUnavailable, _cfg, simulate_package_savings  # noqa: E402
from greenhome.models.energy import GreyboxModel, MLModel  # noqa: E402
from greenhome.models.energy_greybox import physics_retrofit_ratios  # noqa: E402
from greenhome.models.need_data import features, load_need  # noqa: E402
from greenhome.schema import Property  # noqa: E402

RESULTS = ROOT / "eval" / "results"
LABELS = ROOT / "data" / "processed" / "need_retrofit_labels.parquet"
REAL = ROOT / "data" / "processed" / "real" / "properties_share.parquet"
GEN = ROOT / "data" / "generated"
TO_ENGINE = {"glazing_upgrade": "double_glazing"}


def solar_check() -> dict:
    sol = _cfg("simulator.yaml")["solar"]
    lab = pd.read_parquet(LABELS)
    pv = lab[lab.measure == "solar_pv"]
    reduction = sol["default_kwp"] * sol["yield_kwh_per_kwp"]["default"] * sol["self_consumption"]
    predicted = -np.minimum(reduction, sol["max_share_of_import"] * pv.pre_kwh) / pv.pre_kwh
    out = {"fitted_on": "train (self_consumption, config/simulator.yaml); val and test are checks",
           "rule": f"{sol['default_kwp']} kWp x {sol['yield_kwh_per_kwp']['default']} kWh/kWp x "
                   f"{sol['self_consumption']} self-consumption, capped at {sol['max_share_of_import']:.0%} of import"}
    for split, g in pv.groupby("split"):
        p = predicted[g.index]
        out[split] = {"n": len(g), "observed_median_change": round(float(g.did_change.median()), 4),
                      "observed_median_did_kwh": round(float((g.did_change * g.pre_kwh).median()), 1),
                      "rule_median_change": round(float(p.median()), 4),
                      "observed_median_kwh": round(float((g.post_kwh - g.pre_kwh).median()), 1),
                      "rule_median_kwh": round(float((p * g.pre_kwh).median()), 1)}
    return out


def catalogue_check(n: int = 1000) -> dict:
    props = pd.read_parquet(REAL).sample(n, random_state=7)
    fields = Property.model_fields
    pct: dict[str, list[float]] = {r: [] for r in FABRIC}
    for _, row in props.iterrows():
        d = {k: (None if (isinstance(v, float) and v != v) or v is pd.NA else v) for k, v in row.items() if k in fields}
        r = build_report(Property(**d))
        if not r.energy_before or r.energy_before.heat_demand_kwh <= 0:
            continue
        heat = r.energy_before.heat_demand_kwh
        for rec in r.recommendations:
            if rec.retrofit in pct and "energy model (trained" in rec.reason:
                pct[rec.retrofit].append(100 * rec.marginal_saving.total_kwh / heat)
    out = {}
    for name, v in pct.items():
        if not v:
            continue
        v = np.array(v)
        lo, hi = RETROFIT_CATALOGUE[name]["saving_pct_heating"]
        out[name] = {"n": int(len(v)), "catalogue_range_pct": [lo, hi], "median_pct_of_heating": round(float(np.median(v)), 1),
                     "below_range": round(float((v < lo).mean()), 3), "above_range": round(float((v > hi).mean()), 3)}
    return out


def v1_v2_installs() -> dict:
    """Observed vs V1 vs V2 median change in gas for the real NEED cavity and loft installs, per split."""
    beta = GreyboxModel().meta["gas"]["physics_elasticity"]
    v2 = MLModel()
    booster, meta = v2.boosters["gas"], v2.meta["gas"]
    lab = pd.read_parquet(LABELS)
    f = features(load_need()).fillna({"imd_quintile": v2.meta.get("imd_fill", 3.0)})
    out = {"frozen_models": "shipped V1 / V2 artifacts; post-test reporting, nothing tuned"}
    for split in ("train", "val", "test"):
        for measure, d in physics_retrofit_ratios(split).items():
            r = {"n": len(d), "observed_median_change": round(float(np.median(d.observed_ratio) - 1), 4),
                 "v1_median_change": round(float(np.median(d.physics_ratio ** beta) - 1), 4)}
            if measure == "cavity_wall_insulation":
                rows = lab[(lab.split == split) & (lab.measure == measure)].need_row.to_numpy()
                x = f.loc[rows]
                p0, p1 = (booster.predict(x.assign(cavity_installed=c)[meta["features"]]) for c in (0, 1))
                ratio = p1 / p0 if meta["raw_target"] else np.exp(p1 - p0)
                r["v2_median_change"] = round(float(np.median(ratio) - 1), 4)
            else:
                r["v2_median_change"] = None
                r["v2_note"] = "V2 has no loft feature (NEED's loft flag fails EPC parity); the engine uses physics^beta"
            out.setdefault(measure, {})[split] = r
    return out


def synthetic_events() -> dict:
    """Engine-predicted vs hidden true annual change for every synthetic mid-series retrofit, V1 and V2."""
    props = pd.read_csv(GEN / "properties.csv").set_index("property_id")
    ev = pd.read_csv(GEN / "retrofit_events.csv").merge(pd.read_csv(GEN / "_hidden" / "retrofit_truth.csv"),
                                                       on=["property_id", "retrofit"])
    monthly = pd.read_csv(GEN / "monthly_energy.csv", parse_dates=["month"])
    monthly = monthly.merge(ev[["property_id", "install_month"]], on="property_id")
    pre = monthly[monthly.month < pd.to_datetime(monthly.install_month)]
    # The home's own space-heating share from a degree-day fit on its pre-install months only.
    from greenhome.data.degree_day_fit import fit, load_hdd
    _, typical = load_hdd()
    pre_fit = fit(props.reset_index(), pre.assign(month=pre.month.dt.strftime("%Y-%m-%d"))[["property_id", "month", "elec_import_kwh", "gas_kwh"]])
    pre_fit = pre_fit[pre_fit.fit_ok].merge(props.reset_index()[["property_id", "local_authority"]], on="property_id")
    heat = pre_fit.slope_kwh_per_hdd * pre_fit.local_authority.map(typical)
    own_share = dict(zip(pre_fit.property_id, heat / (heat + 12 * pre_fit.base_kwh_per_month)))
    bills = {}
    for pid, g in pre.groupby("property_id"):
        e, _ = annualise_series(list(g.month.dt.month), list(g.elec_import_kwh), "elec")
        gas, _ = annualise_series(list(g.month.dt.month), list(g.gas_kwh), "gas")
        bills[pid] = (e, gas)
    rows, skipped = [], 0
    for _, e in ev.iterrows():
        d = _clean(props.loc[e.property_id].to_dict() | {"property_id": e.property_id})
        d["annual_elec_kwh"], d["annual_gas_kwh"] = bills.get(e.property_id, (None, None))
        # Only pre-install information: the 2-year history fields span the install (fewer than 24 pre-install
        # months), so they are blanked and the 1-year model runs on the pre-install annualised bill.
        d |= {f"{h}_{f}_kwh": None for h in ("recent", "prior") for f in ("elec", "gas")}
        d["heating_share"] = own_share.get(e.property_id)
        p = Property(**d)
        name = TO_ENGINE.get(e.retrofit, e.retrofit)
        rec = {"measure": name, "true_elec": -e.true_elec_change_kwh_yr, "true_gas": -e.true_gas_change_kwh_yr}
        try:
            runs = (("greybox", "greybox", p), ("ml", "ml", p),
                    ("greybox_default_share", "greybox", p.model_copy(update={"heating_share": None})))
            for col, model, prop in runs:
                s = simulate_package_savings(prop, [name], model)[3]
                rec[f"{col}_elec"], rec[f"{col}_gas"] = s.elec_kwh, s.gas_kwh
                rec[f"{col}_in_band"] = s.low <= rec["true_elec"] + rec["true_gas"] <= s.high
            rec["own_share"] = p.heating_share is not None
        except EnergyUnavailable:
            skipped += 1
            continue
        rows.append(rec)
    df = pd.DataFrame(rows)
    for who in ("true", "greybox", "ml", "greybox_default_share"):
        df[f"{who}_total"] = df[f"{who}_elec"] + df[f"{who}_gas"]
        df[f"{who}_carbon"] = [calculate_carbon_kg(a, b) for a, b in zip(df[f"{who}_elec"], df[f"{who}_gas"])]
    out = {"what": "saving = before - after, kWh/yr typical weather; true = generator's hidden change",
           "events": len(df), "skipped_energy_unavailable": skipped, "by_measure": {}}

    def summary(g: pd.DataFrame) -> dict:
        r = {"n": len(g)}
        for k in ("elec", "gas", "total", "carbon"):
            r[f"true_median_{k}"] = round(float(g[f"true_{k}"].median()), 1)
        r["share_with_own_heating_share"] = round(float(g.own_share.mean()), 3)
        for m, label in (("greybox", "v1"), ("ml", "v2"), ("greybox_default_share", "v1_default_share")):
            for k in ("elec", "gas", "total", "carbon"):
                r[f"{label}_median_{k}"] = round(float(g[f"{m}_{k}"].median()), 1)
            r[f"{label}_mae_total"] = round(float((g[f"{m}_total"] - g.true_total).abs().mean()), 1)
            r[f"{label}_sum_ratio_total"] = round(float(g[f"{m}_total"].sum() / g.true_total.sum()), 3) if g.true_total.sum() else None
            r[f"{label}_sign_agrees"] = round(float((np.sign(g[f"{m}_total"]) == np.sign(g.true_total)).mean()), 3)
            r[f"{label}_true_in_band"] = round(float(g[f"{m}_in_band"].mean()), 3)
        return r

    for name, g in df.groupby("measure"):
        out["by_measure"][name] = summary(g)
    out["all"] = summary(df)
    return out


def main() -> None:
    warnings.filterwarnings("ignore")
    out = {"solar_vs_need_pv_installs": solar_check(), "model_savings_vs_catalogue_ranges": catalogue_check(),
           "v1_v2_on_real_installs": v1_v2_installs(), "synthetic_mid_series_events": synthetic_events()}
    (RESULTS / "savings_checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
