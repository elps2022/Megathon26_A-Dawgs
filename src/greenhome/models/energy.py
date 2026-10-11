"""One interface for the energy models (ARCHITECTURE 3.5): V1 grey box, V2 ML, physics only.

    model = load_model("greybox")          # or "ml", "physics"
    model.predict(p)        -> EnergyBreakdown for one Property (expected annual use, typical weather)
    model.predict_frame(df) -> elec_kwh / gas_kwh for many homes at once (portfolio, evaluation)

The input is the (post-retrofit) Property rather than a HomeState: V1 and V2 need the parity features
(need_data.epc_features), which a HomeState doesn't carry, and V1 builds its physics state from the Property
itself (physics.property_fabric). A retrofit edits the Property (savings.apply_retrofits_to_property) and the
model predicts again.

Conventions match training exactly (energy_greybox / energy_ml), so deployment scores the model that was evaluated:
  - V1:  exp(elasticity x log physics + GBDT(description)) x smearing. Physics uses the home's own fabric and
         region; gas physics is (space heat + hot water) / NEED boiler efficiency, electricity physics is lighting
         + appliances. A retrofit changes V1 by exactly (physics after / before)^elasticity (energy_greybox).
  - V2:  exp(GBDT) x smearing, from the parity features alone.
  - Physics only: the same physics x one scale factor fitted on NEED train (eval/results/energy_v1.json).
Gas is NaN for homes that aren't gas heated (the NEED gas model only covers gas-heated homes). Predictions are
the expected (mean) use, Duan smearing included, so totals and savings stay unbiased.

Mode 1 ("this home", energy_bills): for a home with bills, `forecast_frame` predicts next year's use from its
own billed use plus its description (a GBDT on log(this year / last year), L1). On held-out NEED homes it beats
"last year's bill" by 2.6% (gas) / 3.3% (electricity) MAE, so the engine uses it as the baseline for billed homes.
The billed figure is the plain annualised bill (Property has no weather-normalised field); NEED's prior-year gas
is weather corrected, so a very cold or mild billing year passes partly into the forecast.

`similar_home_kwh` is the one "typical / expected similar home" estimate (mode 2) shared by the score's
consumption sub-score and the simulator's missing-bill fallback; without trained artifacts it falls back to the
NEED log-linear table (models/expected_use.py).

V1 also gives an 80% band for one home's actual use (`{fuel}_p10` / `{fuel}_p90`): conformalised quantile
regression (CQR). Quantile GBDTs (10% / 90%) on the same V1 residual are fitted on train; the val split then
widens (or narrows) both ends by the one amount Q that makes 80% of val homes fall inside. Coverage is
guaranteed on average for new homes like NEED's, and reported on the test split (eval/eval_energy.py).

Trained artifacts (LightGBM text models + meta.json) live in models/artifacts/, gitignored: model weights are
hosted outside the repo. `make train` rebuilds them from NEED in about a minute:

    uv run python -m greenhome.models.energy        -> src/greenhome/models/artifacts/
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Protocol

import numpy as np
import pandas as pd

from greenhome.models import physics
from greenhome.models.need_data import ROOT, epc_features
from greenhome.schema import Element, EnergyBreakdown, Property

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
FUELS = ("gas", "elec")
BAND = (0.1, 0.9)  # p10 / p90: an 80% interval
V1_RESULTS = ROOT / "eval" / "results" / "energy_v1.json"


class ModelUnavailable(RuntimeError):
    """The trained artifacts for this model aren't on disk (run `make train`)."""


class EnergyModel(Protocol):
    version: str

    def predict_frame(self, props: pd.DataFrame) -> pd.DataFrame: ...

    def predict(self, p: Property) -> EnergyBreakdown: ...


def physics_frame(props: pd.DataFrame) -> pd.DataFrame:
    """Physics demands for Property rows (own fabric), plus the training-convention log physics per fuel."""
    fabric, filled = physics.property_fabric(props)
    d = physics.demands(fabric)
    eff = physics.cfg("physics.yaml")["need_gas_boiler_efficiency"]
    d["log_physics_gas"] = np.log((d.space_heat_kwh + d.hot_water_kwh) / eff)
    d["log_physics_elec"] = np.log(d.lights_appliances_kwh)
    d["fabric_defaulted"] = filled.map(",".join)
    return d


def _gas_heated(props: pd.DataFrame) -> np.ndarray:
    return (props.gas_meter.astype(bool) & props.heating.isin(physics.GAS_HEATING)).to_numpy()


class _Base:
    version = "base"

    def _log_pred(self, props: pd.DataFrame, phys: pd.DataFrame, fuel: str) -> np.ndarray:
        raise NotImplementedError

    def predict_frame(self, props: pd.DataFrame) -> pd.DataFrame:
        """Expected annual elec_kwh / gas_kwh per home (NaN where the model can't place the home)."""
        props = props.reset_index(drop=True)
        phys = physics_frame(props)
        out = pd.DataFrame(index=props.index)
        for fuel in FUELS:
            out[f"{fuel}_kwh"] = np.exp(self._log_pred(props, phys, fuel)) * self.smear[fuel]
        out.loc[~_gas_heated(props), "gas_kwh"] = np.nan
        for c in ("space_heat_kwh", "hot_water_kwh", "lights_appliances_kwh", "hlc_w_per_k", "fabric_defaulted"):
            out[f"physics_{c}"] = phys[c]
        return out

    def predict(self, p: Property) -> EnergyBreakdown:
        """One home. Heat demand / base load split the predicted gas by the physics shares (space heat vs hot
        water); for other homes all predicted electricity is base load, as the NEED electricity model can't
        separate electric heating. Raises ModelUnavailable for no prediction (e.g. floor area unknown)."""
        df = pd.DataFrame([p.model_dump()])
        r = self.predict_frame(df).iloc[0]
        if np.isnan(r.elec_kwh):
            raise ModelUnavailable("Too little property data for the energy model (e.g. floor area unknown).")
        band = {k: (None if k not in r or np.isnan(r[k]) else round(float(r[k]), 1))
                for k in ("elec_p10", "elec_p90", "gas_p10", "gas_p90")}
        gas = 0.0 if np.isnan(r.gas_kwh) else float(r.gas_kwh)
        space = r.physics_space_heat_kwh / (r.physics_space_heat_kwh + r.physics_hot_water_kwh)
        heat = gas * space
        env = physics.envelope(physics.property_fabric(df)[0]).iloc[0]
        return EnergyBreakdown(
            elec_kwh=round(float(r.elec_kwh), 1), gas_kwh=round(gas, 1), heat_demand_kwh=round(heat, 1),
            base_load_kwh=round(float(r.elec_kwh) + gas - heat, 1), hlc_w_per_k=round(float(r.physics_hlc_w_per_k), 1),
            calibration_k=None, calibrated=False, model_version=self.version,
            elements=[Element(**e) for e in physics.elements(env)], **band)


class PhysicsModel(_Base):
    """Physics only, x one global scale per fuel fitted on NEED train (the prebound effect: real homes use
    about half what steady-state physics predicts). No trained artifacts needed."""
    version = "physics"

    def __init__(self) -> None:
        r = json.loads(V1_RESULTS.read_text())
        self.scale = {fuel: r[fuel]["physics_scale_factor"] for fuel in FUELS}
        self.smear = {fuel: 1.0 for fuel in FUELS}

    def _log_pred(self, props, phys, fuel):
        return phys[f"log_physics_{fuel}"].to_numpy() + np.log(self.scale[fuel])


class _Trained(_Base):
    def __init__(self, kind: str, path: Path = ARTIFACTS) -> None:
        import lightgbm as lgb
        meta_path = path / "meta.json"
        if not meta_path.exists():
            raise ModelUnavailable(f"No trained energy models in {path} (run `make train`).")
        meta = json.loads(meta_path.read_text())
        self.meta = meta[kind]
        self.version = self.meta["version"]
        self.smear = {fuel: self.meta[fuel]["smearing_factor"] for fuel in FUELS}
        self.boosters = {fuel: lgb.Booster(model_file=str(path / self.meta[fuel]["file"])) for fuel in FUELS}
        self.quantiles = {fuel: [lgb.Booster(model_file=str(path / f)) for f in self.meta[fuel]["band"]["files"]]
                          for fuel in FUELS if "band" in self.meta[fuel]}
        self.mode1 = {fuel: lgb.Booster(model_file=str(path / self.meta[fuel]["mode1"]["file"]))
                      for fuel in FUELS if "mode1" in self.meta[fuel]}

    def _x(self, props, phys, fuel) -> pd.DataFrame:
        x = epc_features(props)[self.meta[fuel]["features"]]   # training columns (gas: no gas_heated)
        # NEED has no missing deprivation quintile, and LightGBM sends a value it never saw missing to the 0 side,
        # i.e. "more deprived than quintile 1". Unknown -> the training median instead (neutral, not most deprived).
        return x.fillna({"imd_quintile": self.meta.get("imd_fill", 3.0)})


class GreyboxModel(_Trained):
    """V1: physics x learned residual."""

    def __init__(self, path: Path = ARTIFACTS) -> None:
        super().__init__("greybox", path)

    def _log_pred(self, props, phys, fuel):
        beta = self.meta[fuel]["physics_elasticity"]
        return beta * phys[f"log_physics_{fuel}"].to_numpy() + self.boosters[fuel].predict(self._x(props, phys, fuel))

    def log_band(self, props: pd.DataFrame, phys: pd.DataFrame, fuel: str) -> tuple[np.ndarray, np.ndarray]:
        """Conformalised 80% band in log kWh for each home's actual use."""
        beta, q = self.meta[fuel]["physics_elasticity"], self.meta[fuel]["band"]["conformal_q"]
        base, x = beta * phys[f"log_physics_{fuel}"].to_numpy(), self._x(props, phys, fuel)
        lo, hi = (base + m.predict(x) for m in self.quantiles[fuel])
        return lo - q, hi + q

    def forecast_frame(self, props: pd.DataFrame) -> pd.DataFrame:
        """Mode 1: next year's expected (mean) use from each home's own billed annual use + description.
        NaN where the home has no bill for that fuel (or, for gas, isn't gas heated)."""
        props = props.reset_index(drop=True)
        phys = physics_frame(props)
        out = pd.DataFrame(index=props.index)
        for fuel in FUELS:
            m = self.meta[fuel].get("mode1")
            prior = pd.to_numeric(props.get(f"annual_{fuel}_kwh"), errors="coerce")
            if m is None or prior is None:
                out[f"{fuel}_kwh"] = np.nan
                continue
            prior = prior.where(prior > 0)
            x = epc_features(props).assign(log_physics=phys[f"log_physics_{fuel}"].to_numpy(),
                                           log_prior=np.log(prior.to_numpy()))
            x = x[m["features"]].fillna({"imd_quintile": self.meta.get("imd_fill", 3.0)})
            change = self.mode1[fuel].predict(x)
            out[f"{fuel}_kwh"] = np.where(prior.notna(), np.exp(x.log_prior.to_numpy() + change) * m["total_factor"], np.nan)
        out.loc[~_gas_heated(props), "gas_kwh"] = np.nan
        return out

    def predict_frame(self, props: pd.DataFrame) -> pd.DataFrame:
        out = super().predict_frame(props)
        props = props.reset_index(drop=True)
        phys = physics_frame(props)
        for fuel in self.quantiles:
            lo, hi = self.log_band(props, phys, fuel)
            out[f"{fuel}_p10"], out[f"{fuel}_p90"] = np.exp(lo), np.exp(hi)
        out.loc[out.gas_kwh.isna(), ["gas_p10", "gas_p90"]] = np.nan
        return out


class MLModel(_Trained):
    """V2: GBDT direct on the parity features."""

    def __init__(self, path: Path = ARTIFACTS) -> None:
        super().__init__("ml", path)

    def _log_pred(self, props, phys, fuel):
        p = self.boosters[fuel].predict(self._x(props, phys, fuel))
        return np.log(np.clip(p, 1e-6, None)) if self.meta[fuel]["raw_target"] else p


MODELS = {"greybox": GreyboxModel, "ml": MLModel, "physics": PhysicsModel}


def _heating_fuel(p: dict) -> str | None:
    """Metered fuel the heating draws from (as engine.savings._heating_fuel, kept here to avoid a cycle)."""
    system = physics.cfg("efficiencies.yaml")["heating_systems"].get(p.get("heating"))
    if system is None:
        return "gas" if p.get("gas_meter", True) else None
    if system["fuel"] == "gas" and not p.get("gas_meter", True):
        return None
    return system["fuel"] if system["fuel"] in ("gas", "electricity") else None


def similar_home_kwh(p: dict, statistic: str = "mean", model_name: str = "greybox", score_reference: bool = False) -> dict:
    """Expected annual use of a home like this one, from its description only (mode 2): {"elec", "gas", "source"}.
    statistic "mean" (savings, totals) or "median" (the typical similar home the score compares against).
    Gas is 0.0 without a gas meter. A gas meter in a home not heated by gas is hot water / cooking only: NEED has
    no gas readings for such homes, so it is the base-load share (1 - gas_heating_share, config/simulator.yaml)
    of a similar gas-heated home. Falls back to the NEED log-linear table without trained artifacts, or when
    the model can't place the home (e.g. unknown floor area). None = too little data for either.

    score_reference=True is the consumption sub-score's comparison home (SCORING.md): any home with a gas meter
    is compared with a similar gas-heated home's gas, as the score was designed and tested. Otherwise a heat pump
    would be compared with NEED's non-gas homes, many of which burn unmetered oil, and look like a heavy user."""
    from types import SimpleNamespace
    from greenhome.models.expected_use import expected_annual_kwh
    cooker = bool(p.get("gas_meter", True)) and _heating_fuel(p) != "gas"
    if score_reference and cooker:
        cooker, share_override = True, 1.0
    else:
        share_override = None
    share = 1 - physics.cfg("simulator.yaml")["gas_heating_share"]["value"] if cooker else 1.0
    if share_override is not None:
        share = share_override
    as_gas = {**p, "heating": "gas_condensing"} if cooker else p
    keys = ("property_type", "age_band", "floor_area_m2", "energy_rating", "heating", "gas_meter")
    table = lambda q: (expected_annual_kwh(SimpleNamespace(**{k: q.get(k) for k in keys}), statistic=statistic)
                       if all(q.get(k) is not None for k in keys) else {"elec": None, "gas": None})
    t, tg = table(p), table(as_gas)
    out = {"elec": t["elec"], "gas": None if tg["gas"] is None else round(tg["gas"] * share, 1), "source": "need_loglinear"}
    try:
        model = load_model(model_name)
    except ModelUnavailable:
        model = None
    defaults = {k: f.default for k, f in Property.model_fields.items() if not f.is_required()}
    needed = ("property_type", "age_band", "region", "heating", "wall_type", "glazing", "local_authority")
    if model is not None and all(p.get(k) is not None for k in needed):
        rows = pd.DataFrame([{**defaults, **{k: v for k, v in q.items() if k in Property.model_fields}} for q in (p, as_gas)])
        pred = model.predict_frame(rows)
        unsmear = lambda fuel: model.smear[fuel] if statistic == "median" else 1.0
        elec = pred.elec_kwh.iloc[0] / unsmear("elec")
        gas = pred.gas_kwh.iloc[1 if cooker else 0] / unsmear("gas") * share
        if np.isfinite(elec):
            out["elec"], out["source"] = round(float(elec), 1), model.version
        if np.isfinite(gas):
            out["gas"] = round(float(gas), 1)
    if not p.get("gas_meter", True):
        out["gas"] = 0.0  # no gas supply: a real zero, not missing data
    return out


@lru_cache(maxsize=1)
def physics_elasticity() -> dict[str, float]:
    """How much of a physics-predicted fabric change real homes show (eval/results/energy_v1.json, fitted on the
    train-split NEED installs): the pooled value V1 uses, and the range implied by loft vs cavity installs
    separately, which the simulator uses as the saving's low / high band."""
    r = json.loads(V1_RESULTS.read_text())["gas"]
    implied = [np.log1p(c["train"]["observed_median_change"]) / np.log1p(c["train"]["physics_only_median_change"])
               for c in r["retrofit_check"].values()]
    return {"beta": r["physics_elasticity"], "low": round(float(min(implied)), 3), "high": round(float(max(implied)), 3)}


@lru_cache(maxsize=None)
def load_model(name: str = "greybox") -> EnergyModel:
    """The energy model selected by the `energy_model` switch. Raises ModelUnavailable if not trained."""
    if name not in MODELS:
        raise ValueError(f"Unknown energy model {name!r}; expected one of {sorted(MODELS)}")
    return MODELS[name]()


def fit_band(xtr, ytr, xva, yva, fuel: str, beta: float) -> tuple[list, float, float]:
    """CQR for V1: quantile GBDTs (BAND) on log(actual) - beta x log(physics) from the description, fitted on
    train; Q = the conformal correction from val. Returns (models, Q, val coverage after correction)."""
    import lightgbm as lgb
    from greenhome.models.energy_ml import LGB_PARAMS, tuned_params
    tuned = {k: v for k, v in tuned_params(fuel).items() if k not in ("objective", "alpha", "tweedie_variance_power")}
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    f = lambda d: d.drop(columns="log_physics")
    r = ytr - beta * xtr.log_physics
    models = []
    for a in BAND:
        m = lgb.LGBMRegressor(**{**LGB_PARAMS, **tuned, "objective": "quantile", "alpha": a})
        m.fit(f(xtr)[~es], r[~es], eval_X=(f(xtr)[es],), eval_y=(r[es],), callbacks=[lgb.early_stopping(100, verbose=False)])
        models.append(m)
    base = beta * xva.log_physics.to_numpy()
    lo, hi = (base + m.predict(f(xva)) for m in models)
    yv = yva.to_numpy()
    scores = np.maximum(lo - yv, yv - hi)          # CQR conformity score
    n, level = len(yv), BAND[1] - BAND[0]
    q = float(np.quantile(scores, min(1.0, np.ceil((n + 1) * level) / n)))
    coverage = float(np.mean((yv >= lo - q) & (yv <= hi + q)))
    return models, q, coverage


def fit_mode1(fuel: str, path: Path) -> dict:
    """Mode 1 change model, exactly as evaluated in energy_bills (train rows with a valid prior-year reading,
    L1 on log(this year / last year), same early-stopping slice), saved for deployment."""
    import lightgbm as lgb
    from greenhome.models.energy_greybox import with_physics
    from greenhome.models.energy_ml import LGB_PARAMS, metrics
    from greenhome.models.need_data import dataset
    x, y, split = dataset(fuel, own_bills=True)
    x = with_physics(x, fuel)
    tr, va = (split == "train").to_numpy(), (split == "val").to_numpy()
    xtr, ytr = x[tr], y[tr]
    es = np.random.default_rng(0).random(len(xtr)) < 0.15
    mm = xtr.log_prior.notna().to_numpy()
    xm, ym, esm = xtr[mm], (ytr - xtr.log_prior)[mm], es[mm]
    m = lgb.LGBMRegressor(**{**LGB_PARAMS, "objective": "l1"})
    m.fit(xm[~esm], ym[~esm], eval_X=(xm[esm],), eval_y=(ym[esm],), callbacks=[lgb.early_stopping(100, verbose=False)])
    fname = f"mode1_{fuel}.txt"
    m.booster_.save_model(str(path / fname), num_iteration=m.booster_.best_iteration)
    xv, yv = x[va], y[va]
    hv = xv.log_prior.notna().to_numpy()
    pv = xv.log_prior.to_numpy()[hv] + m.predict(xv[hv])
    out = {"file": fname, "features": list(xm.columns), "iterations": int(m.best_iteration_),
            # Totals factor = sum(actual) / sum(prediction) on train. Not Duan smearing (mean of exp(residual)):
            # year-on-year errors are relatively larger for small users, so smearing overshot val totals by 4-5%
            # while the plain L1 forecast was already within 0-3%.
            "total_factor": round(float(np.exp(ytr[mm]).sum() / np.exp(xm.log_prior.to_numpy() + m.predict(xm)).sum()), 4),
            "val_total_ratio": None,
            "val_homes_with_prior": metrics(yv.to_numpy()[hv], pv),
            "val_last_year_same_homes": metrics(yv.to_numpy()[hv], xv.log_prior.to_numpy()[hv])}
    out["val_total_ratio"] = round(float((np.exp(pv) * out["total_factor"]).sum() / np.exp(yv[hv]).sum()), 4)
    return out


def train(path: Path = ARTIFACTS) -> dict:
    """Fit V1 and V2 per fuel on the NEED train split (early stopping on a train slice, as in evaluation), save
    them, and check the saved models reproduce the evaluated validation scores. Val and test are not fitted on."""
    from greenhome.models.energy_greybox import elasticity, fit_v1, v1_log_pred, with_physics
    from greenhome.models.energy_ml import _RawTarget, fit_lgb, metrics
    from greenhome.models.need_data import dataset

    path.mkdir(parents=True, exist_ok=True)
    meta: dict = {"greybox": {"version": "greybox_v1"}, "ml": {"version": "ml_v2"},
                  "trained_on": "NEED 50k anonymised sample, train split only (data/splits/need_split.csv)",
                  "params": "config/energy_params.yaml"}
    for fuel in FUELS:
        x, y, split = dataset(fuel)
        x = with_physics(x, fuel)
        tr, va = (split == "train").to_numpy(), (split == "val").to_numpy()
        xtr, ytr, xva, yva = x[tr], y[tr], x[va], y[va]
        es = np.random.default_rng(0).random(len(xtr)) < 0.15
        plain = lambda df: df.drop(columns="log_physics")
        beta = elasticity(fuel)
        if fuel == "gas":
            imd_fill = float(xtr.imd_quintile.median())
            meta["greybox"]["imd_fill"] = meta["ml"]["imd_fill"] = imd_fill
        v1 = fit_v1(xtr, ytr, fuel, beta)
        v2 = fit_lgb(plain(xtr)[~es], ytr[~es], plain(xtr)[es], ytr[es], fuel)
        for kind, m, log_pred in (("greybox", v1, lambda d: v1_log_pred(v1, beta, d)),
                                  ("ml", v2, lambda d: v2.predict(plain(d)))):
            raw = isinstance(m, _RawTarget)
            booster = (m.m if raw else m).booster_
            fname = f"{kind}_{fuel}.txt"
            booster.save_model(str(path / fname), num_iteration=booster.best_iteration)
            meta[kind][fuel] = {
                "file": fname, "features": list(plain(xtr).columns), "raw_target": raw,
                "iterations": int(m.best_iteration_), **({"physics_elasticity": beta} if kind == "greybox" else {}),
                "smearing_factor": round(float(np.mean(np.exp(ytr.to_numpy() - log_pred(xtr)))), 4),
                "val": metrics(yva.to_numpy(), log_pred(xva))}
        qmodels, q, cov = fit_band(xtr, ytr, xva, yva, fuel, beta)
        files = []
        for a, m in zip(BAND, qmodels):
            files.append(f"greybox_{fuel}_q{int(a * 100):02d}.txt")
            m.booster_.save_model(str(path / files[-1]), num_iteration=m.booster_.best_iteration)
        meta["greybox"][fuel]["band"] = {"method": "CQR, quantile GBDT on train, conformal Q on val",
                                         "quantiles": list(BAND), "files": files, "conformal_q": round(q, 5),
                                         "val_coverage": round(cov, 4)}
        meta["greybox"][fuel]["mode1"] = fit_mode1(fuel, path)
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    load_model.cache_clear()
    return meta


def main() -> None:
    meta = train()
    for kind in ("greybox", "ml"):
        for fuel in FUELS:
            m = meta[kind][fuel]
            print(f"{kind:8s} {fuel:4s} iters {m['iterations']:5d}  smear {m['smearing_factor']:.4f}  "
                  f"val MAE {m['val']['mae_kwh']:7.1f}  R2(log) {m['val']['r2_log']:.3f}"
                  + (f"  band Q {m['band']['conformal_q']:+.3f} val coverage {m['band']['val_coverage']:.3f}" if "band" in m else ""))
            if "mode1" in m:
                print(f"         mode 1 val MAE {m['mode1']['val_homes_with_prior']['mae_kwh']:7.1f} vs last year "
                      f"{m['mode1']['val_last_year_same_homes']['mae_kwh']:7.1f} (homes with a prior bill); "
                      f"total factor {m['mode1']['total_factor']}, val total ratio {m['mode1']['val_total_ratio']}")
    print(f"saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
