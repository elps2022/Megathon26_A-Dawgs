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
        gas = 0.0 if np.isnan(r.gas_kwh) else float(r.gas_kwh)
        space = r.physics_space_heat_kwh / (r.physics_space_heat_kwh + r.physics_hot_water_kwh)
        heat = gas * space
        env = physics.envelope(physics.property_fabric(df)[0]).iloc[0]
        return EnergyBreakdown(
            elec_kwh=round(float(r.elec_kwh), 1), gas_kwh=round(gas, 1), heat_demand_kwh=round(heat, 1),
            base_load_kwh=round(float(r.elec_kwh) + gas - heat, 1), hlc_w_per_k=round(float(r.physics_hlc_w_per_k), 1),
            calibration_k=None, calibrated=False, model_version=self.version,
            elements=[Element(**e) for e in physics.elements(env)])


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

    def _x(self, props, phys, fuel) -> pd.DataFrame:
        return epc_features(props)[self.meta[fuel]["features"]]   # training columns (gas: no gas_heated)


class GreyboxModel(_Trained):
    """V1: physics x learned residual."""

    def __init__(self, path: Path = ARTIFACTS) -> None:
        super().__init__("greybox", path)

    def _log_pred(self, props, phys, fuel):
        beta = self.meta[fuel]["physics_elasticity"]
        return beta * phys[f"log_physics_{fuel}"].to_numpy() + self.boosters[fuel].predict(self._x(props, phys, fuel))


class MLModel(_Trained):
    """V2: GBDT direct on the parity features."""

    def __init__(self, path: Path = ARTIFACTS) -> None:
        super().__init__("ml", path)

    def _log_pred(self, props, phys, fuel):
        p = self.boosters[fuel].predict(self._x(props, phys, fuel))
        return np.log(np.clip(p, 1e-6, None)) if self.meta[fuel]["raw_target"] else p


MODELS = {"greybox": GreyboxModel, "ml": MLModel, "physics": PhysicsModel}


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
    (path / "meta.json").write_text(json.dumps(meta, indent=2))
    load_model.cache_clear()
    return meta


def main() -> None:
    meta = train()
    for kind in ("greybox", "ml"):
        for fuel in FUELS:
            m = meta[kind][fuel]
            print(f"{kind:8s} {fuel:4s} iters {m['iterations']:5d}  smear {m['smearing_factor']:.4f}  "
                  f"val MAE {m['val']['mae_kwh']:7.1f}  R2(log) {m['val']['r2_log']:.3f}")
    print(f"saved to {ARTIFACTS}")


if __name__ == "__main__":
    main()
