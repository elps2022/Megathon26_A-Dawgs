"""Exploratory analysis of the energy-model data and results (ARCHITECTURE 3.5 step 1), for the report.

    uv run python eval/eda_energy.py      -> eval/figures/*.png, eval/results/energy_eda.json

Data figures use the NEED TRAIN split only. The calibration figure reads the already-scored test predictions
(eval/results/test_predictions.csv); it reports, it doesn't choose anything.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from greenhome.models.energy_greybox import with_physics  # noqa: E402
from greenhome.models.need_data import TARGET_YEAR, dataset, load_need, load_split  # noqa: E402

FIG = ROOT / "eval" / "figures"
RESULTS = ROOT / "eval" / "results"
LABEL = {"gas": "Gas (gas-heated homes)", "elec": "Electricity (all homes)"}
EPC = {1: "A/B", 2: "C", 3: "D", 4: "E", 5: "F/G"}
AGE = {1: "pre-1930", 2: "1930-72", 3: "1973-99", 4: "2000+"}
BANDS = ["prop_type", "age_band", "floor_area_band", "epc_band", "region"]
plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.spines.top": False, "axes.spines.right": False})


def save(fig, name: str) -> str:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG / name)
    plt.close(fig)
    return f"eval/figures/{name}"


def data_summary() -> dict:
    d = load_need()
    split = load_split(d)
    tr = d[split == "train"]
    out = {"homes": len(d), "train_homes": len(tr)}
    for fuel, (cons, flag) in {"gas": ("Gcons", "GasValFlag"), "elec": ("Econs", "ElecValFlag")}.items():
        v = tr[f"{cons}{TARGET_YEAR}"]
        f = tr[f"{flag}{TARGET_YEAR}"]
        out[fuel] = {"missing_share": round(float(v.isna().mean()), 4),
                     "flag_counts": f.value_counts().to_dict(),
                     "min_reported": float(v.min()), "max_reported": float(v.max()),
                     "median": float(v.median()), "mean": round(float(v.mean()), 1)}
    out["truncation_note"] = ("NEED blanks gas outside 1,000-50,000 kWh and electricity outside 500-25,000 kWh: the "
                              "target is truncated, so extreme homes are under-represented (documented, not fixed)")
    return out


def figures() -> dict:
    out, figs = {}, []
    data = {}
    for fuel in ("gas", "elec"):
        x, y, s = dataset(fuel)
        x = with_physics(x, fuel)
        data[fuel] = (x[s == "train"], y[s == "train"])

    # 1. Target distributions (log scale is near-symmetric: why the models work in log kWh)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    for a, fuel in zip(ax, ("gas", "elec")):
        y = np.exp(data[fuel][1])
        a.hist(y, bins=np.logspace(np.log10(y.min()), np.log10(y.max()), 60), color="#2e7d5b")
        a.set_xscale("log"); a.set_title(LABEL[fuel]); a.set_xlabel("annual kWh (log scale)"); a.set_ylabel("homes")
    figs.append(save(fig, "eda_target_distribution.png"))

    # 2. Use by EPC band and by age (the expected ordering, and how much overlap there is)
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.8))
    x, y = data["gas"]
    for a, col, names in ((ax[0], "epc_band", EPC), (ax[1], "age_band", AGE)):
        groups = [np.exp(y[x[col] == k]) for k in names]
        a.boxplot(groups, tick_labels=list(names.values()), showfliers=False)
        a.set_ylabel("gas kWh / yr"); a.set_xlabel(col.replace("_", " "))
    ax[0].set_title("Gas use by EPC band (train)"); ax[1].set_title("Gas use by age band (train)")
    figs.append(save(fig, "eda_gas_by_band.png"))
    out["gas_median_by_epc"] = {EPC[k]: float(np.exp(y[x.epc_band == k]).median()) for k in EPC}

    # 3. The noise ceiling: spread of homes inside identical description cells
    fig, ax = plt.subplots(figsize=(4.5, 2.8))
    for fuel, colour in (("gas", "#2e7d5b"), ("elec", "#c47f17")):
        x, y = data[fuel]
        cell = x[BANDS].astype(object).fillna("none").astype(str).agg("|".join, axis=1)
        dev = y - y.groupby(cell).transform("median")
        big = y.groupby(cell).transform("size") >= 20
        ratio = np.exp(dev[big])
        ax.hist(ratio, bins=np.linspace(0.2, 3, 60), alpha=0.6, color=colour, label=fuel, density=True)
        out[f"{fuel}_within_cell"] = {"homes_in_cells_ge_20": int(big.sum()),
                                      "p10_ratio_to_cell_median": round(float(ratio.quantile(0.1)), 3),
                                      "p90_ratio_to_cell_median": round(float(ratio.quantile(0.9)), 3)}
    ax.set_xlabel("use / median of identical homes (type, age, size, EPC, region)")
    ax.set_title("Identical descriptions, very different use"); ax.legend()
    figs.append(save(fig, "eda_within_cell_spread.png"))

    # 4. Physics vs metered gas: the prebound effect
    x, y = data["gas"]
    fig, ax = plt.subplots(figsize=(4.5, 3.2))
    sample = np.random.default_rng(0).choice(len(x), 4000, replace=False)
    ax.scatter(np.exp(x.log_physics.to_numpy()[sample]), np.exp(y.to_numpy()[sample]), s=2, alpha=0.25, color="#2e7d5b")
    lim = [2000, 80000]
    ax.plot(lim, lim, "k--", lw=0.8, label="actual = physics")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(lim); ax.set_ylim(800, 60000)
    ax.set_xlabel("physics prediction, kWh"); ax.set_ylabel("metered gas, kWh"); ax.legend(loc="upper left")
    ax.set_title("Physics overpredicts real homes ~2x")
    figs.append(save(fig, "eda_physics_vs_actual.png"))
    out["gas_median_actual_over_physics"] = round(float(np.exp(np.median(y - x.log_physics))), 3)

    # 5. Retrofit check: observed vs predicted change for real installs (val and test splits)
    v1 = json.loads((RESULTS / "energy_v1.json").read_text())["gas"]["retrofit_check"]
    te = json.loads((RESULTS / "energy_test.json").read_text())["retrofit_installs_test"]
    rows = []
    for m, label in (("loft_insulation", "Loft"), ("cavity_wall_insulation", "Cavity wall")):
        for split, src in (("val", v1[m]["val"]), ("test", te[m])):
            rows.append((f"{label}\n({split}, n={src['n']})", src["observed_median_change"], src["v1_median_change"],
                         src["physics_only_median_change"]))
    fig, ax = plt.subplots(figsize=(6.5, 2.8))
    xs = np.arange(len(rows)); w = 0.27
    for i, (name, colour) in enumerate((("Observed (real installs)", "#333333"), ("V1 model", "#2e7d5b"),
                                        ("Physics only", "#bbbbbb"))):
        ax.bar(xs + (i - 1) * w, [100 * r[i + 1] for r in rows], w, label=name, color=colour)
    ax.set_xticks(xs, [r[0] for r in rows]); ax.set_ylabel("median change in gas use, %"); ax.axhline(0, color="k", lw=0.5)
    ax.legend(fontsize=7); ax.set_title("Retrofit savings vs real installs (held out)")
    figs.append(save(fig, "eval_retrofit_check.png"))

    # 6. Test calibration: mean actual vs mean predicted by decile of V1 prediction, and band coverage
    tp = pd.read_csv(RESULTS / "test_predictions.csv")
    fig, ax = plt.subplots(1, 2, figsize=(8, 2.9))
    for a, fuel in zip(ax, ("gas", "elec")):
        t = tp[tp.fuel == fuel]
        dec = pd.qcut(t.v1_mean, 10, labels=False)
        g = t.groupby(dec).agg(pred=("v1_mean", "mean"), act=("actual_kwh", "mean"),
                               lo=("v1_p10", "mean"), hi=("v1_p90", "mean"))
        a.plot(g.pred, g.act, "o-", color="#2e7d5b", label="mean metered use")
        a.fill_between(g.pred, g.lo, g.hi, alpha=0.15, color="#2e7d5b", label="mean p10-p90 band")
        lim = [g.pred.min() * 0.9, g.pred.max() * 1.1]
        a.plot(lim, lim, "k--", lw=0.8)
        cov = ((t.actual_kwh >= t.v1_p10) & (t.actual_kwh <= t.v1_p90)).mean()
        a.set_title(f"{fuel.capitalize()} (test): band covers {cov:.0%}")
        a.set_xlabel("V1 predicted kWh (decile mean)"); a.set_ylabel("kWh")
    ax[0].legend(fontsize=7)
    figs.append(save(fig, "eval_test_calibration.png"))
    out["figures"] = figs
    return out


def main() -> None:
    out = {"data": data_summary(), **figures()}
    RESULTS.mkdir(parents=True, exist_ok=True)
    (RESULTS / "energy_eda.json").write_text(json.dumps(out, indent=2, default=str))
    print(json.dumps({k: v for k, v in out.items() if k != "data"}, indent=1, default=str))


if __name__ == "__main__":
    main()
