"""Comprehensive evaluation reporter for Megathon X (PS2: Green Home Score).

Executes and reports 4-tier validation matrix:
1. Model evaluation: GBDT vs cell-median & log-linear baselines (MAE, MAPE, R2).
2. LLM grounding audit: 50-scenario adversarial hallucination benchmark.
3. Rule layer verification: weights and monotonicity of the piecewise score mappings.

Energy numbers are 5-fold CV on the NEED train split plus the validation split; the test split is untouched
until the final evaluation. The grounding numbers measure the checker on labelled statements (no LLM yet).

Usage:
    python scripts/run_eval.py
"""

from pathlib import Path
import json
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenhome.llm.grounding import verify_grounding
from greenhome.score.score import load_mappings
from tests.test_grounding import SCENARIOS_ADVERSARIAL_50, build_mock_report


def print_banner(title: str):
    print("\n" + "=" * 70)
    print(f" {title.upper()}")
    print("=" * 70)


def eval_models():
    print_banner("1. Energy Model Evaluation (NEED 5-fold CV + validation split; test untouched)")
    res_path = Path("eval/results/energy_first_cut.json")
    if not res_path.exists():
        print("  [!] eval/results/energy_first_cut.json not found.")
        return

    with open(res_path) as f:
        data = json.load(f)

    for fuel, label in (("gas", "gas"), ("elec", "electricity")):
        if fuel not in data:
            print(f"  [!] no {label} results in {res_path}")
            continue
        print(f"\n[{label.upper()} CONSUMPTION MODEL — 5-Fold Cross Validation on train]")
        print(f"{'Model':<18} | {'MAE (kWh)':<10} | {'MAPE (%)':<10} | {'Med APE (%)':<12} | {'R²':<8}")
        print("-" * 68)
        cv = data[fuel]["cv"]
        for m_name in ("global_median", "cell_median", "log_linear", "lightgbm"):
            if m_name in cv:
                m = cv[m_name]
                print(f"{m_name:<18} | {m['mae_kwh']:<10.1f} | {m['mape_pct']:<10.2f} | {m['median_ape_pct']:<12.2f} | {m['r2_log']:<8.4f}")

        val = data[fuel]["val"]
        print(f"\n[{label.upper()} VALIDATION SPLIT (first cut, untuned)]")
        print(f"LightGBM Validation MAE: {val['lightgbm']['mae_kwh']:.1f} kWh | MAPE: {val['lightgbm']['mape_pct']:.2f}% | R²: {val['lightgbm']['r2_log']:.4f}")
        improvement = (val['global_median']['mae_kwh'] - val['lightgbm']['mae_kwh']) / val['global_median']['mae_kwh'] * 100.0
        vs_ll = (val['log_linear']['mae_kwh'] - val['lightgbm']['mae_kwh']) / val['log_linear']['mae_kwh'] * 100.0
        print(f"--> LightGBM MAE vs global median: {improvement:+.1f}% reduction; vs log-linear: {vs_ll:+.1f}% reduction.")


def eval_grounding():
    print_banner("2. Grounding checker on labelled statements (no LLM in the loop yet)")
    report = build_mock_report()

    total = len(SCENARIOS_ADVERSARIAL_50)
    correctly_handled = 0
    traps_blocked = 0
    legit_passed = 0

    for statement, expected_pass in SCENARIOS_ADVERSARIAL_50:
        res = verify_grounding(statement, report)
        if expected_pass:
            if res.grounded and len(res.unsupported_claims) == 0:
                correctly_handled += 1
                legit_passed += 1
        else:
            if not res.grounded or len(res.unsupported_claims) > 0:
                correctly_handled += 1
                traps_blocked += 1

    n_legit = sum(1 for _, ok in SCENARIOS_ADVERSARIAL_50 if ok)
    n_traps = total - n_legit
    accuracy = correctly_handled / total * 100.0
    print(f"Total Evaluated Statements: {total}")
    print(f"  • Traps blocked:          {traps_blocked} / {n_traps} ({traps_blocked / max(n_traps, 1) * 100:.1f}%)")
    print(f"  • Legitimate passed:      {legit_passed} / {n_legit} ({legit_passed / max(n_legit, 1) * 100:.1f}%)")
    print(f"  • Checker accuracy:       {accuracy:.1f}% -> {'PASS' if accuracy >= 90.0 else 'FAIL'} (target >= 90%)")


def eval_invariants():
    print_banner("3. Deterministic Invariant & Monotonicity Verification")
    mappings = load_mappings()
    total_w = sum(mappings["weights"].values())
    print(f"Score Component Weights Sum: {total_w:.4f} (Target: 1.0000) -> {'PASS' if abs(total_w - 1.0) < 1e-4 else 'FAIL'}")

    comps = len(mappings["weights"])
    print(f"Score Components Defined:    {comps} components (Target: 7) -> {'PASS' if comps == 7 else 'FAIL'}")
    piecewise = {"consumption_intensity": (mappings["consumption_intensity"]["points"], -1),
                 "insulation.loft_mm": (mappings["insulation"]["loft_mm_points"], +1)}
    for name, (pts, direction) in piecewise.items():
        xs, ys = [x for x, _ in pts], [y for _, y in pts]
        ok = all(b > a for a, b in zip(xs, xs[1:])) and all(direction * (b - a) >= 0 for a, b in zip(ys, ys[1:]))
        print(f"Monotonic {name:<22}: {'PASS' if ok else 'FAIL'}")
    print("Categorical mappings (heating, glazing, controls, walls): ordering is tested in tests/test_score.py")


def main():
    eval_models()
    eval_grounding()
    eval_invariants()
    print("\n" + "=" * 70)
    print(" EVALUATION SUITES EXECUTED")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
