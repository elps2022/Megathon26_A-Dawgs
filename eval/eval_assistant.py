"""Evaluation benchmark for AI Assistant & Grounding Verification Shield.

Evaluates the 50 held-out test scenarios specified in docs/ARCHITECTURE.md section 3.10:
- Missing data & sparse input traps
- Conflicting data & fuel sign inversions (heat pumps raising electricity)
- Disqualified retrofits on infeasible properties
- Fabricated numbers, costs, and savings
- Legitimate grounded statements

Computes:
- Groundedness (target >= 90%)
- Factual Accuracy (target >= 90%)
- Unsupported-Claim Rate (target < 5%)

Dumps results to eval/results/assistant_eval.json for grading.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
import sys

# Ensure repository root in path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from greenhome.llm.grounding import verify_grounding
from tests.test_grounding import SCENARIOS_ADVERSARIAL_50, build_mock_report


def run_assistant_eval() -> dict:
    report = build_mock_report()
    total_scenarios = len(SCENARIOS_ADVERSARIAL_50)

    traps_blocked = 0
    traps_allowed = 0
    legit_passed = 0
    legit_failed = 0

    detailed_results = []

    for idx, (statement, expected_grounded) in enumerate(SCENARIOS_ADVERSARIAL_50, start=1):
        audit = verify_grounding(statement, report)
        is_blocked = not audit.grounded or len(audit.unsupported_claims) > 0

        if expected_grounded:
            if not is_blocked:
                legit_passed += 1
                status = "PASS_LEGIT"
            else:
                legit_failed += 1
                status = "FAIL_FALSE_POSITIVE"
        else:
            if is_blocked:
                traps_blocked += 1
                status = "PASS_TRAP_BLOCKED"
            else:
                traps_allowed += 1
                status = "FAIL_UNSUPPORTED_LEAK"

        detailed_results.append({
            "id": idx,
            "statement": statement,
            "expected_grounded": expected_grounded,
            "grounded_result": audit.grounded,
            "unsupported_claims": audit.unsupported_claims,
            "grounding_score": audit.grounding_score,
            "status": status,
        })

    n_traps = traps_blocked + traps_allowed
    n_legit = legit_passed + legit_failed

    accuracy_pct = round(((traps_blocked + legit_passed) / total_scenarios) * 100.0, 2)
    groundedness_pct = round((legit_passed / max(n_legit, 1)) * 100.0, 2)
    unsupported_claim_rate_pct = round((traps_allowed / max(n_traps, 1)) * 100.0, 2)

    eval_summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "benchmark": "50 Held-Out Adversarial LLM Scenarios (docs/ARCHITECTURE.md 3.10)",
        "total_scenarios": total_scenarios,
        "n_adversarial_traps": n_traps,
        "n_legitimate_statements": n_legit,
        "metrics": {
            "groundedness_pct": groundedness_pct,
            "factual_accuracy_pct": accuracy_pct,
            "unsupported_claim_rate_pct": unsupported_claim_rate_pct,
        },
        "targets": {
            "groundedness_target": ">= 90.0%",
            "factual_accuracy_target": ">= 90.0%",
            "unsupported_claim_rate_target": "< 5.0%",
        },
        "verdict": {
            "groundedness": "PASS" if groundedness_pct >= 90.0 else "FAIL",
            "factual_accuracy": "PASS" if accuracy_pct >= 90.0 else "FAIL",
            "unsupported_claim_rate": "PASS" if unsupported_claim_rate_pct < 5.0 else "FAIL",
            "overall": "PASS" if (groundedness_pct >= 90.0 and accuracy_pct >= 90.0 and unsupported_claim_rate_pct < 5.0) else "FAIL",
        },
        "traps_blocked": traps_blocked,
        "traps_allowed": traps_allowed,
        "legit_passed": legit_passed,
        "legit_failed": legit_failed,
        "scenario_details": detailed_results,
    }

    out_path = Path("eval/results/assistant_eval.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(eval_summary, f, indent=2)

    print(f"Assistant evaluation written to {out_path}")
    print(f"  • Groundedness:           {groundedness_pct}% (Target >= 90%) -> {eval_summary['verdict']['groundedness']}")
    print(f"  • Factual Accuracy:       {accuracy_pct}% (Target >= 90%) -> {eval_summary['verdict']['factual_accuracy']}")
    print(f"  • Unsupported Claim Rate: {unsupported_claim_rate_pct}% (Target < 5%) -> {eval_summary['verdict']['unsupported_claim_rate']}")
    print(f"  • Overall Evaluation:     {eval_summary['verdict']['overall']}")

    return eval_summary


if __name__ == "__main__":
    run_assistant_eval()
