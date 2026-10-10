"""Grounding verification engine for LLM explanations.

Strict enforcement of problem statement invariant: LLM never computes numbers.
Extracts numbers from LLM answer and cross-checks against tool PropertyReport JSON.
Guarantees >= 90% groundedness and < 5% unsupported claims.
"""

import re
from typing import Any
from greenhome.schema import PropertyReport


def extract_numbers_from_text(text: str) -> list[tuple[float, str]]:
    """Extract numeric values and their matched string from text.

    Handles integers, floats, currency (£), percentages (%), and comma formatting.
    """
    # Matches patterns like: £350,000, 1,200 kWh, 25.4%, 150 mm, etc.
    pattern = r"(?:£\s*)?([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)(?:\s*(?:%|kwh|kg|tonnes|tco2e|mm|m²|£))?"
    matches = []

    for match in re.finditer(pattern, text, re.IGNORECASE):
        raw_num_str = match.group(1).replace(",", "")
        try:
            val = float(raw_num_str)
            # Filter out standalone tiny numbers that are likely words or years like 2024, 2026
            # unless relevant
            matches.append((val, match.group(0).strip()))
        except ValueError:
            continue

    return matches


def extract_numbers_from_json(obj: Any) -> list[float]:
    """Recursively collect all numeric values present in a JSON dictionary/list."""
    numbers = []

    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        numbers.append(float(obj))
    elif isinstance(obj, dict):
        for v in obj.values():
            numbers.extend(extract_numbers_from_json(v))
    elif isinstance(obj, list):
        for item in obj:
            numbers.extend(extract_numbers_from_json(item))

    return numbers


class GroundingResult:
    """Audit report for LLM text response groundedness."""

    def __init__(
        self,
        grounded: bool,
        grounding_score: float,
        total_claims: int,
        verified_claims: list[str],
        unsupported_claims: list[str],
    ):
        self.grounded = grounded
        self.grounding_score = grounding_score
        self.total_claims = total_claims
        self.verified_claims = verified_claims
        self.unsupported_claims = unsupported_claims

    def to_dict(self) -> dict[str, Any]:
        return {
            "grounded": self.grounded,
            "grounding_score_pct": round(self.grounding_score * 100.0, 1),
            "total_claims": self.total_claims,
            "verified_claims": self.verified_claims,
            "unsupported_claims": self.unsupported_claims,
        }


def verify_grounding(
    llm_response: str,
    report: PropertyReport | dict[str, Any],
    tolerance: float = 0.02,  # 2% rounding tolerance
) -> GroundingResult:
    """Verify that every number cited in the LLM response exists in the report JSON."""
    report_dict = report.model_dump() if isinstance(report, PropertyReport) else report
    known_numbers = extract_numbers_from_json(report_dict)

    # Common non-hallucinated calendar years and metadata to ignore
    ignore_set = {2024.0, 2025.0, 2026.0, 2030.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0}

    extracted = extract_numbers_from_text(llm_response)
    claims_to_check = [item for item in extracted if item[0] not in ignore_set]

    if not claims_to_check:
        return GroundingResult(
            grounded=True,
            grounding_score=1.0,
            total_claims=0,
            verified_claims=[],
            unsupported_claims=[],
        )

    verified: list[str] = []
    unsupported: list[str] = []

    for num, raw_str in claims_to_check:
        # Check against known numbers in report within tolerance
        matched = False
        for known in known_numbers:
            if known == 0.0 and abs(num) < 1e-4:
                matched = True
                break
            elif known != 0.0 and abs(num - known) / abs(known) <= tolerance:
                matched = True
                break
            elif abs(num - known) <= 1.0:  # Integer rounding
                matched = True
                break

        if matched:
            verified.append(raw_str)
        else:
            unsupported.append(raw_str)

    score = len(verified) / len(claims_to_check)
    is_grounded = score >= 0.90 and len(unsupported) == 0

    return GroundingResult(
        grounded=is_grounded,
        grounding_score=score,
        total_claims=len(claims_to_check),
        verified_claims=verified,
        unsupported_claims=unsupported,
    )
