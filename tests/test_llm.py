"""Unit tests for LLM Advisor client and deterministic grounding shield."""

import json
from unittest.mock import MagicMock, patch

import pytest
from greenhome.llm.client import (
    LLMAdvisor,
    format_report_for_prompt,
    generate_rule_fallback,
)
from greenhome.llm.grounding import verify_grounding
from tests.test_grounding import build_mock_report


def test_format_report_for_prompt():
    report = build_mock_report()
    json_str = format_report_for_prompt(report)
    parsed = json.loads(json_str)

    assert parsed["green_home_score"] == 58
    assert parsed["package_saving_kwh"] == 8900.0
    assert parsed["package_gas_kwh"] == 10300.0
    assert parsed["package_elec_kwh"] == -1400.0
    assert parsed["package_carbon_cut_kg"] == 1815.0
    assert len(parsed["recommendations"]) == 2


def test_rule_fallback_groundedness():
    report = build_mock_report()
    queries = [
        "How do I reach EPC Band C?",
        "What government grants can I claim?",
        "How much energy will I save on fuel bills?",
        "How much carbon will my home cut?",
        "Tell me about my home.",
    ]
    for q in queries:
        resp = generate_rule_fallback(q, report)
        audit = verify_grounding(resp, report)
        assert audit.grounded, f"Fallback for query '{q}' failed grounding: {audit.unsupported_claims}"
        assert audit.grounding_score >= 0.90


def test_llm_advisor_offline_mode():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="offline")
    resp, audit, label = advisor.query("How do I save energy?", report)

    assert audit.grounded
    assert "Deterministic Rule Engine" in label


def test_llm_advisor_attempt1_grounded():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="anthropic", api_key="dummy_key")

    grounded_mock_text = (
        "Your home currently has a Green Home Score of 58/100. "
        "Implementing the recommended measures can save up to 8,900 kWh per year, "
        "comprising 10,300 kWh in gas savings."
    )

    with patch.object(advisor, "_call_anthropic", return_value=grounded_mock_text):
        resp, audit, label = advisor.query("Tell me about savings", report)
        assert audit.grounded
        assert resp == grounded_mock_text
        assert "Anthropic" in label
        assert "Shield Corrected" not in label


def test_llm_advisor_attempt2_shield_correction():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="anthropic", api_key="dummy_key")

    hallucinated_text = "You will save £99,999 and 88,888 kWh every single month!"
    corrected_text = (
        "Corrected: The engine estimates 8,900 kWh total annual savings with 10,300 kWh gas reduction."
    )

    # First call returns hallucination; retry returns grounded text
    with patch.object(advisor, "_call_anthropic", side_effect=[hallucinated_text, corrected_text]) as mock_call:
        resp, audit, label = advisor.query("What are my savings?", report)
        assert mock_call.call_count == 2
        assert audit.grounded
        assert resp == corrected_text
        assert "Shield Corrected" in label


def test_llm_advisor_double_failure_fallback():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="anthropic", api_key="dummy_key")

    hallucinated_text_1 = "You will save £99,999 and 88,888 kWh!"
    hallucinated_text_2 = "Still saving £77,777 and 66,666 kWh!"

    with patch.object(advisor, "_call_anthropic", side_effect=[hallucinated_text_1, hallucinated_text_2]):
        resp, audit, label = advisor.query("What are my savings?", report)
        assert audit.grounded
        assert "Shield Intervened" in label
        assert "Unverified figures were blocked" in resp


def test_history_starting_with_assistant_is_normalised():
    """The chat UI seeds history with an assistant greeting; providers require user-first."""
    report = build_mock_report()
    advisor = LLMAdvisor(provider="anthropic", api_key="dummy_key")
    history = [
        {"role": "assistant", "content": "Welcome greeting"},
        {"role": "user", "content": "First question"},
        {"role": "assistant", "content": "First answer"},
    ]
    with patch.object(advisor, "_call_anthropic", return_value="Your score is 58/100.") as mock_call:
        advisor.query("Follow-up?", report, history=history)
        sent = mock_call.call_args[0][0]
        assert sent[0]["role"] == "user"
        assert [m["role"] for m in sent] == ["user", "assistant", "user"]
        assert sent[-1]["content"] == "Follow-up?"


def test_gemini_preferred_over_anthropic(monkeypatch):
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "a-key")
    monkeypatch.setenv("GEMINI_API_KEY", "g-key")
    advisor = LLMAdvisor()
    assert advisor.provider == "gemini"
    assert advisor.api_key == "g-key"


def test_gemini_call_uses_header_and_skips_thoughts():
    advisor = LLMAdvisor(provider="gemini", api_key="secret-key", model="gemini-3.8-flash")
    fake_resp = MagicMock()
    fake_resp.json.return_value = {
        "candidates": [{"content": {"parts": [
            {"text": "internal reasoning", "thought": True},
            {"text": "Your score is 58/100."},
        ]}}]
    }
    with patch("greenhome.llm.client.httpx.Client") as client_cls:
        client = client_cls.return_value.__enter__.return_value
        client.post.return_value = fake_resp
        out = advisor._call_gemini([{"role": "user", "content": "hi"}], "sys")
        url = client.post.call_args[0][0]
        headers = client.post.call_args[1]["headers"]
    assert out == "Your score is 58/100."
    assert "secret-key" not in url
    assert headers["x-goog-api-key"] == "secret-key"


def test_guardrails_safety_blocks():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="offline")

    offensive_queries = [
        "women are lesser than men",
        "black lives dont matter?",
        "white supremacy is the way",
    ]
    for q in offensive_queries:
        resp, audit, label = advisor.query(q, report)
        assert "Safety Guardrail (Blocked" in label
        assert "violates our acceptable use" in resp
        assert audit.grounded


def test_guardrails_off_topic_and_gibberish():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="offline")

    # Gibberish
    resp_g, audit_g, label_g = advisor.query("wdwdpwefpoekd", report)
    assert "Gibberish" in label_g
    assert "couldn't understand that message" in resp_g
    assert audit_g.grounded

    # Trivia / off-topic
    resp_o, audit_o, label_o = advisor.query("what is the colour of the sky during night", report)
    assert "Off-Topic" in label_o
    assert "outside the scope of home energy" in resp_o
    assert audit_o.grounded

    # Random single word
    resp_d, audit_d, label_d = advisor.query("dogs", report)
    assert "Off-Topic" in label_d
    assert "outside the scope" in resp_d
    assert audit_d.grounded


def test_what_if_scenario_rerun_in_query():
    report = build_mock_report()
    advisor = LLMAdvisor(provider="offline")

    # Asking what-if scenario triggers engine recalculation
    resp, audit, label = advisor.query("What if I add solar panels to my roof?", report)
    assert "Scenario Engine" in label
    assert "solar_pv" in label
    assert "Simulating Solar Pv" in resp
    assert audit.grounded
    assert len(audit.unsupported_claims) == 0


def test_advisor_tools_and_execution():
    from greenhome.llm.client import ADVISOR_TOOLS, execute_advisor_tool

    # Verify tool declarations
    tool_names = [t["name"] for t in ADVISOR_TOOLS]
    assert "get_property_report" in tool_names
    assert "run_scenario" in tool_names
    assert "get_customer_memory" in tool_names
    assert "update_customer_memory" in tool_names

    # Test tool execution
    report = build_mock_report()
    prop = report.property

    # 1. get_property_report
    rep_res = execute_advisor_tool("get_property_report", {"property_id": prop.property_id}, report, prop)
    assert rep_res["green_home_score"] == report.score.total

    # 2. run_scenario tool
    scen_res = execute_advisor_tool("run_scenario", {"property_id": prop.property_id, "retrofits": ["solar_pv"]}, report, prop)
    assert "green_home_score" in scen_res

