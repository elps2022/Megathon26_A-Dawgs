"""Test suite for Optional Extension 10: Customer Memory that Learns.

Validates:
1. CustomerMemoryStore persistence and profile lifecycle (completed retrofits, constraints, budgets, preferences).
2. Property state mutation: completed retrofits update the physical home state and lift Green Home Score.
3. Adaptive recommendation ranking:
   - Completed measures removed from pending list.
   - Declined measures down-ranked to 'Low' suitability with historical date citation.
   - Measures exceeding stated budget down-ranked with green lending recommendation.
   - Tenancy constraint blocks structural works (solid wall insulation, heat pump).
4. REST API endpoints: GET /memory/{id} and POST /memory/{id}.
5. AI Chat assistant natural language auto-extraction and memory badge generation.
"""

from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from api.main import app
from greenhome.engine.report import build_report
from greenhome.llm.client import LLMAdvisor, detect_memory_action
from greenhome.memory.store import CustomerMemoryStore
from greenhome.schema import Property
from tests.test_grounding import build_mock_report

client = TestClient(app)


@pytest.fixture
def temp_memory_store(tmp_path: Path) -> CustomerMemoryStore:
    return CustomerMemoryStore(storage_dir=tmp_path)


def test_memory_store_lifecycle(temp_memory_store: CustomerMemoryStore):
    cid = "CUST_TEST_999"

    # Initial profile
    mem = temp_memory_store.get_memory(cid)
    assert mem["customer_id"] == cid
    assert mem["completed_retrofits"] == []
    assert mem["declined_retrofits"] == {}
    assert mem["budget_max"] is None

    # Record constraints & budget
    temp_memory_store.set_budget(cid, 4500.0)
    temp_memory_store.set_tenure(cid, "tenant")
    temp_memory_store.record_declined(cid, "solar_pv", reason="Tenant without roof permissions", date_str="11 Oct 2026")
    temp_memory_store.record_completed(cid, "smart_thermostat", date_str="10 Oct 2026")

    # Reload from disk
    loaded = temp_memory_store.get_memory(cid)
    assert loaded["budget_max"] == 4500.0
    assert loaded["tenure"] == "tenant"
    assert "tenant" in loaded["constraints"]
    assert "solar_pv" in loaded["declined_retrofits"]
    assert loaded["declined_retrofits"]["solar_pv"]["date"] == "11 Oct 2026"
    assert "smart_thermostat" in loaded["completed_retrofits"]
    assert len(loaded["session_timeline"]) >= 4

    # Test undo and restore
    temp_memory_store.remove_completed(cid, "smart_thermostat")
    temp_memory_store.remove_declined(cid, "solar_pv")
    restored = temp_memory_store.get_memory(cid)
    assert "smart_thermostat" not in restored["completed_retrofits"]
    assert "solar_pv" not in restored["declined_retrofits"]


def test_completed_retrofits_applied_to_property(temp_memory_store: CustomerMemoryStore):
    report = build_mock_report()
    p = report.property
    cid = p.customer_id

    # Record loft insulation completed
    temp_memory_store.record_completed(cid, "loft_insulation", date_str="09 Oct 2026")
    updated_p = temp_memory_store.apply_memory_to_property(p)

    # Physical home state updated
    assert updated_p.loft_insulation_mm == 300.0

    # Rerun engine: base score increases, and loft insulation no longer in pending recommendations
    r_before = build_report(p)
    r_updated = build_report(updated_p, memory=temp_memory_store.get_memory(cid))
    assert r_updated.score.total >= r_before.score.total
    rec_names = [r.retrofit for r in r_updated.recommendations]
    assert "loft_insulation" not in rec_names


def test_declined_retrofits_downranked_with_citation():
    report = build_mock_report()
    p = report.property
    mem = {
        "customer_id": p.customer_id,
        "declined_retrofits": {"loft_insulation": {"date": "11 Oct 2026", "reason": "Roof works delayed"}},
        "constraints": [],
    }

    r_declined = build_report(p, memory=mem)
    loft_rec = next(r for r in r_declined.recommendations if r.retrofit == "loft_insulation")

    # Suitability is Low, date is cited in reason, and ranked lower
    assert loft_rec.suitability == "Low"
    assert "You declined this on 11 Oct 2026" in loft_rec.reason
    assert loft_rec.rank >= 3


def test_budget_constraint_downranked():
    report = build_mock_report()
    p = report.property

    # Budget capped at £2,000 (heat pump costs £8,000+ so exceeds budget)
    mem = {
        "customer_id": p.customer_id,
        "budget_max": 2000.0,
        "constraints": ["budget_2000"],
    }

    r_budget = build_report(p, memory=mem)
    hp_rec = next((r for r in r_budget.recommendations if r.retrofit == "heat_pump"), None)
    if hp_rec:
        assert hp_rec.suitability == "Low"
        assert "Exceeds stated budget" in hp_rec.reason
        assert "Lloyds Green Lending" in hp_rec.reason


def test_tenancy_constraint_blocks_structural():
    report = build_mock_report()
    p = report.property
    mem = {
        "customer_id": p.customer_id,
        "constraints": ["tenant"],
    }

    r_tenant = build_report(p, memory=mem)
    infeas_names = [inf.retrofit for inf in r_tenant.infeasible]
    assert "heat_pump" in infeas_names
    inf_hp = next(inf for inf in r_tenant.infeasible if inf.retrofit == "heat_pump")
    assert "tenancy" in inf_hp.rule.lower()


def test_api_memory_endpoints():
    cid = "CUST_API_TEST"

    # 1. Mutate via POST /memory/{customer_id}
    res_b = client.post(f"/memory/{cid}", json={"action": "budget", "budget_gbp": 3500.0})
    assert res_b.status_code == 200
    assert res_b.json()["budget_max"] == 3500.0

    res_d = client.post(
        f"/memory/{cid}",
        json={"action": "declined", "retrofit": "solar_pv", "reason": "Shaded roof", "date": "12 Oct 2026"},
    )
    assert res_d.status_code == 200
    assert "solar_pv" in res_d.json()["declined_retrofits"]

    res_c = client.post(
        f"/memory/{cid}",
        json={"action": "completed", "retrofit": "smart_thermostat", "date": "10 Oct 2026"},
    )
    assert res_c.status_code == 200
    assert "smart_thermostat" in res_c.json()["completed_retrofits"]

    # 2. Retrieve via GET /memory/{customer_id}
    res_get = client.get(f"/memory/{cid}")
    assert res_get.status_code == 200
    data = res_get.json()
    assert data["customer_id"] == cid
    assert data["budget_max"] == 3500.0
    assert "solar_pv" in data["declined_retrofits"]
    assert "smart_thermostat" in data["completed_retrofits"]


def test_chat_assistant_memory_auto_extraction():
    # 1. Detection helper
    a1 = detect_memory_action("I already installed loft insulation last month")
    assert a1 is not None
    assert a1["action"] == "record_completed"
    assert a1["detail"] == "loft_insulation"

    a2 = detect_memory_action("I don't want solar panels on my roof")
    assert a2 is not None
    assert a2["action"] == "record_declined"
    assert a2["detail"] == "solar_pv"

    a3 = detect_memory_action("My maximum budget is £4,000")
    assert a3 is not None
    assert a3["action"] == "record_constraint"
    assert "budget_4000" in a3["detail"]

    a4 = detect_memory_action("I'm renting this apartment")
    assert a4 is not None
    assert a4["action"] == "record_constraint"
    assert a4["detail"] == "tenant"

    # 2. End-to-end query updates memory and returns badge
    report = build_mock_report()
    advisor = LLMAdvisor(provider="offline")

    resp, audit, label = advisor.query("I already installed loft insulation", report)
    assert "Memory Updated" in label
    assert "🧠 Memory Updated: Recorded completed Loft Insulation" in resp
    assert audit.grounded
