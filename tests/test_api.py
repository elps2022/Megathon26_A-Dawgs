"""Integration tests for FastAPI endpoints and Grounding Inspector."""

from fastapi.testclient import TestClient
from api.main import app

client = TestClient(app)


def test_health_endpoint() -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"status": "ok", "service": "greenhome-api"}


def test_property_and_report_endpoints() -> None:
    # 1. Get raw property
    res = client.get("/property/P00001")
    assert res.status_code == 200
    p_data = res.json()
    assert p_data["property_id"] == "P00001"

    # 2. Get baseline report
    rep_res = client.get("/report/P00001")
    assert rep_res.status_code == 200
    rep = rep_res.json()
    assert "score" in rep
    assert "recommendations" in rep
    assert rep["package_saving"]["total_kwh"] >= 0.0

    # 3. Post scenario (what-if: add solar and heat pump)
    scen_res = client.post("/scenario/P00001", json={"retrofits": ["heat_pump", "solar_pv"]})
    assert scen_res.status_code == 200
    scen = scen_res.json()
    assert scen["score_after"]["total"] >= rep["score"]["total"]


def test_portfolio_endpoint() -> None:
    res = client.get("/portfolio?sample=50")
    assert res.status_code == 200
    data = res.json()
    assert data["total_properties"] == 50
    assert data["total_financed_emissions_tonnes"] >= 0.0
    assert len(data["high_potential_targets"]) > 0


def test_grounding_verifier() -> None:
    # Retrieve report for P00001
    rep = client.get("/report/P00001").json()
    total_saved = rep["package_saving"]["total_kwh"]

    # Factual response grounded in report
    grounded_text = f"Your property can save {total_saved} kWh per year with recommended upgrades."
    res = client.post("/verify_grounding", json={"llm_text": grounded_text, "report": rep})
    assert res.status_code == 200
    data = res.json()
    assert data["grounded"] is True
    assert data["grounding_score_pct"] == 100.0

    # Hallucinated response with invented numbers
    fake_text = "This upgrade will save 99,999 kWh and cost exactly £49,999."
    res2 = client.post("/verify_grounding", json={"llm_text": fake_text, "report": rep})
    assert res2.status_code == 200
    data2 = res2.json()
    assert data2["grounded"] is False
    assert len(data2["unsupported_claims"]) > 0
