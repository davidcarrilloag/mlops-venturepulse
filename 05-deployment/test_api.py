"""External API tests for the running VenturePulse FastAPI service.

Mirrors professor's 04-deployment/test_api.py pattern.

Requires the server already running:
    python app.py   (uvicorn on port 9696)

Run with:
    pytest -q test_api.py
"""

import requests

BASE_URL = "http://localhost:9696"


def test_health_endpoint():
    resp = requests.get(f"{BASE_URL}/health")
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()
    assert data.get("status") == "ok"
    assert isinstance(data.get("run_id"), str) and len(data["run_id"]) > 5


def test_predict_endpoint():
    payload = {
        "sector": "Fintech",
        "location": "NYC",
        "funding_stage": "Seed",
        "founder_background": "Technical",
        "team_diversity": "High",
        "initial_funding_amount": 500000,
        "team_size": 8,
        "months_since_founded": 12,
        "capital_efficiency": 0.65,
        "market_timing": 0.72,
        "pmf_score": 0.58,
        "cac_efficiency": 0.45,
        "burn_rate": 50000,
        "network_strength": 0.60,
        "founder_commitment": 0.85,
        "technical_moat": 0.55,
        "revenue_growth": 0.30,
        "has_customers": 1,
        "prev_experience": 0.70,
        "investor_quality": 0.65,
        "tier1_location": 1,
        "hot_sector": 1,
    }
    resp = requests.post(f"{BASE_URL}/predict", json=payload)
    assert resp.status_code == 200, (
        f"Unexpected status: {resp.status_code} body={resp.text}"
    )
    data = resp.json()

    # Validate structure
    assert "prediction" in data
    assert "probability" in data
    assert "confidence" in data
    assert "flagged_for_review" in data
    assert "model_version" in data

    # Sanity checks
    assert data["prediction"] in [0, 1]
    assert 0.0 <= data["probability"] <= 1.0
    assert isinstance(data["model_version"], str) and len(data["model_version"]) > 5
