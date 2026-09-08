import pytest
from fastapi.testclient import TestClient

from src.pipelines.run_pipeline import run
from src.serving.api import app


@pytest.fixture(scope="module")
def client():
    run(rows=1500, config="configs/features.yaml")
    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_realtime_predict(client):
    r = client.get("/predict/1")
    assert r.status_code == 200
    assert 0.0 <= r.json()["attrition_risk"] <= 1.0


def test_predict_vector(client):
    payload = {
        "tenure_bucket": 1,
        "comp_ratio_z": -1.2,
        "engagement_gap": 1.5,
        "promo_velocity": 0.0,
        "overtime_load": 1.4,
        "mobility_flag": 1,
    }
    r = client.post("/predict", json=payload)
    assert r.status_code == 200
    assert 0.0 <= r.json()["attrition_risk"] <= 1.0


def test_batch_predict(client):
    r = client.post("/predict/batch", json={"employee_ids": [1, 2, 999999]})
    assert r.status_code == 200
    results = r.json()["results"]
    assert len(results) == 3
    assert any(x["attrition_risk"] is None for x in results)
