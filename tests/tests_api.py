from fastapi.testclient import TestClient

from app import app

client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_prometheus_metrics_exposed():
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "http_requests_total" in response.text
