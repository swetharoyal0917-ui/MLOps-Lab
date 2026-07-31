from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200


def test_health():
    response = client.get("/health")
    assert response.status_code == 200


def test_automation():
    response = client.get("/automation")
    assert response.status_code == 200
    assert response.json() == {
        "automation": "Automation completed successfully"
    }