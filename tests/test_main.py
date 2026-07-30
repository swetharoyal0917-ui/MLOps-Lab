import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)


def test_home():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {
        "message": "Advanced Testing Pipeline is running"
    }


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {
        "status": "Healthy"
    }


def test_deploy():
    response = client.get("/deploy")
    assert response.status_code == 200
    assert response.json() == {
        "deployment": "Application deployed successfully"
    }