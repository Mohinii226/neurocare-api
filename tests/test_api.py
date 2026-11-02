from fastapi.testclient import TestClient

from app.main import app

from .helpers import sample_features


def test_health_and_predict(artifact_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is True
        assert len(client.get("/features").json()["feature_columns"]) == 23
        response = client.post("/predict", json={"features": sample_features()})
        assert response.status_code == 200
        assert "predicted_class" in response.json()


def test_predict_rejects_incomplete_input(artifact_dir, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(artifact_dir))
    with TestClient(app) as client:
        response = client.post("/predict", json={"features": {"Age": 30}})
        assert response.status_code == 422


def test_predict_returns_503_without_model(tmp_path, monkeypatch):
    monkeypatch.setenv("ARTIFACT_DIR", str(tmp_path))
    with TestClient(app) as client:
        assert client.get("/health").json()["model_loaded"] is False
        response = client.post("/predict", json={"features": {"Age": 30}})
        assert response.status_code == 503
