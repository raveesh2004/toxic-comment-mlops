"""API tests — the model dependency is stubbed so CI never loads weights."""

from fastapi.testclient import TestClient

from app.main import app, get_classifier


def fake_classifier(texts):
    if isinstance(texts, str):
        texts = [texts]
    return [{"label": "offensive", "score": 0.97} for _ in texts]


app.dependency_overrides[get_classifier] = lambda: fake_classifier
client = TestClient(app)


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_predict():
    response = client.post("/predict", json={"text": "you are terrible"})
    assert response.status_code == 200
    body = response.json()
    assert body["label"] == "offensive"
    assert 0.0 <= body["score"] <= 1.0


def test_predict_batch():
    response = client.post("/predict/batch", json={"texts": ["a", "b", "c"]})
    assert response.status_code == 200
    assert len(response.json()) == 3


def test_predict_rejects_empty_text():
    response = client.post("/predict", json={"text": ""})
    assert response.status_code == 422


def test_predict_rejects_missing_field():
    response = client.post("/predict", json={})
    assert response.status_code == 422
