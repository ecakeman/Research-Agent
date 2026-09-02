from fastapi.testclient import TestClient

from app.main import app


def test_healthz():
    client = TestClient(app)
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json()["ok"] is True


def test_post_research_uses_service(monkeypatch):
    def fake_execute(query, baseline="agentic"):
        return {
            "run_id": "11111111-1111-1111-1111-111111111111",
            "status": "completed",
            "answer": "ok",
            "citations": [{"chunk_id": "c1", "title": "T", "section": "S"}],
        }

    monkeypatch.setattr("app.api.routes.execute_research", fake_execute)
    client = TestClient(app)
    r = client.post("/research", json={"query": "What is LangGraph State?"})
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "completed"
    assert body["answer"] == "ok"
    assert body["citations"][0]["chunk_id"] == "c1"


def test_eval_endpoint_empty_dataset(monkeypatch):
    monkeypatch.setattr("app.evaluation.dataset.load_cases", lambda path=None: [])
    monkeypatch.setattr("app.evaluation.runner.load_cases", lambda path=None: [])
    client = TestClient(app)
    r = client.post("/evaluation/run", json={"baseline": "vector", "live": False})
    assert r.status_code == 200
    assert r.json()["cases"] == 0
