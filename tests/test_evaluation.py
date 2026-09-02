from pathlib import Path

from app.evaluation.dataset import validate_case
from app.evaluation.metrics import citation_precision_recall, mrr, ndcg_at_k, recall_at_k
from app.evaluation.runner import NOT_READY, run_eval


def test_recall_mrr_ndcg():
    relevant = {"a", "b"}
    ranked = ["x", "a", "b"]
    assert recall_at_k(relevant, ranked, 5) == 1.0
    assert recall_at_k(relevant, ranked, 1) == 0.0
    assert abs(mrr(relevant, ranked) - 0.5) < 1e-9
    assert ndcg_at_k(relevant, ranked, 10) > 0


def test_citation_metrics():
    p, r = citation_precision_recall(["e1", "ghost"], {"e1", "e2"}, {"e1"})
    assert abs(p - 0.5) < 1e-9
    assert r == 1.0


def test_runner_empty_does_not_invent_scores(tmp_path: Path, monkeypatch):
    empty = tmp_path / "questions.jsonl"
    empty.write_text("", encoding="utf-8")
    monkeypatch.setattr("app.evaluation.runner.load_cases", lambda: [])
    report = run_eval(baseline="vector", live=False, print_report=False)
    assert report["cases"] == 0
    assert report["error"] == NOT_READY
    assert report["recall_5"] is None


def test_runner_fixture_metrics(monkeypatch):
    cases = [
        {
            "id": "q001",
            "category": "fact",
            "question": "q",
            "expected_sources": ["c1"],
            "ranked_ids": ["c1", "c2"],
        },
        {
            "id": "q002",
            "category": "insufficient-evidence",
            "question": "q2",
            "expected_sources": [],
            "abstention_actual": True,
        },
    ]
    monkeypatch.setattr("app.evaluation.runner.load_cases", lambda: cases)
    report = run_eval(baseline="hybrid", live=False, print_report=False)
    assert report["cases"] == 2
    assert report["recall_5"] == 1.0
    assert report["correct_abstention"] == 1.0
    assert report["rewrite_recovery_rate"] is None


def test_schema_accepts_required_fields_and_alias():
    case = validate_case(
        {
            "id": "q001",
            "category": "insufficient-evidence",
            "question": "q",
            "expected_sources": [],
            "expected_claims": [],
        }
    )
    assert case["category"] == "insufficient"


class _DummyCtx:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False


def test_live_eval_scores_sources_and_abstain(monkeypatch):
    cases = [
        {
            "id": "q001",
            "category": "fact",
            "question": "q",
            "expected_sources": ["langgraph"],
        },
        {
            "id": "q002",
            "category": "insufficient",
            "question": "q2",
            "expected_sources": [],
        },
    ]
    results = [
        {
            "run_id": "00000000-0000-0000-0000-000000000001",
            "status": "completed",
            "citations": [{"chunk_id": "c1", "quote": "State"}],
            "ranked_sources": ["langgraph"],
            "retrieval_rounds": 1,
            "rewritten_query": None,
            "first_pass_evidence_sufficient": True,
        },
        {
            "run_id": "00000000-0000-0000-0000-000000000002",
            "status": "abstained",
            "citations": [],
            "ranked_sources": [],
            "retrieval_rounds": 2,
            "rewritten_query": "other",
            "first_pass_evidence_sufficient": False,
        },
    ]

    by_q = {"q": results[0], "q2": results[1]}

    def fake_exec(q, baseline="agentic"):
        return by_q[q]

    monkeypatch.setattr("app.service.execute_research", fake_exec)
    monkeypatch.setattr("app.evaluation.runner.load_cases", lambda: cases)
    monkeypatch.setattr(
        "app.evaluation.live.list_steps_full",
        lambda conn, run_id: [
            {"duration_ms": 10, "output": {"model_role": "pro", "prompt_tokens": 1, "completion_tokens": 1}}
        ],
    )
    monkeypatch.setattr(
        "app.evaluation.live._chunk_meta",
        lambda conn, ids: {i: {"source_name": "langgraph", "content": "State is shared"} for i in ids},
    )
    monkeypatch.setattr("app.evaluation.live.get_conn", lambda: _DummyCtx())
    report = run_eval(baseline="agentic", live=True, print_report=False, model_routing="dual")
    assert report["live"] is True
    assert report["completed"] == 1
    assert report["abstained"] == 1
    assert report["recall_5"] == 1.0
    assert report["correct_abstention"] == 1.0
    assert report["citation_precision"] == 1.0
    assert report["rewrite_attempted"] == 1
    assert report["eligible_for_recovery"] == 0  # insufficient gold, not answerable


def test_live_citation_not_zeroed_by_false_abstain(monkeypatch):
    cases = [
        {
            "id": "q001",
            "category": "fact",
            "question": "ok",
            "expected_sources": ["langgraph"],
        },
        {
            "id": "q002",
            "category": "fact",
            "question": "miss",
            "expected_sources": ["langgraph"],
        },
    ]
    by_q = {
        "ok": {
            "run_id": "r1",
            "status": "completed",
            "citations": [{"chunk_id": "c1", "quote": "State"}],
            "ranked_sources": ["langgraph"],
            "retrieval_rounds": 1,
            "first_pass_evidence_sufficient": True,
        },
        "miss": {
            "run_id": "r2",
            "status": "abstained",
            "citations": [],
            "ranked_sources": ["langgraph"],
            "retrieval_rounds": 2,
            "rewritten_query": "x",
            "first_pass_evidence_sufficient": False,
        },
    }

    monkeypatch.setattr("app.service.execute_research", lambda q, baseline="agentic": by_q[q])
    monkeypatch.setattr("app.evaluation.runner.load_cases", lambda: cases)
    monkeypatch.setattr("app.evaluation.live.list_steps_full", lambda conn, run_id: [])
    monkeypatch.setattr(
        "app.evaluation.live._chunk_meta",
        lambda conn, ids: {i: {"source_name": "langgraph", "content": "State is shared"} for i in ids},
    )
    monkeypatch.setattr("app.evaluation.live.get_conn", lambda: _DummyCtx())
    report = run_eval(baseline="agentic", live=True, print_report=False, model_routing="dual", concurrency=1)
    assert report["completed"] == 1
    assert report["abstained"] == 1
    assert report["citation_precision"] == 1.0
    assert report["eligible_for_recovery"] == 1
    assert report["rewrite_recovered"] == 0
    assert report["rewrite_recovery_rate"] == 0.0
    assert report["failure_rate"] == 0.0

