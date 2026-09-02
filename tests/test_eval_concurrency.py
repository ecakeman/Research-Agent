from __future__ import annotations

import threading
import time

from app.evaluation.concurrent import resolve_concurrency, run_cases
from app.evaluation.runner import run_eval


def test_t1_t5_semaphore_caps_active():
    lock = threading.Lock()
    active = 0
    max_active = 0

    def worker(case: dict) -> dict:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
        time.sleep(0.04)
        with lock:
            active -= 1
        return {"id": case["id"], "status": "ok"}

    cases = [{"id": f"q{i:03d}"} for i in range(1, 9)]
    rows = run_cases(cases, worker, concurrency=4)
    assert [r["id"] for r in rows] == [c["id"] for c in cases]
    assert max_active <= 4
    assert max_active >= 2

    max_active = 0
    active = 0
    run_cases(cases[:3], worker, concurrency=1)
    assert max_active == 1


def test_t2_result_order_despite_finish_time():
    def worker(case: dict) -> dict:
        delay = {"q001": 0.06, "q002": 0.03, "q003": 0.0}[case["id"]]
        time.sleep(delay)
        return {"id": case["id"], "status": "ok", "finished": time.time()}

    cases = [{"id": "q001"}, {"id": "q002"}, {"id": "q003"}]
    rows = run_cases(cases, worker, concurrency=3)
    assert [r["id"] for r in rows] == ["q001", "q002", "q003"]
    assert rows[2]["finished"] < rows[0]["finished"]


def test_t3_single_case_pipeline_stays_serial():
    def worker(case: dict) -> dict:
        order: list[str] = []
        order.append("Analyze")
        order.append("Retrieve")
        order.append("Grade")
        order.append("Answer")
        return {"id": case["id"], "status": "completed", "order": order}

    rows = run_cases([{"id": "q001"}, {"id": "q002"}], worker, concurrency=2)
    for row in rows:
        assert row["order"] == ["Analyze", "Retrieve", "Grade", "Answer"]
        assert row["order"].index("Answer") > row["order"].index("Grade")


def test_t4_one_failure_does_not_abort_batch(monkeypatch):
    cases = [
        {"id": "q001", "category": "fact", "question": "q1", "expected_sources": ["a"], "ranked_ids": ["a"]},
        {"id": "q002", "category": "fact", "question": "q2", "expected_sources": ["a"], "ranked_ids": ["a"]},
        {"id": "q003", "category": "fact", "question": "q3", "expected_sources": ["a"], "ranked_ids": ["a"]},
    ]

    import app.evaluation.runner as runner_mod

    orig = runner_mod._retrieval_one

    def boom(case, baseline):
        if case["id"] == "q002":
            raise RuntimeError("boom")
        return orig(case, baseline)

    monkeypatch.setattr("app.evaluation.runner.load_cases", lambda: cases)
    monkeypatch.setattr(runner_mod, "_retrieval_one", boom)
    report = run_eval(baseline="hybrid", live=False, print_report=False, concurrency=3)
    assert report["failed"] == 1
    assert report["completed"] == 2
    assert report["recall_5"] == 1.0
    ids = [r["id"] for r in report["case_results"]]
    assert ids == ["q001", "q002", "q003"]
    assert report["case_results"][1]["status"] == "failed"


def test_resolve_concurrency_cli_overrides_default():
    assert resolve_concurrency(1) == 1
    assert resolve_concurrency(16) == 16
    assert resolve_concurrency(None) >= 1


def test_failed_record_structured_and_redacted():
    from app.errors import RunError, redact
    from app.evaluation.concurrent import failed_record

    err = RunError(
        "bad json",
        run_id="run-1",
        node="compress_evidence",
        model_role="fast",
        model_name="deepseek-v4-flash",
        raw_excerpt="Bearer sk-secretvaluehere {",
        parse_error="Expecting property name",
    )
    rec = failed_record({"id": "q039"}, err)
    assert rec["question_id"] == "q039"
    assert rec["run_id"] == "run-1"
    assert rec["node"] == "compress_evidence"
    assert rec["model_role"] == "fast"
    assert rec["parse_error"] == "Expecting property name"
    assert "sk-secret" not in (rec.get("raw_excerpt") or "")
    assert "[REDACTED]" in redact("Authorization: Bearer sk-abc.def")
