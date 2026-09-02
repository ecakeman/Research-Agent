from __future__ import annotations

from typing import Any

from psycopg.types.json import Json


def create_run(conn, query: str) -> str:
    with conn.cursor() as cur:
        cur.execute(
            "INSERT INTO research_runs (query, status) VALUES (%s, 'running') RETURNING id",
            (query,),
        )
        return str(cur.fetchone()["id"])


def finish_run(
    conn,
    run_id: str,
    *,
    status: str,
    answer: str | None,
    failure_reason: str | None,
    retrieval_rounds: int,
    citations: list[dict],
) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            UPDATE research_runs
            SET status=%s, final_answer=%s, failure_reason=%s,
                retrieval_rounds=%s, citations=%s, completed_at=now()
            WHERE id=%s
            """,
            (status, answer, failure_reason, retrieval_rounds, Json(citations), run_id),
        )


def next_step_index(conn, run_id: str) -> int:
    with conn.cursor() as cur:
        cur.execute("SELECT COALESCE(MAX(step_index), -1) AS m FROM research_steps WHERE run_id=%s", (run_id,))
        return int(cur.fetchone()["m"]) + 1


def append_step(
    conn,
    run_id: str,
    node: str,
    input_data: dict[str, Any],
    output_data: dict[str, Any],
    duration_ms: int | None,
) -> None:
    if conn is None:
        return
    idx = next_step_index(conn, run_id)
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO research_steps (run_id, step_index, node, input, output, duration_ms)
            VALUES (%s,%s,%s,%s,%s,%s)
            """,
            (run_id, idx, node, Json(_public_payload(input_data)), Json(_public_payload(output_data)), duration_ms),
        )


def _public_payload(data: dict[str, Any]) -> dict[str, Any]:
    blocked = {"prompt", "messages", "api_key", "secret"}
    return {k: v for k, v in data.items() if k not in blocked}


def get_run(conn, run_id: str) -> dict | None:
    with conn.cursor() as cur:
        cur.execute("SELECT * FROM research_runs WHERE id=%s", (run_id,))
        row = cur.fetchone()
        return dict(row) if row else None


def list_steps(conn, run_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT step_index, node, duration_ms, created_at
            FROM research_steps WHERE run_id=%s ORDER BY step_index
            """,
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def list_steps_full(conn, run_id: str) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT step_index, node, duration_ms, input, output, created_at
            FROM research_steps WHERE run_id=%s ORDER BY step_index
            """,
            (run_id,),
        )
        return [dict(r) for r in cur.fetchall()]


def list_documents(conn, limit: int = 50) -> list[dict]:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT id, source_type, source_name, title, version, url, created_at
            FROM documents ORDER BY created_at DESC LIMIT %s
            """,
            (limit,),
        )
        return [dict(r) for r in cur.fetchall()]


def insert_evaluation_result(conn, row: dict) -> None:
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO evaluation_results (
                case_id, baseline, retrieval_recall_5, retrieval_recall_10, mrr, ndcg_10,
                groundedness, citation_precision, citation_recall,
                abstention_expected, abstention_actual, retrieval_rounds,
                latency_ms, input_tokens, output_tokens, model_calls, status
            ) VALUES (
                %(case_id)s, %(baseline)s, %(retrieval_recall_5)s, %(retrieval_recall_10)s,
                %(mrr)s, %(ndcg_10)s, %(groundedness)s, %(citation_precision)s,
                %(citation_recall)s, %(abstention_expected)s, %(abstention_actual)s,
                %(retrieval_rounds)s, %(latency_ms)s, %(input_tokens)s, %(output_tokens)s,
                %(model_calls)s, %(status)s
            )
            """,
            row,
        )
