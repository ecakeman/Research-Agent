from __future__ import annotations

from app.db import get_conn
from app.evaluation.metrics import citation_precision_recall, mrr, ndcg_at_k, recall_at_k
from app.store import list_steps_full


def _chunk_meta(conn, ids: list[str]) -> dict[str, dict]:
    if not ids:
        return {}
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT c.id::text AS id, d.source_name, c.content
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE c.id::text = ANY(%s)
            """,
            (ids,),
        )
        return {str(r["id"]): dict(r) for r in cur.fetchall()}


def _step_totals(steps: list[dict]) -> dict:
    latency = 0
    prompt = completion = 0
    pro = fast = 0
    for s in steps:
        latency += int(s.get("duration_ms") or 0)
        out = s.get("output") or {}
        if not isinstance(out, dict):
            continue
        prompt += int(out.get("prompt_tokens") or 0)
        completion += int(out.get("completion_tokens") or 0)
        role = out.get("model_role")
        if role:
            n = int(out["llm_calls"]) if "llm_calls" in out else 1
            if role == "pro":
                pro += n
            elif role == "fast":
                fast += n
    return {
        "latency_ms": latency,
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": prompt + completion,
        "pro_calls": pro,
        "fast_calls": fast,
    }


def _groundedness(citations: list, meta: dict[str, dict]) -> float | None:
    if not citations:
        return None
    ok = 0
    n = 0
    for c in citations:
        if not isinstance(c, dict):
            continue
        n += 1
        cid = str(c.get("chunk_id") or "")
        quote = str(c.get("quote") or "")
        row = meta.get(cid) or {}
        text = str(row.get("content") or "")
        if not cid or cid not in meta:
            continue
        if quote:
            if quote in text:
                ok += 1
        else:
            ok += 1
    return ok / n if n else None


def score_live_case(case: dict, result: dict) -> dict:
    expected_abs = case.get("category") in {"insufficient", "insufficient-evidence"}
    status = str(result.get("status") or "")
    actual_abs = status == "abstained"
    relevant = {str(x) for x in (case.get("expected_sources") or []) if str(x).strip()}
    ranked = [str(x) for x in (result.get("ranked_sources") or [])]
    r5 = r10 = mrr_v = ndcg = None
    if relevant:
        r5 = recall_at_k(relevant, ranked, 5)
        r10 = recall_at_k(relevant, ranked, 10)
        mrr_v = mrr(relevant, ranked)
        ndcg = ndcg_at_k(relevant, ranked, 10)

    cites = [c for c in (result.get("citations") or []) if isinstance(c, dict)]
    ids = [str(c.get("chunk_id") or "") for c in cites if c.get("chunk_id")]
    with get_conn() as conn:
        steps = list_steps_full(conn, str(result.get("run_id") or ""))
        meta = _chunk_meta(conn, ids)
    cited_sources = []
    seen: set[str] = set()
    for cid in ids:
        name = str((meta.get(cid) or {}).get("source_name") or "").strip()
        if name and name not in seen:
            seen.add(name)
            cited_sources.append(name)
    rounds = int(result.get("retrieval_rounds") or 0)
    gold_answerable = not expected_abs
    rewrite_attempted = rounds >= 2 or bool(result.get("rewritten_query"))
    fps = result.get("first_pass_evidence_sufficient")
    if fps is None:
        first_pass_insuf = None
        eligible = None
        recovered = None
    else:
        first_pass_insuf = fps is False
        eligible = bool(first_pass_insuf and gold_answerable)
        recovered = bool(first_pass_insuf and gold_answerable and status == "completed")
    stats = _step_totals(steps)
    g = cp = cr = None
    if status == "completed":
        g = _groundedness(cites, meta)
        gold = relevant if relevant else None
        if gold is None:
            cp, cr = citation_precision_recall(cited_sources, set(cited_sources), None)
        else:
            cp, cr = citation_precision_recall(cited_sources, set(cited_sources), gold)
    return {
        "recall_5": r5,
        "recall_10": r10,
        "mrr": mrr_v,
        "ndcg_10": ndcg,
        "groundedness": g,
        "citation_precision": cp,
        "citation_recall": cr,
        "abstention_expected": expected_abs,
        "abstention_actual": actual_abs,
        "gold_answerable": gold_answerable,
        "first_pass_insufficient": first_pass_insuf,
        "eligible_for_recovery": eligible,
        "rewrite_attempted": rewrite_attempted,
        "rewrite_recovered": recovered,
        "status": status,
        "run_id": result.get("run_id"),
        **stats,
    }
