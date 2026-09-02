from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Callable, Literal

from app.config import Settings, settings as default_settings
from app.generation.answer import generate_answer
from app.generation.citations import validate_answer_citations
from app.generation.compress import compress_evidence
from app.generation.prompts import ANALYZE_PROMPT, GRADE_PROMPT, REWRITE_PROMPT
from app.graph.sufficiency import evidence_sufficient
from app.models.clients import GenerateResult, LLMClient
from app.errors import RunError, find_run_error
from app.models.routing import ModelRole, ModelRouter, answer_role
from app.models.schemas import EvidenceItem, GradeResult, ResearchState, RetrievedChunk
from app.retrieval.normalize import normalize_query
from app.retrieval.rerank import Reranker
from app.store import append_step

Searcher = Callable[[str, int | None], list[RetrievedChunk]]
Baseline = Literal["vector", "hybrid", "rerank", "agentic"]
MAX_CITATION_ATTEMPTS = 3


@dataclass
class GraphDeps:
    models: ModelRouter
    search: Searcher
    reranker: Reranker
    conn: Any | None = None
    settings: Settings = field(default_factory=lambda: default_settings)
    baseline: Baseline = "agentic"
    max_retrieval_rounds: int | None = None


def _messages(prompt, **kwargs) -> list[dict[str, str]]:
    messages = prompt.format_messages(**kwargs)
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]


def _usage_fields(result: GenerateResult | None) -> dict[str, Any]:
    if result is None:
        return {"prompt_tokens": None, "completion_tokens": None, "total_tokens": None}
    return {
        "prompt_tokens": result.input_tokens,
        "completion_tokens": result.output_tokens,
        "total_tokens": result.total_tokens,
    }


def _model_step(
    deps: GraphDeps,
    state: ResearchState,
    node: str,
    role: ModelRole | None,
    inp: dict,
    out: dict,
    started: float,
    result: GenerateResult | None = None,
) -> None:
    extra: dict[str, Any] = dict(out)
    if role is not None:
        extra["model_role"] = role.value
        extra["model_name"] = deps.models.model_name(role)
        extra.update(_usage_fields(result))
    append_step(
        deps.conn,
        state.get("run_id") or "",
        node,
        inp,
        extra,
        int((time.perf_counter() - started) * 1000),
    )


def _call_json(
    llm: LLMClient,
    messages: list[dict[str, str]],
    *,
    node: str,
    role: ModelRole,
    model_name: str,
) -> tuple[dict, GenerateResult | None]:
    try:
        data = llm.generate_json(messages)
        return data, getattr(llm, "last_result", None)
    except Exception as exc:
        err = find_run_error(exc) or RunError(str(exc))
        err.annotate(node=node, model_role=role.value, model_name=model_name)
        raise err from exc


def _bump_usage(state: ResearchState, n: int = 1) -> dict:
    return {"model_calls": int(state.get("model_calls") or 0) + n}


def analyze_query(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    role = ModelRole.PRO
    llm = deps.models.client(role)
    q = state["original_query"]
    data, result = _call_json(
        llm, _messages(ANALYZE_PROMPT, query=q), node="analyze_query", role=role, model_name=deps.models.model_name(role)
    )
    intent = str(data.get("intent") or "unknown")
    entities = [str(x) for x in (data.get("entities") or [])]
    sub_qs = [str(x) for x in (data.get("sub_questions") or [])]
    out = {
        "normalized_query": normalize_query(q),
        "intent": intent,
        "entities": entities,
        "sub_questions": sub_qs,
        "research_plan": sub_qs,
        "rewritten_query": None,
        "retrieval_round": 0,
        "max_retrieval_rounds": deps.max_retrieval_rounds or deps.settings.max_retrieval_rounds,
        "citation_attempts": 0,
        "generation_invalid": False,
        "evidence_sufficient": False,
        "status": "running",
        **_bump_usage(state),
    }
    _model_step(
        deps, state, "analyze_query", role, {"query": q}, {"intent": intent, "sub_questions": sub_qs}, t0, result
    )
    return out


def retrieve(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    round_n = int(state.get("retrieval_round") or 0) + 1
    query = state.get("rewritten_query") or state.get("normalized_query") or state["original_query"]
    hits = deps.search(query, deps.settings.fusion_top_k)
    payload = [h.model_dump() for h in hits]
    out = {
        "retrieval_round": round_n,
        "retrieved_chunks": payload,
        "evidence_sufficient": False,
    }
    _model_step(
        deps,
        state,
        "retrieve",
        None,
        {"query": query, "round": round_n, "candidate_count": len(payload)},
        {"candidate_count": len(payload)},
        t0,
    )
    return out


def rerank(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    chunks = [RetrievedChunk.model_validate(x) for x in state.get("retrieved_chunks") or []]
    query = state.get("rewritten_query") or state.get("normalized_query") or state["original_query"]
    if deps.baseline == "vector" or deps.baseline == "hybrid":
        ranked = []
        for i, ch in enumerate(chunks[: deps.settings.rerank_top_k]):
            row = ch.model_copy()
            row.rank = i
            ranked.append(row)
    else:
        ranked = deps.reranker.rank(query, chunks, deps.settings.rerank_top_k)
    payload = [h.model_dump() for h in ranked]
    _model_step(deps, state, "rerank", None, {"count": len(chunks)}, {"rerank_count": len(payload)}, t0)
    return {"reranked_chunks": payload}


def grade_evidence(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    role = ModelRole.FAST
    llm = deps.models.client(role)
    chunks = [RetrievedChunk.model_validate(x) for x in state.get("reranked_chunks") or []]
    sub_qs = list(state.get("sub_questions") or [])
    query = state["original_query"]
    grades: list[GradeResult] = []
    calls = 0
    last: GenerateResult | None = None
    in_tok = out_tok = 0
    for ch in chunks:
        data, last = _call_json(
            llm,
            _messages(
                GRADE_PROMPT,
                query=query,
                sub_questions=sub_qs,
                chunk_id=ch.chunk_id,
                chunk=ch.content,
            ),
            node="grade_evidence",
            role=role,
            model_name=deps.models.model_name(role),
        )
        calls += 1
        if last is not None:
            in_tok += int(last.input_tokens or 0)
            out_tok += int(last.output_tokens or 0)
        covers = [str(x) for x in (data.get("covers") or [])]
        raw_level = str(data.get("support_level") or "none")
        level = raw_level if raw_level in {"direct", "partial", "weak", "none"} else "none"
        grades.append(
            GradeResult(
                chunk_id=str(data.get("chunk_id") or ch.chunk_id),
                relevant=bool(data.get("relevant")),
                support_level=level,  # type: ignore[arg-type]
                reason=str(data.get("reason") or ""),
                covers=covers,
            )
        )
    sufficient = evidence_sufficient(sub_qs, grades)
    kept = [g for g in grades if g.relevant and g.support_level in {"direct", "partial"}]
    out = {
        "graded_chunks": [g.model_dump() for g in grades],
        "evidence_chunks": [g.model_dump() for g in kept],
        "evidence_sufficient": sufficient,
        "model_calls": int(state.get("model_calls") or 0) + calls,
    }
    if not sufficient:
        out["failure_reason"] = "insufficient_evidence"
    usage = GenerateResult(
        text="",
        input_tokens=in_tok or None,
        output_tokens=out_tok or None,
        total_tokens=(in_tok + out_tok) or None,
        model=last.model if last else None,
    )
    _model_step(
        deps,
        state,
        "grade_evidence",
        role,
        {"graded_count": len(grades)},
        {"sufficient": sufficient, "kept": len(kept)},
        t0,
        usage,
    )
    return out


def rewrite_query(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    role = ModelRole.FAST
    llm = deps.models.client(role)
    summary = "; ".join(
        (g.get("chunk_id") + ":" + (g.get("support_level") or ""))
        for g in (state.get("graded_chunks") or [])[:8]
    )
    data, result = _call_json(
        llm,
        _messages(
            REWRITE_PROMPT,
            original_query=state["original_query"],
            sub_questions=state.get("sub_questions") or [],
            evidence_summary=summary or "(none)",
            failure_reasons=state.get("failure_reason") or "insufficient_evidence",
        ),
        node="rewrite_query",
        role=role,
        model_name=deps.models.model_name(role),
    )
    q = str(data.get("rewritten_query") or "").strip() or state.get("normalized_query")
    out = {"rewritten_query": q, **_bump_usage(state)}
    _model_step(deps, state, "rewrite_query", role, {"focus": data.get("focus")}, {"rewritten_query": q}, t0, result)
    return out


def compress_node(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    role = ModelRole.FAST
    llm = deps.models.client(role)
    chunks = [RetrievedChunk.model_validate(x) for x in state.get("reranked_chunks") or []]
    grades = [GradeResult.model_validate(x) for x in state.get("graded_chunks") or []]
    try:
        items = compress_evidence(
            llm,
            state["original_query"],
            chunks,
            grades,
            max_items=deps.settings.max_evidence_items,
            max_tokens=deps.settings.max_evidence_tokens,
        )
    except Exception as exc:
        err = find_run_error(exc) or RunError(str(exc))
        err.annotate(node="compress_evidence", model_role=role.value, model_name=deps.models.model_name(role))
        raise err from exc
    text = "\n".join(f"{it.chunk_id}: {it.quote}" for it in items)
    out = {
        "evidence_chunks": [it.model_dump() for it in items],
        "compressed_context": text,
        **_bump_usage(state, len(items)),
    }
    _model_step(
        deps,
        state,
        "compress_evidence",
        role,
        {"evidence_count": len(items)},
        {"evidence_count": len(items)},
        t0,
        getattr(llm, "last_result", None),
    )
    return out


def generate_node(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    attempts = int(state.get("citation_attempts") or 0)
    role = answer_role(attempts)
    llm = deps.models.client(role)
    items = [EvidenceItem.model_validate(x) for x in state.get("evidence_chunks") or []]
    try:
        payload = generate_answer(llm, state["original_query"], items)
    except Exception as exc:
        err = find_run_error(exc) or RunError(str(exc))
        err.annotate(node="generate_answer", model_role=role.value, model_name=deps.models.model_name(role))
        raise err from exc
    out = {
        "draft_answer": str(payload.get("answer") or ""),
        "citations": payload.get("citations") or [],
        "citation_attempts": attempts + 1,
        **_bump_usage(state),
    }
    _model_step(
        deps,
        state,
        "generate_answer",
        role,
        {"evidence_count": len(items), "citation_attempts": attempts},
        {"citation_count": len(out["citations"])},
        t0,
        getattr(llm, "last_result", None),
    )
    return out


def validate_citations_node(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    items = [EvidenceItem.model_validate(x) for x in state.get("evidence_chunks") or []]
    payload = {"answer": state.get("draft_answer"), "citations": state.get("citations") or []}
    ok, reason = validate_answer_citations(payload, items)
    out = {"generation_invalid": not ok}
    if ok:
        out["final_answer"] = state.get("draft_answer")
        out["failure_reason"] = None
        out["status"] = "completed"
    else:
        out["failure_reason"] = reason
        out["status"] = "running"
    _model_step(deps, state, "validate_citations", None, {"ok": ok}, {"reason": reason}, t0)
    return out


ABSTAIN_TEXT = "当前知识库没有足够证据支持该问题。"


def finalize(state: ResearchState, deps: GraphDeps) -> dict:
    t0 = time.perf_counter()
    if state.get("generation_invalid") and int(state.get("citation_attempts") or 0) >= MAX_CITATION_ATTEMPTS:
        out = {
            "status": "failed",
            "failure_reason": state.get("failure_reason") or "generation invalid",
            "final_answer": None,
        }
    elif not state.get("evidence_sufficient"):
        out = {
            "status": "abstained",
            "failure_reason": "insufficient_evidence",
            "final_answer": ABSTAIN_TEXT,
            "citations": [],
        }
    else:
        out = {
            "status": "completed",
            "final_answer": state.get("final_answer") or state.get("draft_answer"),
            "failure_reason": None,
        }
    _model_step(deps, state, "finalize", None, {"status": out["status"]}, out, t0)
    return out


def route_after_grade(state: ResearchState) -> str:
    if state.get("evidence_sufficient"):
        return "compress_evidence"
    max_r = int(state.get("max_retrieval_rounds") or 2)
    round_n = int(state.get("retrieval_round") or 0)
    if round_n < max_r:
        return "rewrite_query"
    return "finalize"


def route_after_validate(state: ResearchState) -> str:
    if not state.get("generation_invalid"):
        return "finalize"
    if int(state.get("citation_attempts") or 0) < MAX_CITATION_ATTEMPTS:
        return "generate_answer"
    return "finalize"
