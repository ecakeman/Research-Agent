from __future__ import annotations

from app.db import get_conn
from app.graph.nodes import GraphDeps
from app.graph.workflow import run_research
from app.models.clients import HTTPEmbeddingClient, HTTPRerankClient
from app.errors import RunError, find_run_error
from app.models.routing import ModelRouter
from app.retrieval.bm25 import KeywordRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.rerank import Reranker
from app.retrieval.vector import VectorRetriever
from app.store import create_run, finish_run, list_steps_full


def execute_research(query: str, *, baseline: str = "agentic", conn=None) -> dict:
    router = ModelRouter.from_settings()
    embedder = HTTPEmbeddingClient()
    rerank_client = HTTPRerankClient() if baseline in {"rerank", "agentic"} else None

    def _go(c):
        run_id = create_run(c, query)
        keyword = KeywordRetriever(c)
        vector = VectorRetriever(c, embedder)

        def search(q: str, top_k: int | None = None):
            if baseline == "vector":
                return vector.search(q, top_k)
            return HybridRetriever(keyword, vector).search(q, top_k)

        deps = GraphDeps(
            models=router,
            search=search,
            reranker=Reranker(rerank_client),
            conn=c,
            baseline=baseline,  # type: ignore[arg-type]
        )
        try:
            state = run_research(deps, query, run_id)
        except Exception as exc:
            err = find_run_error(exc) or RunError(str(exc)[:800])
            err.annotate(run_id=run_id)
            if not err.node:
                try:
                    steps = list_steps_full(c, run_id)
                    if steps:
                        last = steps[-1]
                        out = last.get("output") or {}
                        err.annotate(
                            node=str(last.get("node") or ""),
                            model_role=out.get("model_role"),
                            model_name=out.get("model_name"),
                        )
                except Exception:
                    pass
            finish_run(
                c,
                run_id,
                status="failed",
                answer=None,
                failure_reason=err.parse_error or str(err)[:200],
                retrieval_rounds=0,
                citations=[],
            )
            raise err from exc
        citations = list(state.get("citations") or [])
        finish_run(
            c,
            run_id,
            status=str(state.get("status") or "failed"),
            answer=state.get("final_answer"),
            failure_reason=state.get("failure_reason"),
            retrieval_rounds=int(state.get("retrieval_round") or 0),
            citations=citations,
        )
        ranked = []
        for raw in state.get("reranked_chunks") or state.get("retrieved_chunks") or []:
            name = ""
            if isinstance(raw, dict):
                name = str(raw.get("source_name") or "").strip()
            if name and name not in ranked:
                ranked.append(name)
        return {
            "run_id": run_id,
            "status": state.get("status"),
            "answer": state.get("final_answer"),
            "citations": citations,
            "retrieval_rounds": state.get("retrieval_round"),
            "failure_reason": state.get("failure_reason"),
            "ranked_sources": ranked,
            "citation_attempts": state.get("citation_attempts"),
            "evidence_sufficient": state.get("evidence_sufficient"),
            "first_pass_evidence_sufficient": state.get("first_pass_evidence_sufficient"),
            "rewritten_query": state.get("rewritten_query"),
        }

    if conn is not None:
        return _go(conn)
    with get_conn() as c:
        return _go(c)
