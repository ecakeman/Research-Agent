from __future__ import annotations

from app.config import settings
from app.db import get_conn
from app.models.clients import HTTPEmbeddingClient, HTTPRerankClient
from app.models.schemas import RetrievedChunk
from app.retrieval.bm25 import KeywordRetriever
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.vector import VectorRetriever


def unique_source_names(hits: list[RetrievedChunk]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for h in hits:
        name = (h.source_name or "").strip()
        if name and name not in seen:
            seen.add(name)
            out.append(name)
    return out


def retrieve_baseline(query: str, baseline: str, conn) -> list[RetrievedChunk]:
    embedder = HTTPEmbeddingClient()
    keyword = KeywordRetriever(conn, settings.bm25_top_k)
    vector = VectorRetriever(conn, embedder, settings.vector_top_k)
    if baseline == "vector":
        return vector.search(query, settings.vector_top_k)
    hybrid = HybridRetriever(keyword, vector)
    hits = hybrid.search(query, settings.fusion_top_k)
    if baseline in {"rerank", "agentic"}:
        client = HTTPRerankClient()
        payload = [h.model_dump() for h in hits]
        ranked = client.rank(query, payload, settings.rerank_top_k)
        by_id = {h.chunk_id: h for h in hits}
        out: list[RetrievedChunk] = []
        for item in ranked:
            base = by_id.get(str(item.get("chunk_id")))
            if base is None:
                continue
            row = base.model_copy()
            row.rerank_score = item.get("rerank_score")
            out.append(row)
        if not out:
            raise RuntimeError("rerank 无有效结果，拒绝编造 retrieval 数字")
        return out
    return hits


def ranked_sources_for_query(query: str, baseline: str) -> list[str]:
    with get_conn() as conn:
        hits = retrieve_baseline(query, baseline, conn)
    return unique_source_names(hits)
