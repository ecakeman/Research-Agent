from __future__ import annotations

from app.config import settings
from app.models.clients import RerankClient
from app.models.schemas import RetrievedChunk


class Reranker:
    def __init__(self, client: RerankClient | None = None, top_k: int | None = None):
        self.client = client
        self.top_k = top_k or settings.rerank_top_k

    def rank(self, query: str, chunks: list[RetrievedChunk], top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        if not chunks:
            return []
        if self.client is None:
            out = []
            for i, ch in enumerate(chunks[:k]):
                row = ch.model_copy()
                row.rank = i
                row.rerank_score = ch.score
                out.append(row)
            return out
        payload = [ch.model_dump() for ch in chunks]
        try:
            ranked = self.client.rank(query, payload, k)
        except Exception:
            ranked = []
            for i, ch in enumerate(chunks[:k]):
                row = ch.model_copy()
                row.rank = i
                row.rerank_score = ch.score
                ranked.append(row)
            return ranked
        by_id = {ch.chunk_id: ch for ch in chunks}
        out: list[RetrievedChunk] = []
        for i, item in enumerate(ranked[:k]):
            base = by_id.get(str(item.get("chunk_id")))
            if base is None:
                continue
            row = base.model_copy()
            row.rerank_score = float(item.get("rerank_score") or 0)
            row.rank = i
            row.score = row.rerank_score
            out.append(row)
        return out
