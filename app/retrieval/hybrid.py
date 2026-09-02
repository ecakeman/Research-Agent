from __future__ import annotations

from collections import defaultdict

from app.config import settings
from app.models.schemas import RetrievedChunk
from app.retrieval.bm25 import KeywordRetriever
from app.retrieval.vector import VectorRetriever


def rrf_fuse(
    ranked_lists: list[list[RetrievedChunk]],
    *,
    rrf_k: int = 60,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, RetrievedChunk] = {}
    for ranking in ranked_lists:
        for rank, item in enumerate(ranking, start=1):
            scores[item.chunk_id] += 1.0 / (rrf_k + rank)
            prev = best.get(item.chunk_id)
            if prev is None or item.score > prev.score:
                best[item.chunk_id] = item
    fused = []
    for chunk_id, score in scores.items():
        row = best[chunk_id].model_copy()
        row.score = score
        fused.append(row)
    fused.sort(key=lambda x: x.score, reverse=True)
    return fused[:top_k]


class HybridRetriever:
    def __init__(
        self,
        keyword: KeywordRetriever,
        vector: VectorRetriever,
        *,
        bm25_top_k: int | None = None,
        vector_top_k: int | None = None,
        rrf_k: int | None = None,
        fusion_top_k: int | None = None,
    ):
        self.keyword = keyword
        self.vector = vector
        self.bm25_top_k = bm25_top_k or settings.bm25_top_k
        self.vector_top_k = vector_top_k or settings.vector_top_k
        self.rrf_k = rrf_k or settings.rrf_k
        self.fusion_top_k = fusion_top_k or settings.fusion_top_k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.fusion_top_k
        bm25_hits = self.keyword.search(query, self.bm25_top_k)
        vec_hits = self.vector.search(query, self.vector_top_k)
        return rrf_fuse([bm25_hits, vec_hits], rrf_k=self.rrf_k, top_k=k)
