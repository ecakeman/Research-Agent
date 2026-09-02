from __future__ import annotations

import json

from app.models.clients import EmbeddingClient, HTTPEmbeddingClient
from app.models.schemas import RetrievedChunk


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


class VectorRetriever:
    def __init__(
        self,
        conn=None,
        embedder: EmbeddingClient | None = None,
        top_k: int = 20,
    ):
        self.conn = conn
        self.embedder = embedder or HTTPEmbeddingClient()
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        if self.conn is None:
            return []
        vec = self.embedder.embed([query])[0]
        literal = "[" + ",".join(str(float(x)) for x in vec) + "]"
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.content, c.content_with_context,
                       c.heading_path, c.parent_section, c.metadata,
                       d.title, d.url, d.source_name,
                       1 - (e.embedding <=> %s::vector) AS score
                FROM chunk_embeddings e
                JOIN chunks c ON c.id = e.chunk_id
                JOIN documents d ON d.id = c.document_id
                ORDER BY e.embedding <=> %s::vector
                LIMIT %s
                """,
                (literal, literal, k),
            )
            rows = cur.fetchall()
        out: list[RetrievedChunk] = []
        for row in rows:
            heading = row["heading_path"] or []
            if isinstance(heading, str):
                heading = json.loads(heading)
            out.append(
                RetrievedChunk(
                    chunk_id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    score=float(row["score"] or 0),
                    content=row["content"],
                    content_with_context=row["content_with_context"],
                    metadata=row["metadata"] or {},
                    heading_path=list(heading),
                    source_title=row["title"],
                    source_name=row.get("source_name") or "",
                    source_url=row["url"],
                    parent_section=row["parent_section"],
                )
            )
        return out
