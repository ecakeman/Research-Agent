from __future__ import annotations

import math
import re
import unicodedata
from collections import Counter

from app.models.schemas import RetrievedChunk

BM25_K1 = 1.2
BM25_B = 0.75
_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]")


def tokenize(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower()
    return _TOKEN_RE.findall(text)


def bm25_rank(docs: list[RetrievedChunk], query: str) -> list[RetrievedChunk]:
    qtoks = list(dict.fromkeys(tokenize(query)))
    if not docs or not qtoks:
        return []
    bags: list[tuple[RetrievedChunk, Counter[str], int]] = []
    df: Counter[str] = Counter()
    total_len = 0
    for doc in docs:
        toks = tokenize(doc.content_with_context or doc.content)
        tf: Counter[str] = Counter(toks)
        bags.append((doc, tf, len(toks)))
        total_len += len(toks)
        df.update(tf.keys())
    n = len(bags)
    avgdl = total_len / n if n else 0.0
    scored: list[RetrievedChunk] = []
    for doc, tf, dl in bags:
        score = 0.0
        for tok in qtoks:
            if tf[tok] == 0:
                continue
            n_q = df[tok]
            idf = math.log(1 + (n - n_q + 0.5) / (n_q + 0.5))
            denom = tf[tok] + BM25_K1 * (1 - BM25_B + BM25_B * (dl / avgdl if avgdl else 0))
            score += idf * (tf[tok] * (BM25_K1 + 1)) / denom
        row = doc.model_copy()
        row.score = score
        scored.append(row)
    scored.sort(key=lambda x: x.score, reverse=True)
    return scored


class KeywordRetriever:
    def __init__(self, conn=None, top_k: int = 20):
        self.conn = conn
        self.top_k = top_k

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        if self.conn is None:
            return []
        candidates = self._fts(query, max(k * 3, 40))
        ranked = bm25_rank(candidates, query)
        return ranked[:k]

    def _fts(self, query: str, limit: int) -> list[RetrievedChunk]:
        tokens = tokenize(query)
        tsquery = " | ".join(t for t in tokens if t)
        if not tsquery:
            return []
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.id, c.document_id, c.content, c.content_with_context,
                       c.heading_path, c.parent_section, c.metadata,
                       d.title, d.url, d.source_name,
                       ts_rank(c.search_tsv, to_tsquery('simple', %s)) AS fts_score
                FROM chunks c
                JOIN documents d ON d.id = c.document_id
                WHERE c.search_tsv @@ to_tsquery('simple', %s)
                ORDER BY fts_score DESC
                LIMIT %s
                """,
                (tsquery, tsquery, limit),
            )
            rows = cur.fetchall()
        out: list[RetrievedChunk] = []
        for row in rows:
            heading = row["heading_path"] or []
            if isinstance(heading, str):
                import json

                heading = json.loads(heading)
            out.append(
                RetrievedChunk(
                    chunk_id=str(row["id"]),
                    document_id=str(row["document_id"]),
                    score=float(row["fts_score"] or 0),
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
