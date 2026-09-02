from __future__ import annotations

import json
from pathlib import Path
from app.config import settings
from app.db import get_conn
from app.ingestion.parser import ParsedDocument, parse_markdown
from app.ingestion.splitter import split_document
from app.models.clients import EmbeddingClient, HTTPEmbeddingClient


def ingest_path(
    raw_dir: str | Path,
    *,
    embedder: EmbeddingClient | None = None,
    database_url: str | None = None,
) -> dict[str, int]:
    raw_dir = Path(raw_dir)
    embedder = embedder or HTTPEmbeddingClient()
    inserted = 0
    skipped = 0
    chunks_n = 0
    files = sorted(
        p for p in raw_dir.rglob("*.md") if p.name.upper() != "SOURCES.MD"
    )
    with get_conn(database_url) as conn:
        for path in files:
            parsed = parse_markdown(
                path.read_text(encoding="utf-8"),
                default_title=path.stem,
                path=str(path.relative_to(raw_dir)),
            )
            if _exists(conn, parsed):
                skipped += 1
                continue
            doc_id = _insert_document(conn, parsed)
            parts = split_document(parsed.content, title=parsed.title)
            texts = [p.content_with_context for p in parts]
            vectors: list[list[float]] = []
            batch = 16
            for i in range(0, len(texts), batch):
                vectors.extend(embedder.embed(texts[i : i + batch]))
            if texts and len(vectors) != len(parts):
                raise RuntimeError("embedding 数量与 chunk 不一致")
            _insert_chunks(conn, doc_id, parts, vectors, settings.embedding_model)
            inserted += 1
            chunks_n += len(parts)
            conn.commit()
    return {"documents_inserted": inserted, "documents_skipped": skipped, "chunks": chunks_n}


def _exists(conn, parsed: ParsedDocument) -> bool:
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT 1 FROM documents
            WHERE source_name = %s AND version = %s AND doc_key = %s
            """,
            (parsed.source_name, parsed.version, parsed.metadata.get("path") or parsed.title),
        )
        return cur.fetchone() is not None


def _insert_document(conn, parsed: ParsedDocument) -> str:
    doc_key = parsed.metadata.get("path") or parsed.title
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents (
                source_type, source_name, title, section, url, version, doc_key,
                content, metadata
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s::jsonb)
            RETURNING id
            """,
            (
                parsed.source_type,
                parsed.source_name,
                parsed.title,
                parsed.section,
                parsed.url,
                parsed.version,
                doc_key,
                parsed.content,
                json.dumps(parsed.metadata),
            ),
        )
        return str(cur.fetchone()["id"])


def _insert_chunks(conn, document_id: str, parts, vectors: list[list[float]], model: str) -> None:
    with conn.cursor() as cur:
        for i, part in enumerate(parts):
            cur.execute(
                """
                INSERT INTO chunks (
                    document_id, chunk_index, parent_section, heading_path,
                    content, content_with_context, token_count, char_count,
                    start_offset, end_offset, chunk_type, metadata, search_tsv
                ) VALUES (
                    %s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s,%s,%s,%s::jsonb,
                    to_tsvector('simple', %s)
                )
                RETURNING id
                """,
                (
                    document_id,
                    part.chunk_index,
                    part.parent_section,
                    json.dumps(part.heading_path),
                    part.content,
                    part.content_with_context,
                    part.token_count,
                    part.char_count,
                    part.start_offset,
                    part.end_offset,
                    part.chunk_type,
                    json.dumps(part.metadata),
                    part.content_with_context,
                ),
            )
            chunk_id = cur.fetchone()["id"]
            if vectors:
                vec = vectors[i]
                literal = "[" + ",".join(str(float(x)) for x in vec) + "]"
                cur.execute(
                    """
                    INSERT INTO chunk_embeddings (chunk_id, embedding, model)
                    VALUES (%s, %s::vector, %s)
                    """,
                    (chunk_id, literal, model or "unknown"),
                )
