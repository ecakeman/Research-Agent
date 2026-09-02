from __future__ import annotations

from uuid import UUID

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from app.db import get_conn
from app.ingestion.pipeline import ingest_path
from app.service import execute_research
from app.store import get_run, list_documents, list_steps

router = APIRouter()


class ResearchRequest(BaseModel):
    query: str = Field(min_length=1)
    baseline: str = "agentic"


class DocumentIn(BaseModel):
    title: str
    content: str
    source_type: str = "official_docs"
    source_name: str = "upload"
    version: str = "0.1"
    url: str | None = None


class EvalRunRequest(BaseModel):
    baseline: str = "agentic"
    live: bool = False


@router.post("/research")
def post_research(body: ResearchRequest):
    result = execute_research(body.query, baseline=body.baseline)
    citations = []
    for c in result.get("citations") or []:
        if isinstance(c, dict):
            citations.append(
                {
                    "chunk_id": c.get("chunk_id"),
                    "title": c.get("title"),
                    "section": c.get("section"),
                }
            )
    return {
        "run_id": result["run_id"],
        "status": result["status"],
        "answer": result["answer"],
        "citations": citations,
    }


@router.get("/research/{run_id}")
def get_research(run_id: UUID):
    with get_conn() as conn:
        row = get_run(conn, str(run_id))
    if not row:
        raise HTTPException(404, "run not found")
    return {
        "id": str(row["id"]),
        "query": row["query"],
        "status": row["status"],
        "retrieval_rounds": row["retrieval_rounds"],
        "answer": row["final_answer"],
        "citations": row.get("citations") or [],
        "created_at": row["created_at"],
        "completed_at": row["completed_at"],
        "failure_reason": row.get("failure_reason"),
    }


@router.get("/research/{run_id}/steps")
def get_steps(run_id: UUID):
    with get_conn() as conn:
        if not get_run(conn, str(run_id)):
            raise HTTPException(404, "run not found")
        steps = list_steps(conn, str(run_id))
    return {
        "run_id": str(run_id),
        "steps": [
            {
                "step_index": s["step_index"],
                "node": s["node"],
                "duration_ms": s["duration_ms"],
            }
            for s in steps
        ],
    }


@router.get("/documents")
def get_documents():
    with get_conn() as conn:
        rows = list_documents(conn)
    for r in rows:
        r["id"] = str(r["id"])
    return {"documents": rows}


@router.post("/documents")
def post_document(body: DocumentIn):
    from pathlib import Path
    import tempfile

    md = (
        f"---\ntitle: {body.title}\nsource_type: {body.source_type}\n"
        f"source_name: {body.source_name}\nversion: {body.version}\n"
        f"url: {body.url or ''}\n---\n\n{body.content}\n"
    )
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "upload.md"
        path.write_text(md, encoding="utf-8")
        stats = ingest_path(td)
    return stats


@router.post("/evaluation/run")
def post_eval(body: EvalRunRequest):
    from app.evaluation.runner import run_eval

    return run_eval(baseline=body.baseline, live=body.live)
