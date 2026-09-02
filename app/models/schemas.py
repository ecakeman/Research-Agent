from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, TypedDict

from pydantic import BaseModel, Field

SourceType = Literal["official_docs", "github", "paper", "api_docs", "model_docs"]
ChunkType = Literal["text", "code", "table", "list", "quote", "mixed"]
Intent = Literal["fact", "comparison", "explanation", "multi_hop", "how_to", "unknown"]
SupportLevel = Literal["direct", "partial", "weak", "none"]
RunStatus = Literal["running", "completed", "abstained", "failed"]


class DocumentRecord(BaseModel):
    id: str
    source_type: str
    source_name: str
    title: str
    section: str | None = None
    url: str | None = None
    version: str | None = None
    published_at: datetime | None = None
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None
    updated_at: datetime | None = None


class ChunkRecord(BaseModel):
    id: str | None = None
    document_id: str | None = None
    chunk_index: int
    parent_section: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    content: str
    content_with_context: str
    token_count: int
    char_count: int
    start_offset: int
    end_offset: int
    chunk_type: str
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievedChunk(BaseModel):
    chunk_id: str
    document_id: str
    score: float
    content: str
    content_with_context: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    heading_path: list[str] = Field(default_factory=list)
    source_title: str = ""
    source_name: str = ""
    source_url: str | None = None
    parent_section: str | None = None
    rerank_score: float | None = None
    rank: int | None = None


class GradeResult(BaseModel):
    chunk_id: str
    relevant: bool
    support_level: SupportLevel
    reason: str = ""
    covers: list[str] = Field(default_factory=list)


class EvidenceItem(BaseModel):
    chunk_id: str
    document_id: str
    claim: str
    quote: str
    source_title: str
    source_url: str | None = None
    section: str | None = None
    support_level: str
    retrieval_score: float = 0.0
    rerank_score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    claim_index: int = 0
    title: str | None = None
    section: str | None = None


class EvaluationResult(BaseModel):
    case_id: str
    retrieval_recall_5: float | None = None
    retrieval_recall_10: float | None = None
    mrr: float | None = None
    ndcg_10: float | None = None
    groundedness: float | None = None
    citation_precision: float | None = None
    citation_recall: float | None = None
    abstention_expected: bool | None = None
    abstention_actual: bool | None = None
    retrieval_rounds: int | None = None
    latency_ms: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    model_calls: int | None = None
    status: str
    created_at: datetime | None = None


class ResearchState(TypedDict, total=False):
    run_id: str
    original_query: str
    normalized_query: str
    rewritten_query: str | None
    research_plan: list[str]
    intent: str
    entities: list[str]
    sub_questions: list[str]
    retrieval_round: int
    max_retrieval_rounds: int
    retrieved_chunks: list[dict]
    reranked_chunks: list[dict]
    graded_chunks: list[dict]
    evidence_chunks: list[dict]
    evidence_sufficient: bool
    compressed_context: str | None
    draft_answer: str | None
    final_answer: str | None
    citations: list[dict]
    failure_reason: str | None
    token_budget: int
    citation_attempts: int
    generation_invalid: bool
    model_calls: int
    input_tokens: int
    output_tokens: int
    status: str
