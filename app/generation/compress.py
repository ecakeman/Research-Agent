from __future__ import annotations

from dataclasses import dataclass

from app.generation.citations import quote_from_chunk
from app.generation.prompts import COMPRESS_PROMPT
from app.ingestion.tokens import count_tokens
from app.models.clients import LLMClient
from app.models.schemas import EvidenceItem, GradeResult, RetrievedChunk


@dataclass
class CompressOutcome:
    items: list[EvidenceItem]
    llm_calls: int
    input_tokens: int
    output_tokens: int


def _lc_messages(prompt, **kwargs) -> list[dict[str, str]]:
    messages = prompt.format_messages(**kwargs)
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]


def compress_evidence(
    llm: LLMClient | None,
    query: str,
    chunks: list[RetrievedChunk],
    grades: list[GradeResult],
    *,
    max_items: int,
    max_tokens: int,
) -> CompressOutcome:
    grade_by_id = {g.chunk_id: g for g in grades}
    items: list[EvidenceItem] = []
    used = 0
    llm_calls = 0
    in_tok = out_tok = 0
    for ch in chunks:
        g = grade_by_id.get(ch.chunk_id)
        if g is None or not (g.relevant and g.support_level in {"direct", "partial"}):
            continue
        claim, quote, n_call, tin, tout = _extract(llm, query, ch)
        llm_calls += n_call
        in_tok += tin
        out_tok += tout
        if not quote:
            continue
        item = EvidenceItem(
            chunk_id=ch.chunk_id,
            document_id=ch.document_id,
            claim=claim,
            quote=quote,
            source_title=ch.source_title,
            source_url=ch.source_url,
            section=ch.parent_section or (" > ".join(ch.heading_path) if ch.heading_path else None),
            support_level=g.support_level,
            retrieval_score=ch.score,
            rerank_score=ch.rerank_score,
        )
        cost = count_tokens(item.quote) + count_tokens(item.claim)
        if used + cost > max_tokens and items:
            break
        items.append(item)
        used += cost
        if len(items) >= max_items:
            break
    return CompressOutcome(items, llm_calls, in_tok, out_tok)


def _extract(llm: LLMClient | None, query: str, ch: RetrievedChunk) -> tuple[str, str, int, int, int]:
    text = ch.content
    if llm is None:
        quote = text.strip().split("\n")[0][:400]
        if quote not in text:
            quote = text[: min(200, len(text))]
        return quote, quote, 0, 0, 0
    data = llm.generate_json(_lc_messages(COMPRESS_PROMPT, query=query, chunk=text))
    last = getattr(llm, "last_result", None)
    tin = int(getattr(last, "input_tokens", None) or 0) if last else 0
    tout = int(getattr(last, "output_tokens", None) or 0) if last else 0
    quote = quote_from_chunk(text, str(data.get("quote") or ""))
    if not quote:
        quote = text.strip()[:400]
        if quote not in text:
            quote = text[: min(200, len(text))]
    claim = str(data.get("claim") or quote)[:500]
    return claim, quote, 1, tin, tout


def format_evidence_for_prompt(items: list[EvidenceItem]) -> str:
    from app.generation.citations import format_evidence

    return format_evidence(items)
