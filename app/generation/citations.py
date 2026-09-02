from __future__ import annotations

from app.models.schemas import EvidenceItem


def quote_from_chunk(chunk_text: str, proposed: str) -> str | None:
    if not proposed:
        return None
    if proposed in chunk_text:
        return proposed
    stripped = proposed.strip()
    if stripped and stripped in chunk_text:
        return stripped
    return None


def format_evidence(items: list[EvidenceItem]) -> str:
    lines = []
    for i, item in enumerate(items):
        lines.append(
            f"[{i}] chunk_id={item.chunk_id} title={item.source_title} "
            f"section={item.section or ''} support={item.support_level}\n"
            f"claim: {item.claim}\nquote: {item.quote}\n"
        )
    return "\n".join(lines)


def validate_answer_citations(
    payload: dict,
    evidence: list[EvidenceItem],
) -> tuple[bool, str | None]:
    citations = payload.get("citations") or []
    if not isinstance(citations, list):
        return False, "citations_not_list"
    allowed = {e.chunk_id for e in evidence}
    if not evidence:
        return False, "no_evidence"
    for c in citations:
        if not isinstance(c, dict):
            return False, "citation_not_object"
        cid = str(c.get("chunk_id") or "")
        if cid not in allowed:
            return False, "citation_not_in_evidence"
        if c.get("claim_index") is not None:
            try:
                idx = int(c["claim_index"])
            except (TypeError, ValueError):
                return False, "bad_claim_index"
            if idx < 0:
                return False, "bad_claim_index"
    answer = str(payload.get("answer") or "")
    if not answer.strip():
        return False, "empty_answer"
    return True, None
