from __future__ import annotations

from app.graph.sufficiency import grade_covers_subquestion, is_direct_cover
from app.models.schemas import GradeResult, RetrievedChunk

_REASON_MAX = 280
_GAP_MAX = 500
_SNIPPET_MAX = 400
REWRITE_EVIDENCE_MAX = 8
_SUPPORT_RANK = {"direct": 0, "partial": 1, "weak": 2}


def _section(ch: RetrievedChunk) -> str:
    if ch.parent_section:
        return ch.parent_section
    if ch.heading_path:
        return " > ".join(ch.heading_path)
    return ""


def collect_evidence_gaps(
    sub_questions: list[str],
    grades: list[GradeResult],
    chunks: list[RetrievedChunk] | None = None,
) -> tuple[list[str], list[str]]:
    """未 direct 覆盖的 sub-question；gap 只拼原文、grade.reason、已有 chunk 字段。"""
    by_id = {ch.chunk_id: ch for ch in (chunks or [])}
    missing: list[str] = []
    gaps: list[str] = []
    for sq in sub_questions:
        if is_direct_cover(grades, sq):
            continue
        missing.append(sq)
        reason_txt = ""
        observed = ""
        for g in grades:
            if not grade_covers_subquestion(g, sq):
                continue
            r = (g.reason or "").strip()
            if r and not reason_txt:
                reason_txt = r[:_REASON_MAX]
            ch = by_id.get(g.chunk_id)
            if ch is not None and not observed:
                bits = []
                if ch.source_title:
                    bits.append(ch.source_title)
                sec = _section(ch)
                if sec:
                    bits.append(sec)
                snip = (ch.content or "").strip()[:120]
                if snip:
                    bits.append(snip)
                observed = " | ".join(bits)
        parts = [sq]
        if reason_txt:
            parts.append(reason_txt)
        if observed:
            parts.append(observed)
        if len(parts) == 1:
            gaps.append(sq[:_GAP_MAX])
        else:
            gaps.append(" — ".join(parts)[:_GAP_MAX])
    return missing, gaps


def format_evidence_summary(
    chunks: list[RetrievedChunk],
    grades: list[GradeResult],
    *,
    limit: int = REWRITE_EVIDENCE_MAX,
) -> str:
    by_id = {g.chunk_id: g for g in grades}
    ranked: list[tuple[int, RetrievedChunk, GradeResult]] = []
    for ch in chunks:
        g = by_id.get(ch.chunk_id)
        if g is None or g.support_level not in _SUPPORT_RANK:
            continue
        ranked.append((_SUPPORT_RANK[g.support_level], ch, g))
    ranked.sort(key=lambda x: x[0])
    parts: list[str] = []
    for _, ch, g in ranked[: max(0, limit)]:
        snippet = (ch.content or "")[:_SNIPPET_MAX]
        parts.append(
            f"- chunk_id={ch.chunk_id} source_title={ch.source_title} "
            f"section={_section(ch)} support_level={g.support_level} reason={g.reason or ''}\n"
            f"  snippet: {snippet}"
        )
    return "\n".join(parts) if parts else "(none)"
