from __future__ import annotations

from app.graph.sufficiency import grade_covers_subquestion, is_direct_cover
from app.models.schemas import GradeResult, RetrievedChunk

_REASON_MAX = 280
_GAP_MAX = 500
_SNIPPET_MAX = 400


def collect_evidence_gaps(
    sub_questions: list[str],
    grades: list[GradeResult],
) -> tuple[list[str], list[str]]:
    """未 direct 覆盖的 sub-question；gap 文本只拼原文 + 相关 grade.reason。"""
    missing: list[str] = []
    gaps: list[str] = []
    for sq in sub_questions:
        if is_direct_cover(grades, sq):
            continue
        missing.append(sq)
        reasons: list[str] = []
        for g in grades:
            if not grade_covers_subquestion(g, sq):
                continue
            r = (g.reason or "").strip()
            if r:
                reasons.append(r[:_REASON_MAX])
        if reasons:
            text = f"{sq} — {reasons[0]}"
        else:
            text = sq
        gaps.append(text[:_GAP_MAX])
    return missing, gaps


def format_evidence_summary(
    chunks: list[RetrievedChunk],
    grades: list[GradeResult],
) -> str:
    by_id = {g.chunk_id: g for g in grades}
    parts: list[str] = []
    for ch in chunks:
        g = by_id.get(ch.chunk_id)
        section = ch.parent_section or ""
        if not section and ch.heading_path:
            section = " > ".join(ch.heading_path)
        snippet = (ch.content or "")[:_SNIPPET_MAX]
        support = g.support_level if g else "ungraded"
        reason = (g.reason or "") if g else ""
        parts.append(
            f"- chunk_id={ch.chunk_id} source_title={ch.source_title} "
            f"section={section} support_level={support} reason={reason}\n"
            f"  snippet: {snippet}"
        )
    return "\n".join(parts) if parts else "(none)"
