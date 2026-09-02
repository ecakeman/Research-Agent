from __future__ import annotations

from app.models.schemas import GradeResult


def cover_matches_subquestion(sub_question: str, cover: str) -> bool:
    key = sub_question.strip().lower()
    ck = cover.strip().lower()
    if not key or not ck:
        return False
    return key == ck or key in ck or ck in key


def grade_covers_subquestion(grade: GradeResult, sub_question: str) -> bool:
    return any(cover_matches_subquestion(sub_question, c) for c in grade.covers)


def is_direct_cover(grades: list[GradeResult], sub_question: str) -> bool:
    for g in grades:
        if g.relevant and g.support_level == "direct" and grade_covers_subquestion(g, sub_question):
            return True
    return False


def evidence_sufficient(sub_questions: list[str], grades: list[GradeResult]) -> bool:
    """必须覆盖主要 sub_questions：direct 覆盖率 >= 2/3。"""
    directs = [g for g in grades if g.relevant and g.support_level == "direct"]
    if not sub_questions:
        return len(directs) >= 2
    covered = sum(1 for sq in sub_questions if is_direct_cover(grades, sq))
    return (covered / len(sub_questions)) >= (2 / 3)
