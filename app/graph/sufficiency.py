from __future__ import annotations

from app.models.schemas import GradeResult


def evidence_sufficient(sub_questions: list[str], grades: list[GradeResult]) -> bool:
    """必须覆盖主要 sub_questions：direct 覆盖率 >= 2/3。"""
    directs = [g for g in grades if g.relevant and g.support_level == "direct"]
    if not sub_questions:
        return len(directs) >= 2
    covered = 0
    for sq in sub_questions:
        key = sq.strip().lower()
        hit = False
        for g in directs:
            for c in g.covers:
                ck = c.strip().lower()
                if key == ck or key in ck or ck in key:
                    hit = True
                    break
            if hit:
                break
        if hit:
            covered += 1
    return (covered / len(sub_questions)) >= (2 / 3)
