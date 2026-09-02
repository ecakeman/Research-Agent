from app.graph.sufficiency import evidence_sufficient
from app.models.schemas import GradeResult


def _g(cid: str, level: str, covers: list[str], relevant: bool = True) -> GradeResult:
    return GradeResult(chunk_id=cid, relevant=relevant, support_level=level, covers=covers)


def test_two_thirds_direct_coverage():
    subs = ["state", "control flow", "checkpointing"]
    grades = [
        _g("1", "direct", ["state"]),
        _g("2", "direct", ["control flow"]),
        _g("3", "none", [], relevant=False),
    ]
    assert evidence_sufficient(subs, grades) is True


def test_insufficient_when_only_one_of_three():
    subs = ["state", "control flow", "checkpointing"]
    grades = [_g("1", "direct", ["state"]), _g("2", "partial", ["control flow"])]
    assert evidence_sufficient(subs, grades) is False


def test_partial_does_not_count_as_direct_cover():
    subs = ["a", "b", "c"]
    grades = [
        _g("1", "partial", ["a"]),
        _g("2", "partial", ["b"]),
        _g("3", "partial", ["c"]),
    ]
    assert evidence_sufficient(subs, grades) is False


def test_empty_subquestions_requires_two_direct():
    assert evidence_sufficient([], [_g("1", "direct", [])]) is False
    assert evidence_sufficient([], [_g("1", "direct", []), _g("2", "direct", [])]) is True
