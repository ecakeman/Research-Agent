from app.models.schemas import RetrievedChunk
from app.retrieval.hybrid import rrf_fuse


def _hit(cid: str, score: float = 0.0) -> RetrievedChunk:
    return RetrievedChunk(chunk_id=cid, document_id="d", score=score, content=cid)


def test_rrf_prefers_items_high_in_both_lists():
    a = [_hit("x"), _hit("y"), _hit("z")]
    b = [_hit("y"), _hit("x"), _hit("w")]
    fused = rrf_fuse([a, b], rrf_k=60, top_k=10)
    ids = [h.chunk_id for h in fused]
    assert ids[0] in {"x", "y"}
    assert set(ids) == {"x", "y", "z", "w"}
    y = next(h for h in fused if h.chunk_id == "y")
    z = next(h for h in fused if h.chunk_id == "z")
    assert y.score > z.score


def test_rrf_formula():
    a = [_hit("only")]
    fused = rrf_fuse([a], rrf_k=60, top_k=5)
    assert abs(fused[0].score - 1 / 61) < 1e-9
