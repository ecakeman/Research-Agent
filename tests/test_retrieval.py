from app.models.schemas import RetrievedChunk
from app.retrieval.bm25 import bm25_rank
from app.retrieval.hybrid import HybridRetriever
from app.retrieval.lc_retriever import LangChainHybridRetriever
from app.retrieval.rerank import Reranker
from tests.fakes import FakeReranker


def _doc(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d",
        score=0,
        content=text,
        content_with_context=text,
    )


def test_bm25_ranks_exact_term_over_long_noise():
    docs = [
        _doc("noise", "the the the the the the the unrelated filler text " * 20),
        _doc("hit", "LangGraph checkpointing saves graph state after each step."),
    ]
    ranked = bm25_rank(docs, "LangGraph checkpointing")
    assert ranked[0].chunk_id == "hit"


def test_hybrid_memory_rrf():
    class K:
        def search(self, query, top_k=None):
            return [_doc("a", "alpha token"), _doc("b", "beta")]

    class V:
        def search(self, query, top_k=None):
            return [_doc("b", "beta"), _doc("c", "gamma")]

    h = HybridRetriever(K(), V(), bm25_top_k=2, vector_top_k=2, rrf_k=60, fusion_top_k=10)
    hits = h.search("q")
    assert {x.chunk_id for x in hits} == {"a", "b", "c"}
    assert hits[0].chunk_id == "b"


def test_reranker_orders_by_client():
    chunks = [_doc("1", "unrelated"), _doc("2", "LangGraph state object")]
    ranked = Reranker(FakeReranker(), top_k=2).rank("LangGraph state", chunks)
    assert ranked[0].chunk_id == "2"


def test_langchain_retriever_wraps_hybrid():
    class K:
        def search(self, query, top_k=None):
            return [_doc("a", "hello world")]

    class V:
        def search(self, query, top_k=None):
            return [_doc("a", "hello world")]

    hybrid = HybridRetriever(K(), V(), fusion_top_k=5)
    retriever = LangChainHybridRetriever(hybrid=hybrid, top_k=5)
    docs = retriever.invoke("hello")
    assert docs[0].metadata["chunk_id"] == "a"
    assert docs[0].page_content == "hello world"
