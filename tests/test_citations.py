from app.generation.citations import validate_answer_citations
from app.graph.nodes import GraphDeps, generate_node, validate_citations_node
from app.models.routing import ModelRouter
from app.models.schemas import EvidenceItem, ResearchState
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker


def _item() -> EvidenceItem:
    return EvidenceItem(
        chunk_id="c-real",
        document_id="d1",
        claim="state is shared",
        quote="State stores the shared structure",
        source_title="LangGraph",
        section="State",
        support_level="direct",
    )


def test_invalid_chunk_id_rejected():
    ok, reason = validate_answer_citations(
        {"answer": "hi", "citations": [{"chunk_id": "ghost", "claim_index": 0}]},
        [_item()],
    )
    assert ok is False
    assert reason == "citation_not_in_evidence"


def test_valid_citation_accepted():
    ok, reason = validate_answer_citations(
        {"answer": "hi", "citations": [{"chunk_id": "c-real", "claim_index": 0}]},
        [_item()],
    )
    assert ok is True
    assert reason is None


def test_generation_retries_then_fails_on_hallucinated_citation():
    calls = {"n": 0}

    def handler(blob, _msgs):
        if "Answer using ONLY" in blob:
            calls["n"] += 1
            return {"answer": "x", "citations": [{"chunk_id": "does-not-exist", "claim_index": 0}]}
        if "analyze" in blob.lower() and "You analyze" in blob:
            return {"intent": "fact", "entities": [], "sub_questions": ["state", "control flow", "checkpointing"]}
        return {"relevant": True, "support_level": "direct", "covers": ["state"], "chunk_id": "c-real", "reason": "ok"}

    llm = FakeLLM(handler)
    deps = GraphDeps(
        models=ModelRouter.from_single_client(llm),
        search=lambda q, k=None: [],
        reranker=Reranker(FakeReranker()),
        conn=None,
    )
    state: ResearchState = {
        "run_id": "t",
        "original_query": "q",
        "evidence_chunks": [_item().model_dump()],
        "draft_answer": None,
        "citations": [],
        "citation_attempts": 0,
        "model_calls": 0,
        "evidence_sufficient": True,
    }
    state.update(generate_node(state, deps))
    state.update(validate_citations_node(state, deps))
    assert state["generation_invalid"] is True
    state.update(generate_node(state, deps))
    state.update(validate_citations_node(state, deps))
    assert state["citation_attempts"] == 2
    assert state["generation_invalid"] is True
    state.update(generate_node(state, deps))
    state.update(validate_citations_node(state, deps))
    assert state["citation_attempts"] == 3
    assert state["generation_invalid"] is True
    from app.graph.nodes import finalize

    fin = finalize(state, deps)
    assert fin["status"] == "failed"
    assert calls["n"] == 3
