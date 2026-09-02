import pytest

from app.errors import RunError
from app.graph.nodes import GraphDeps, route_after_grade
from app.graph.workflow import run_research
from app.models.routing import ModelRouter
from app.models.schemas import ResearchState, RetrievedChunk
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker, is_grade_prompt, is_rewrite_prompt


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d",
        score=1.0,
        content=text,
        content_with_context=text,
        source_title="LangGraph",
        parent_section="State",
        heading_path=["LangGraph", "State"],
    )


GOOD = _chunk(
    "c-good",
    "State stores the shared structure. Control flow uses conditional edges. Checkpointing saves graph state.",
)
NOISE = _chunk("c-noise", "Unrelated gardening advice about tomatoes.")


def test_f4_rewrite_then_sufficient():
    def handler(blob, _):
        if "You analyze a technical research question" in blob:
            return {
                "intent": "comparison",
                "entities": ["LangGraph", "Chain"],
                "sub_questions": ["state handling", "control flow", "checkpointing"],
            }
        if is_grade_prompt(blob):
            if "State stores" in blob:
                return {
                    "chunk_id": "c-good",
                    "relevant": True,
                    "support_level": "direct",
                    "reason": "covers core topics",
                    "covers": ["state handling", "control flow", "checkpointing"],
                }
            return {
                "chunk_id": "c-noise",
                "relevant": False,
                "support_level": "none",
                "reason": "off topic",
                "covers": [],
            }
        if is_rewrite_prompt(blob):
            return {"rewritten_query": "LangGraph state control flow checkpointing", "focus": ["checkpointing"]}
        if "Extract evidence items" in blob:
            return {
                "claim": "State is shared and checkpointed",
                "quote": "State stores the shared structure.",
            }
        if "Answer using ONLY" in blob:
            return {
                "answer": "LangGraph keeps named state, branches with conditional edges, and checkpoints.",
                "citations": [{"chunk_id": "c-good", "claim_index": 0}],
            }
        return {}

    def search(query, top_k=None):
        if "checkpointing" in query.lower() and "state control flow" in query.lower():
            return [GOOD]
        return [NOISE]

    deps = GraphDeps(
        models=ModelRouter.from_single_client(FakeLLM(handler)),
        search=search,
        reranker=Reranker(FakeReranker()),
        conn=None,
        baseline="agentic",
        max_retrieval_rounds=2,
    )
    state = run_research(deps, "Compare LangGraph with traditional chain workflows.", "run-f4")
    assert state["retrieval_round"] == 2
    assert state["evidence_sufficient"] is True
    assert state["status"] == "completed"
    assert state["citations"][0]["chunk_id"] == "c-good"


def test_rerank_client_failure_fails_the_run():
    class Boom:
        def rank(self, query, chunks, top_n):
            raise RuntimeError("rerank api down")

    deps = GraphDeps(
        models=ModelRouter.from_single_client(FakeLLM()),
        search=lambda q, top_k=None: [_chunk("1", "first"), _chunk("2", "second")],
        reranker=Reranker(Boom()),
        conn=None,
        baseline="agentic",
        max_retrieval_rounds=1,
    )
    with pytest.raises(RunError) as ei:
        run_research(deps, "What is LangGraph state?", "run-rerank-fail")
    assert ei.value.node == "rerank"


def test_route_after_grade_rewrites_only_before_max():
    s: ResearchState = {
        "evidence_sufficient": False,
        "retrieval_round": 1,
        "max_retrieval_rounds": 2,
    }
    assert route_after_grade(s) == "rewrite_query"
    s["retrieval_round"] = 2
    assert route_after_grade(s) == "finalize"
    s["evidence_sufficient"] = True
    assert route_after_grade(s) == "compress_evidence"
