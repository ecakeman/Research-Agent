from app.graph.nodes import GraphDeps
from app.graph.workflow import run_research
from app.models.routing import ModelRouter
from app.models.schemas import RetrievedChunk
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker, is_grade_prompt, is_rewrite_prompt


def _chunk(cid: str, text: str) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d",
        score=1.0,
        content=text,
        content_with_context=text,
        source_title="kb",
    )


def test_f1_two_rounds_then_abstain():
    def handler(blob, _):
        if "You analyze a technical research question" in blob:
            return {
                "intent": "comparison",
                "entities": ["LangGraph"],
                "sub_questions": ["state", "control flow", "checkpointing"],
            }
        if is_grade_prompt(blob):
            return {
                "chunk_id": "noise",
                "relevant": False,
                "support_level": "none",
                "reason": "unrelated",
                "covers": [],
            }
        if is_rewrite_prompt(blob):
            return {"rewritten_query": "LangGraph checkpointing internals", "focus": ["checkpointing"]}
        return {}

    def search(query, top_k=None):
        return [_chunk("noise", "The weather is sunny and cats are mammals.")]

    deps = GraphDeps(
        models=ModelRouter.from_single_client(FakeLLM(handler)),
        search=search,
        reranker=Reranker(FakeReranker()),
        conn=None,
        baseline="agentic",
        max_retrieval_rounds=2,
    )
    state = run_research(deps, "Compare LangGraph with chain workflows", "run-f1")
    assert state["status"] == "abstained"
    assert state["failure_reason"] == "insufficient_evidence"
    assert state["retrieval_round"] == 2
    assert "没有足够证据" in (state.get("final_answer") or "")
