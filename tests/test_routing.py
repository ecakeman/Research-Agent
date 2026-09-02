from __future__ import annotations

from app.graph.nodes import GraphDeps
from app.graph.workflow import GRAPH_NODES, build_graph, run_research
from app.models.routing import ModelRole, ModelRouter, answer_role
from app.models.schemas import RetrievedChunk
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker, TaggedLLM, is_rewrite_prompt


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


def _research_handler(blob, _):
    if "You analyze a technical research question" in blob:
        return {
            "intent": "comparison",
            "entities": ["LangGraph", "Chain"],
            "sub_questions": ["state handling", "control flow", "checkpointing"],
        }
    if "Judge whether a chunk supports" in blob:
        if "State stores" in blob:
            return {
                "chunk_id": "c-good",
                "relevant": True,
                "support_level": "direct",
                "reason": "covers",
                "covers": ["state handling", "control flow", "checkpointing"],
            }
        return {"chunk_id": "c-noise", "relevant": False, "support_level": "none", "reason": "off", "covers": []}
    if is_rewrite_prompt(blob):
        return {"rewritten_query": "LangGraph state control flow checkpointing", "focus": ["checkpointing"]}
    if "Extract evidence items" in blob:
        return {"claim": "State is shared", "quote": "State stores the shared structure."}
    if "Answer using ONLY" in blob:
        return {
            "answer": "LangGraph keeps named state and checkpoints.",
            "citations": [{"chunk_id": "c-good", "claim_index": 0}],
        }
    return {}


def _search(query, top_k=None):
    if "checkpointing" in query.lower() and "state control flow" in query.lower():
        return [GOOD]
    return [NOISE]


def test_t1_single_uses_same_client_for_all_roles():
    log: list = []
    llm = TaggedLLM("same", _research_handler, log)
    router = ModelRouter.from_single_client(llm, name="same")
    assert router.client(ModelRole.FAST) is router.client(ModelRole.PRO)
    deps = GraphDeps(
        models=router,
        search=_search,
        reranker=Reranker(FakeReranker()),
        baseline="agentic",
        max_retrieval_rounds=2,
    )
    run_research(deps, "Compare LangGraph with chain workflows.", "t1")
    tags = {t for t, _ in log}
    assert tags == {"same"}
    tasks = {task for _, task in log}
    assert {"analyze", "grade", "rewrite", "compress", "answer"} <= tasks


def test_t2_dual_role_mapping():
    log: list = []
    fast = TaggedLLM("fast", _research_handler, log)
    pro = TaggedLLM("pro", _research_handler, log)
    router = ModelRouter.from_clients(fast=fast, pro=pro, routing="dual")
    deps = GraphDeps(
        models=router,
        search=_search,
        reranker=Reranker(FakeReranker()),
        baseline="agentic",
        max_retrieval_rounds=2,
    )
    run_research(deps, "Compare LangGraph with chain workflows.", "t2")
    by_task: dict[str, set[str]] = {}
    for tag, task in log:
        by_task.setdefault(task, set()).add(tag)
    assert by_task["analyze"] == {"pro"}
    assert by_task["grade"] == {"fast"}
    assert by_task["rewrite"] == {"fast"}
    assert by_task["compress"] == {"fast"}
    assert by_task["answer"] == {"pro"}


def test_t3_citation_escalation_fast_then_pro():
    log: list = []

    def handler(blob, _):
        if "Answer using ONLY" in blob:
            return {"answer": "x", "citations": [{"chunk_id": "ghost", "claim_index": 0}]}
        return _research_handler(blob, _)

    fast = TaggedLLM("fast", handler, log)
    pro = TaggedLLM("pro", handler, log)
    router = ModelRouter.from_clients(fast=fast, pro=pro, routing="dual")
    deps = GraphDeps(
        models=router,
        search=_search,
        reranker=Reranker(FakeReranker()),
        baseline="agentic",
        max_retrieval_rounds=2,
    )
    state = run_research(deps, "Compare LangGraph with chain workflows.", "t3")
    answer_tags = [tag for tag, task in log if task == "answer"]
    assert answer_tags == ["pro", "fast", "pro"]
    assert state["status"] == "failed"
    assert state["citation_attempts"] == 3


def test_t4_graph_topology_unchanged():
    llm = FakeLLM(_research_handler)
    deps = GraphDeps(
        models=ModelRouter.from_single_client(llm),
        search=_search,
        reranker=Reranker(FakeReranker()),
    )
    compiled = build_graph(deps)
    names = set(compiled.get_graph().nodes.keys()) - {"__start__", "__end__"}
    assert names == set(GRAPH_NODES)
    assert GRAPH_NODES == (
        "analyze_query",
        "retrieve",
        "rerank",
        "grade_evidence",
        "rewrite_query",
        "compress_evidence",
        "generate_answer",
        "validate_citations",
        "finalize",
    )
    assert answer_role(0) is ModelRole.PRO
    assert answer_role(1) is ModelRole.FAST
    assert answer_role(2) is ModelRole.PRO


def test_t5_dual_missing_config_errors():
    from app.config import Settings

    missing_pro = Settings(
        model_routing="dual",
        llm_fast_base_url="http://fast",
        llm_fast_model="fast-model",
        llm_pro_base_url="",
        llm_pro_model="",
    )
    try:
        ModelRouter.from_settings(missing_pro)
        raise AssertionError("expected error")
    except RuntimeError as e:
        assert "PRO" in str(e)

    missing_fast = Settings(
        model_routing="dual",
        llm_fast_base_url="",
        llm_fast_model="",
        llm_pro_base_url="http://pro",
        llm_pro_model="pro-model",
    )
    try:
        ModelRouter.from_settings(missing_fast)
        raise AssertionError("expected error")
    except RuntimeError as e:
        assert "FAST" in str(e)
