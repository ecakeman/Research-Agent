from app.graph.evidence_gap import collect_evidence_gaps
from app.graph.nodes import GraphDeps, rewrite_query
from app.graph.workflow import GRAPH_NODES
from app.models.routing import ModelRouter
from app.models.schemas import GradeResult, RetrievedChunk
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker, is_rewrite_prompt


def _g(cid: str, level: str, covers: list[str], reason: str = "") -> GradeResult:
    relevant = level != "none"
    return GradeResult(
        chunk_id=cid,
        relevant=relevant,
        support_level=level,
        covers=covers,
        reason=reason,
    )


def test_t1_missing_subquestions_and_gaps():
    subs = ["state handling", "control flow", "checkpointing"]
    grades = [
        _g("1", "direct", ["state handling"], "explicit state object"),
        _g("2", "partial", ["control flow"], "mentions edges only"),
        _g("3", "none", ["checkpointing"], "no persistence details"),
    ]
    missing, gaps = collect_evidence_gaps(subs, grades)
    assert missing == ["control flow", "checkpointing"]
    assert len(gaps) == 2
    joined = " ".join(gaps)
    assert "control flow" in joined or "mentions edges only" in joined
    assert "checkpointing" in joined or "no persistence details" in joined


def test_t2_rewrite_prompt_includes_evidence_not_id_colon_level():
    llm = FakeLLM(
        lambda blob, _: {"rewritten_query": "LangGraph checkpointing persistence", "focus": ["checkpointing"]}
    )
    deps = GraphDeps(
        models=ModelRouter.from_single_client(llm),
        search=lambda q, top_k=None: [],
        reranker=Reranker(FakeReranker()),
        conn=None,
    )
    chunk = RetrievedChunk(
        chunk_id="c1",
        document_id="d",
        score=0.9,
        content="Checkpointing saves graph state after each super-step in the runtime.",
        content_with_context="",
        source_title="LangGraph Persistence",
        parent_section="Checkpoints",
        heading_path=["LangGraph", "Persistence"],
    )
    state = {
        "original_query": "How does LangGraph persist state?",
        "sub_questions": ["checkpointing"],
        "reranked_chunks": [chunk.model_dump()],
        "graded_chunks": [
            _g("c1", "partial", ["checkpointing"], "only mentions save, not restore").model_dump()
        ],
        "missing_sub_questions": ["restore semantics"],
        "evidence_gaps": ["restore semantics — only mentions save, not restore"],
        "failure_reason": "insufficient_evidence",
    }
    rewrite_query(state, deps)
    blob = "\n".join(llm.calls)
    assert is_rewrite_prompt(blob)
    assert "LangGraph Persistence" in blob
    assert "Checkpoints" in blob
    assert "partial" in blob
    assert "only mentions save, not restore" in blob
    assert "super-step" in blob
    assert "Missing sub-questions" in blob
    assert "Evidence gaps" in blob
    assert "c1:partial" not in blob.replace(" ", "")


def test_t3_rewrite_accepts_json_fields():
    llm = FakeLLM(lambda blob, _: {"rewritten_query": "targeted retrieval query", "focus": ["gap-a"]})
    deps = GraphDeps(
        models=ModelRouter.from_single_client(llm),
        search=lambda q, top_k=None: [],
        reranker=Reranker(FakeReranker()),
        conn=None,
    )
    out = rewrite_query(
        {
            "original_query": "q",
            "sub_questions": [],
            "reranked_chunks": [],
            "graded_chunks": [],
            "missing_sub_questions": [],
            "evidence_gaps": [],
        },
        deps,
    )
    assert out["rewritten_query"] == "targeted retrieval query"


def test_t4_graph_nodes_tuple_order():
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
