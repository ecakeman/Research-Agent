from app.errors import RunError
from app.graph.evidence_gap import REWRITE_EVIDENCE_MAX, collect_evidence_gaps, format_evidence_summary
from app.graph.nodes import GraphDeps, compress_node, rewrite_query
from app.graph.workflow import GRAPH_NODES, run_research
from app.models.routing import ModelRouter
from app.models.schemas import GradeResult, RetrievedChunk
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker, is_rewrite_prompt

import pytest


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


def test_evidence_summary_caps_at_eight():
    chunks = []
    grades = []
    for i in range(12):
        cid = f"c{i}"
        chunks.append(
            RetrievedChunk(
                chunk_id=cid,
                document_id="d",
                score=1.0,
                content=f"snippet body {i}",
                content_with_context="",
                source_title=f"Title {i}",
            )
        )
        grades.append(_g(cid, "partial", ["q"], "partial hit"))
    text = format_evidence_summary(chunks, grades)
    assert text.count("chunk_id=") == REWRITE_EVIDENCE_MAX
    none_chunk = RetrievedChunk(
        chunk_id="none",
        document_id="d",
        score=0.1,
        content="unrelated",
        source_title="Noise",
    )
    mixed = format_evidence_summary([none_chunk, chunks[0]], [_g("none", "none", [], "off"), grades[0]])
    assert "chunk_id=c0" in mixed
    assert "chunk_id=none" not in mixed


def test_compress_counts_llm_call_per_chunk():
    body = "State stores the shared structure."
    llm = FakeLLM(lambda blob, _: {"claim": "state is shared", "quote": body})
    deps = GraphDeps(
        models=ModelRouter.from_single_client(llm),
        search=lambda q, top_k=None: [],
        reranker=Reranker(FakeReranker()),
        conn=None,
    )
    chunks = [
        RetrievedChunk(chunk_id="a", document_id="d", score=1.0, content=body, content_with_context=body),
        RetrievedChunk(chunk_id="b", document_id="d", score=0.9, content=body, content_with_context=body),
    ]
    grades = [
        _g("a", "direct", ["state"], "ok").model_dump(),
        _g("b", "direct", ["state"], "ok").model_dump(),
    ]
    out = compress_node(
        {
            "original_query": "q",
            "reranked_chunks": [c.model_dump() for c in chunks],
            "graded_chunks": grades,
            "model_calls": 0,
        },
        deps,
    )
    compress_calls = [c for c in llm.calls if "Extract evidence items" in c]
    assert len(compress_calls) == 2
    assert out["model_calls"] == 2


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


def test_rerank_client_failure_fails_run_no_fallback():
    class Boom:
        def rank(self, query, chunks, top_n):
            raise RuntimeError("rerank api down")

    def search(q, top_k=None):
        return [
            RetrievedChunk(
                chunk_id="1",
                document_id="d",
                score=0.9,
                content="first fusion",
                content_with_context="first fusion",
            ),
            RetrievedChunk(
                chunk_id="2",
                document_id="d",
                score=0.1,
                content="second fusion",
                content_with_context="second fusion",
            ),
        ]

    deps = GraphDeps(
        models=ModelRouter.from_single_client(FakeLLM()),
        search=search,
        reranker=Reranker(Boom()),
        conn=None,
        baseline="agentic",
    )
    with pytest.raises(RunError) as ei:
        run_research(deps, "LangGraph state", "run-rerank-fail")
    assert ei.value.node == "rerank"
