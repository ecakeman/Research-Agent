from __future__ import annotations

from app.graph.nodes import GraphDeps
from app.graph.workflow import run_research
from app.models.routing import ModelRouter
from app.models.schemas import RetrievedChunk
from app.retrieval.rerank import Reranker
from tests.fakes import FakeLLM, FakeReranker, TaggedLLM


def _chunk(cid: str, text: str, extra: str = "") -> RetrievedChunk:
    body = text + extra
    return RetrievedChunk(
        chunk_id=cid,
        document_id="d",
        score=0.1,
        content=body,
        content_with_context=body,
        source_title="LangGraph",
        parent_section="State",
        heading_path=["LangGraph", "State"],
    )


GOOD = _chunk(
    "c-good",
    "State stores the shared structure. Control flow uses conditional edges. Checkpointing saves graph state.",
)
NOISE = _chunk("c-noise", "Tomato watering schedule for summer gardens.")


def _base_handler(blob, _):
    if "You analyze a technical research question" in blob:
        return {
            "intent": "comparison",
            "entities": ["LangGraph"],
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
    if "Rewrite the search query" in blob:
        return {"rewritten_query": "LangGraph state control flow checkpointing", "focus": ["checkpointing"]}
    if "Extract evidence items" in blob:
        return {"claim": "State is shared", "quote": "State stores the shared structure."}
    if "Answer using ONLY" in blob:
        return {
            "answer": "LangGraph keeps named state and checkpoints.",
            "citations": [{"chunk_id": "c-good", "claim_index": 0}],
        }
    return {}


def _deps(llm, search, routing="single", fast=None, pro=None) -> GraphDeps:
    if routing == "dual":
        models = ModelRouter.from_clients(fast=fast, pro=pro, routing="dual")
    else:
        models = ModelRouter.from_single_client(llm)
    return GraphDeps(
        models=models,
        search=search,
        reranker=Reranker(FakeReranker()),
        baseline="agentic",
        max_retrieval_rounds=2,
    )


def run_g1():
    state = run_research(
        _deps(FakeLLM(_base_handler), lambda q, k=None: [GOOD]),
        "What is LangGraph state?",
        "g1",
    )
    return state["status"] == "completed" and state["retrieval_round"] == 1


def run_g2():
    rounds = {"n": 0}

    def search(q, k=None):
        rounds["n"] += 1
        return [] if rounds["n"] == 1 else [GOOD]

    state = run_research(_deps(FakeLLM(_base_handler), search), "LangGraph state?", "g2")
    return state["status"] == "completed" and state["retrieval_round"] == 2 and state["evidence_sufficient"]


def run_g3():
    def search(q, k=None):
        return [NOISE, GOOD]

    state = run_research(_deps(FakeLLM(_base_handler), search), "LangGraph state checkpointing", "g3")
    ids = [e.get("chunk_id") for e in (state.get("evidence_chunks") or [])]
    graded_noise = [
        g
        for g in (state.get("graded_chunks") or [])
        if g.get("chunk_id") == "c-noise" and g.get("support_level") == "none"
    ]
    return state["status"] == "completed" and "c-good" in ids and graded_noise != []


def run_g4():
    def search(query, k=None):
        if "checkpointing" in query.lower() and "state control flow" in query.lower():
            return [GOOD]
        return [NOISE]

    state = run_research(_deps(FakeLLM(_base_handler), search), "Compare LangGraph with chains", "g4")
    return state["status"] == "completed" and state["retrieval_round"] == 2 and state["evidence_sufficient"]


def run_g5():
    def handler(blob, _):
        if "Judge whether a chunk supports" in blob:
            return {"chunk_id": "c-noise", "relevant": False, "support_level": "none", "reason": "x", "covers": []}
        return _base_handler(blob, _)

    state = run_research(
        _deps(FakeLLM(handler), lambda q, k=None: [NOISE]),
        "Compare LangGraph with chains",
        "g5",
    )
    return state["status"] == "abstained" and state["failure_reason"] == "insufficient_evidence"


def run_g6():
    log: list = []
    n = {"answer": 0}

    def handler(blob, _):
        if "Answer using ONLY" in blob:
            n["answer"] += 1
            if n["answer"] == 1:
                return {"answer": "bad", "citations": [{"chunk_id": "ghost", "claim_index": 0}]}
            return {
                "answer": "ok",
                "citations": [{"chunk_id": "c-good", "claim_index": 0}],
            }
        return _base_handler(blob, _)

    fast = TaggedLLM("fast", handler, log)
    pro = TaggedLLM("pro", handler, log)
    state = run_research(
        _deps(None, lambda q, k=None: [GOOD], routing="dual", fast=fast, pro=pro),
        "What is LangGraph state?",
        "g6",
    )
    answer_tags = [t for t, task in log if task == "answer"]
    return state["status"] == "completed" and answer_tags == ["pro", "fast"]


def run_g7():
    log: list = []

    def handler(blob, _):
        if "Answer using ONLY" in blob:
            return {"answer": "bad", "citations": [{"chunk_id": "ghost", "claim_index": 0}]}
        return _base_handler(blob, _)

    fast = TaggedLLM("fast", handler, log)
    pro = TaggedLLM("pro", handler, log)
    state = run_research(
        _deps(None, lambda q, k=None: [GOOD], routing="dual", fast=fast, pro=pro),
        "What is LangGraph state?",
        "g7",
    )
    answer_tags = [t for t, task in log if task == "answer"]
    return state["status"] == "failed" and answer_tags == ["pro", "fast", "pro"]


CASES = [
    ("G1", run_g1),
    ("G2", run_g2),
    ("G3", run_g3),
    ("G4", run_g4),
    ("G5", run_g5),
    ("G6", run_g6),
    ("G7", run_g7),
]
