from __future__ import annotations

from langgraph.graph import END, START, StateGraph

from app.graph.nodes import (
    GraphDeps,
    analyze_query,
    compress_node,
    finalize,
    generate_node,
    grade_evidence,
    rerank,
    retrieve,
    rewrite_query,
    route_after_grade,
    route_after_validate,
    validate_citations_node,
)
from app.models.schemas import ResearchState


GRAPH_NODES = (
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


def _bind(fn, deps: GraphDeps):
    def node(state, *args, **kwargs):
        return fn(state, deps)

    node.__name__ = fn.__name__
    return node


def build_graph(deps: GraphDeps):
    g = StateGraph(ResearchState)
    g.add_node("analyze_query", _bind(analyze_query, deps))
    g.add_node("retrieve", _bind(retrieve, deps))
    g.add_node("rerank", _bind(rerank, deps))
    g.add_node("grade_evidence", _bind(grade_evidence, deps))
    g.add_node("rewrite_query", _bind(rewrite_query, deps))
    g.add_node("compress_evidence", _bind(compress_node, deps))
    g.add_node("generate_answer", _bind(generate_node, deps))
    g.add_node("validate_citations", _bind(validate_citations_node, deps))
    g.add_node("finalize", _bind(finalize, deps))

    g.add_edge(START, "analyze_query")
    g.add_edge("analyze_query", "retrieve")
    g.add_edge("retrieve", "rerank")
    g.add_edge("rerank", "grade_evidence")
    g.add_conditional_edges(
        "grade_evidence",
        route_after_grade,
        {
            "compress_evidence": "compress_evidence",
            "rewrite_query": "rewrite_query",
            "finalize": "finalize",
        },
    )
    g.add_edge("rewrite_query", "retrieve")
    g.add_edge("compress_evidence", "generate_answer")
    g.add_edge("generate_answer", "validate_citations")
    g.add_conditional_edges(
        "validate_citations",
        route_after_validate,
        {"generate_answer": "generate_answer", "finalize": "finalize"},
    )
    g.add_edge("finalize", END)
    return g.compile()


def run_research(deps: GraphDeps, query: str, run_id: str) -> ResearchState:
    graph = build_graph(deps)
    initial: ResearchState = {
        "run_id": run_id,
        "original_query": query,
        "normalized_query": query,
        "rewritten_query": None,
        "research_plan": [],
        "retrieved_chunks": [],
        "reranked_chunks": [],
        "graded_chunks": [],
        "evidence_chunks": [],
        "evidence_sufficient": False,
        "first_pass_evidence_sufficient": None,
        "citations": [],
        "retrieval_round": 0,
        "max_retrieval_rounds": deps.max_retrieval_rounds or deps.settings.max_retrieval_rounds,
        "token_budget": deps.settings.token_budget,
        "model_calls": 0,
        "input_tokens": 0,
        "output_tokens": 0,
        "citation_attempts": 0,
        "status": "running",
    }
    if deps.baseline != "agentic":
        initial["max_retrieval_rounds"] = 1
        deps.max_retrieval_rounds = 1
    return graph.invoke(initial)
