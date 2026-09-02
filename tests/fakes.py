from __future__ import annotations

import json
from typing import Any, Callable

from app.models.clients import EmbeddingClient, GenerateResult, LLMClient, RerankClient


class FakeLLM(LLMClient):
    def __init__(self, handler: Callable[[str, list[dict[str, str]]], dict[str, Any]] | None = None):
        self.handler = handler
        self.calls: list[str] = []
        self.model = "fake"
        self.last_result: GenerateResult | None = None

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> GenerateResult:
        data = self.generate_json(messages)
        return self.last_result or GenerateResult(json.dumps(data, ensure_ascii=False))

    def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        blob = "\n".join(m.get("content") or "" for m in messages)
        self.calls.append(blob)
        if self.handler:
            data = self.handler(blob, messages)
        else:
            data = _default_handler(blob)
        self.last_result = GenerateResult(
            json.dumps(data, ensure_ascii=False),
            input_tokens=None,
            output_tokens=None,
            total_tokens=None,
            model=self.model,
        )
        return data


def _task(blob: str) -> str:
    if "analyze a technical research question" in blob.lower() or "You analyze a technical" in blob:
        return "analyze"
    if "Judge whether a chunk supports" in blob:
        return "grade"
    if "Rewrite the search query" in blob:
        return "rewrite"
    if "Extract evidence items" in blob:
        return "compress"
    if "Answer using ONLY" in blob:
        return "answer"
    return "unknown"


def _default_handler(blob: str) -> dict[str, Any]:
    task = _task(blob)
    if task == "analyze":
        return {
            "intent": "fact",
            "entities": ["LangGraph"],
            "sub_questions": ["state", "control flow", "checkpointing"],
        }
    if task == "grade":
        return {
            "chunk_id": "c1",
            "relevant": False,
            "support_level": "none",
            "reason": "unrelated",
            "covers": [],
        }
    if task == "rewrite":
        return {"rewritten_query": "LangGraph state control flow checkpointing", "focus": ["checkpointing"]}
    if task == "compress":
        return {"claim": "placeholder", "quote": ""}
    if task == "answer":
        return {"answer": "based on evidence", "citations": []}
    return {}


class TaggedLLM(FakeLLM):
    def __init__(self, tag: str, handler=None, log: list | None = None):
        super().__init__(handler)
        self.model = tag
        self.tag = tag
        self.log = log if log is not None else []

    def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        blob = "\n".join(m.get("content") or "" for m in messages)
        self.log.append((self.tag, _task(blob)))
        return super().generate_json(messages)


class FakeEmbedding(EmbeddingClient):
    def __init__(self, dim: int = 8):
        self.dim = dim

    def embed(self, texts: list[str]) -> list[list[float]]:
        out: list[list[float]] = []
        for t in texts:
            vec = [0.0] * self.dim
            for i, b in enumerate(t.encode("utf-8")):
                vec[i % self.dim] += (b % 13) / 13.0
            n = sum(x * x for x in vec) ** 0.5 or 1.0
            out.append([x / n for x in vec])
        return out


class FakeReranker(RerankClient):
    def rank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        q = query.lower()
        scored = []
        for i, ch in enumerate(chunks):
            text = (ch.get("content") or "").lower()
            score = sum(1.0 for tok in q.split() if tok in text)
            scored.append((score, i, ch))
        scored.sort(key=lambda x: x[0], reverse=True)
        out = []
        for rank, (_, i, ch) in enumerate(scored[:top_n]):
            row = dict(ch)
            row["rerank_score"] = float(scored[rank][0]) if rank < len(scored) else 0.0
            row["index"] = i
            row["rank"] = rank
            out.append(row)
        return out
