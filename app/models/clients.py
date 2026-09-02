from __future__ import annotations

import json
from typing import Any

import httpx

from app.config import settings
from app.errors import RunError, from_httpx, from_json_parse


def _join(base: str, path: str) -> str:
    return base.rstrip("/") + path


class GenerateResult:
    def __init__(
        self,
        text: str,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
        total_tokens: int | None = None,
        model: str | None = None,
    ):
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.model = model


def parse_json_text(text: str) -> dict[str, Any]:
    import json

    raw = text.strip()
    if raw.startswith("```"):
        raw = raw.strip("`")
        if raw.startswith("json"):
            raw = raw[4:]
    data = json.loads(raw)
    if isinstance(data, list):
        data = next((x for x in data if isinstance(x, dict)), {})
    if not isinstance(data, dict):
        return {}
    return data


class LLMClient:
    last_result: GenerateResult | None = None

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> GenerateResult:
        raise NotImplementedError

    def generate_json(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        result = self.generate(messages, json_mode=True)
        self.last_result = result
        try:
            return parse_json_text(result.text)
        except json.JSONDecodeError as exc:
            raise from_json_parse(exc, result.text, model_name=getattr(self, "model", None)) from exc


class EmbeddingClient:
    def embed(self, texts: list[str]) -> list[list[float]]:
        raise NotImplementedError


class RerankClient:
    def rank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        raise NotImplementedError


class HTTPLLMClient(LLMClient):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 120.0,
    ):
        self.base_url = (base_url or settings.llm_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.llm_api_key
        self.model = model or settings.llm_model
        self.timeout = timeout

    def generate(self, messages: list[dict[str, str]], *, json_mode: bool = False) -> GenerateResult:
        if not self.base_url or not self.model:
            raise RuntimeError("LLM_BASE_URL / LLM_MODEL 未配置")
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": 0,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(_join(self.base_url, "/chat/completions"), json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                raise from_httpx(exc, model_name=self.model) from exc
        choice = data["choices"][0]["message"]["content"] or ""
        usage = data.get("usage")
        pt = ct = tt = None
        if isinstance(usage, dict):
            if usage.get("prompt_tokens") is not None:
                pt = int(usage["prompt_tokens"])
            if usage.get("completion_tokens") is not None:
                ct = int(usage["completion_tokens"])
            if usage.get("total_tokens") is not None:
                tt = int(usage["total_tokens"])
            elif pt is not None and ct is not None:
                tt = pt + ct
        result = GenerateResult(
            text=choice,
            input_tokens=pt,
            output_tokens=ct,
            total_tokens=tt,
            model=self.model,
        )
        self.last_result = result
        return result


class HTTPEmbeddingClient(EmbeddingClient):
    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ):
        self.base_url = (base_url or settings.embedding_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.embedding_api_key
        self.model = model or settings.embedding_model
        self.timeout = timeout

    def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        if not self.base_url or not self.model:
            raise RuntimeError("EMBEDDING_BASE_URL / EMBEDDING_MODEL 未配置")
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"model": self.model, "input": texts}
        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(_join(self.base_url, "/embeddings"), json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                raise from_httpx(exc, model_name=self.model) from exc
        items = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in items]


def _rerank_endpoint(base_url: str) -> str:
    """兼容模式根路径不能直 POST；qwen3.7-text-rerank 走 DashScope 原生 text-rerank。"""
    b = base_url.rstrip("/")
    if b.endswith("/reranks") or "text-rerank" in b:
        return b
    if "compatible-mode" in b or "compatible-api" in b:
        host = "https://dashscope-intl.aliyuncs.com" if "dashscope-intl" in b else "https://dashscope.aliyuncs.com"
        return f"{host}/api/v1/services/rerank/text-rerank/text-rerank"
    return _join(b, "/reranks")


class HTTPRerankClient(RerankClient):
    """百炼 text-rerank JSON；兼容模式 BASE_URL 自动解析到原生 rerank 路径。"""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 30.0,
    ):
        self.base_url = (base_url or settings.rerank_base_url).rstrip("/")
        self.api_key = api_key if api_key is not None else settings.rerank_api_key
        self.model = model or settings.rerank_model
        self.timeout = timeout

    def rank(self, query: str, chunks: list[dict], top_n: int) -> list[dict]:
        if not chunks:
            return []
        if not self.base_url or not self.model:
            raise RuntimeError("RERANK_BASE_URL / RERANK_MODEL 未配置")
        if top_n <= 0 or top_n > len(chunks):
            top_n = len(chunks)
        texts = [c.get("content_with_context") or c.get("content") or "" for c in chunks]
        payload = {
            "model": self.model,
            "input": {"query": query, "documents": texts},
            "parameters": {"top_n": top_n},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        url = _rerank_endpoint(self.base_url)
        with httpx.Client(timeout=self.timeout) as client:
            try:
                resp = client.post(url, json=payload, headers=headers)
                resp.raise_for_status()
                data = resp.json()
            except httpx.HTTPError as exc:
                raise from_httpx(exc, model_name=self.model) from exc
        results = ((data.get("output") or {}).get("results")) or data.get("results") or []
        ranked: list[dict] = []
        seen: set[int] = set()
        for item in results:
            idx = int(item.get("index", -1))
            if idx < 0 or idx >= len(chunks) or idx in seen:
                continue
            seen.add(idx)
            row = dict(chunks[idx])
            row["rerank_score"] = float(item.get("relevance_score") or item.get("score") or 0.0)
            row["rank"] = len(ranked)
            ranked.append(row)
            if len(ranked) >= top_n:
                break
        if not ranked:
            raise RuntimeError("rerank_empty")
        return ranked
