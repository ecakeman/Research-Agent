from __future__ import annotations

from typing import Any

from app.generation.compress import format_evidence_for_prompt
from app.generation.prompts import ANSWER_PROMPT
from app.models.clients import LLMClient
from app.models.schemas import EvidenceItem


def _lc_messages(prompt, **kwargs) -> list[dict[str, str]]:
    messages = prompt.format_messages(**kwargs)
    role_map = {"human": "user", "ai": "assistant", "system": "system"}
    return [{"role": role_map.get(m.type, "user"), "content": m.content} for m in messages]


def generate_answer(llm: LLMClient, query: str, evidence: list[EvidenceItem]) -> dict[str, Any]:
    return llm.generate_json(
        _lc_messages(ANSWER_PROMPT, query=query, evidence=format_evidence_for_prompt(evidence))
    )
