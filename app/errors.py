from __future__ import annotations

import json
import re
from typing import Any

def redact(text: str, limit: int = 500) -> str:
    cleaned = re.sub(r"(?i)bearer\s+\S+", "Bearer [REDACTED]", text or "")
    cleaned = re.sub(r"sk-[A-Za-z0-9._-]+", "[REDACTED]", cleaned)
    cleaned = cleaned.replace("\x00", "")
    return cleaned[:limit]


class RunError(Exception):
    """单次 research/eval case 的结构化失败，不含 Authorization。"""

    def __init__(
        self,
        message: str = "",
        *,
        run_id: str | None = None,
        question_id: str | None = None,
        node: str | None = None,
        model_role: str | None = None,
        model_name: str | None = None,
        http_status: int | None = None,
        raw_excerpt: str | None = None,
        parse_error: str | None = None,
    ):
        super().__init__(message or parse_error or "run_failed")
        self.run_id = run_id
        self.question_id = question_id
        self.node = node
        self.model_role = model_role
        self.model_name = model_name
        self.http_status = http_status
        self.raw_excerpt = redact(raw_excerpt or "") if raw_excerpt else None
        self.parse_error = parse_error

    def annotate(
        self,
        *,
        run_id: str | None = None,
        question_id: str | None = None,
        node: str | None = None,
        model_role: str | None = None,
        model_name: str | None = None,
    ) -> RunError:
        self.run_id = self.run_id or run_id
        self.question_id = self.question_id or question_id
        self.node = self.node or node
        self.model_role = self.model_role or model_role
        self.model_name = self.model_name or model_name
        return self

    def to_record(self, question_id: str | None = None) -> dict[str, Any]:
        qid = question_id or self.question_id
        return {
            "id": qid,
            "question_id": qid,
            "status": "failed",
            "error": str(self)[:800],
            "run_id": self.run_id,
            "node": self.node,
            "model_role": self.model_role,
            "model_name": self.model_name,
            "http_status": self.http_status,
            "raw_excerpt": self.raw_excerpt,
            "parse_error": self.parse_error,
        }


def find_run_error(exc: BaseException | None) -> RunError | None:
    seen: set[int] = set()
    cur: BaseException | None = exc
    while cur is not None and id(cur) not in seen:
        seen.add(id(cur))
        if isinstance(cur, RunError):
            return cur
        cur = cur.__cause__ or cur.__context__
    return None


def from_httpx(exc: BaseException, *, model_name: str | None = None) -> RunError:
    import httpx

    status = None
    excerpt = None
    if isinstance(exc, httpx.HTTPStatusError) and exc.response is not None:
        status = exc.response.status_code
        excerpt = redact(exc.response.text or "")
    elif isinstance(exc, httpx.HTTPError):
        excerpt = redact(str(exc))
    return RunError(str(exc)[:400], http_status=status, raw_excerpt=excerpt, model_name=model_name)


def from_json_parse(exc: json.JSONDecodeError, raw: str, *, model_name: str | None = None) -> RunError:
    return RunError(
        str(exc),
        parse_error=str(exc),
        raw_excerpt=redact(raw),
        model_name=model_name,
    )
