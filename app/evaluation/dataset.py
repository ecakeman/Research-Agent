from __future__ import annotations

import json
from pathlib import Path

DEFAULT_PATH = Path(__file__).resolve().parent.parent.parent / "eval" / "questions.jsonl"

CATEGORIES = {"fact", "comparison", "multi-hop", "insufficient", "ambiguous"}
ALIASES = {"insufficient-evidence": "insufficient", "multi_hop": "multi-hop"}
REQUIRED = ("id", "category", "question", "expected_sources", "expected_claims")


def normalize_category(raw: str) -> str:
    value = (raw or "").strip()
    return ALIASES.get(value, value)


def validate_case(obj: dict) -> dict:
    missing = [k for k in REQUIRED if k not in obj]
    if missing:
        raise ValueError(f"eval case missing fields: {missing}")
    cat = normalize_category(str(obj["category"]))
    if cat not in CATEGORIES:
        raise ValueError(f"invalid category: {obj['category']}")
    out = dict(obj)
    out["category"] = cat
    if not isinstance(out.get("expected_sources"), list):
        raise ValueError("expected_sources must be a list")
    if not isinstance(out.get("expected_claims"), list):
        raise ValueError("expected_claims must be a list")
    return out


def load_cases(path: Path | None = None) -> list[dict]:
    p = path or DEFAULT_PATH
    if not p.exists():
        return []
    cases = []
    for line in p.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        cases.append(validate_case(json.loads(line)))
    return cases
