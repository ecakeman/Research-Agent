from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

FRONT_MATTER_RE = re.compile(r"^---\n(.*?)\n---\n?", re.DOTALL)


@dataclass
class ParsedDocument:
    title: str
    source_type: str
    source_name: str
    version: str | None
    url: str | None
    section: str | None
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)


def parse_front_matter(raw: str) -> tuple[dict[str, Any], str]:
    m = FRONT_MATTER_RE.match(raw)
    if not m:
        return {}, raw
    meta: dict[str, Any] = {}
    for line in m.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip().strip('"').strip("'")
    body = raw[m.end() :]
    return meta, body


def parse_markdown(raw: str, *, default_title: str, path: str) -> ParsedDocument:
    meta, body = parse_front_matter(raw)
    title = meta.pop("title", None) or default_title
    source_type = meta.pop("source_type", "official_docs")
    source_name = meta.pop("source_name", default_title)
    version = meta.pop("version", None) or "0.1"
    url = meta.pop("url", None) or None
    section = meta.pop("section", None) or None
    metadata = dict(meta)
    metadata.setdefault("path", path)
    metadata.setdefault("language", "en")
    return ParsedDocument(
        title=title,
        source_type=source_type,
        source_name=source_name,
        version=version,
        url=url,
        section=section,
        content=body.strip() + "\n",
        metadata=metadata,
    )
