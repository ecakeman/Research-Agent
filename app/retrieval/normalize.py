from __future__ import annotations

import re


def normalize_query(query: str) -> str:
    text = query.strip()
    text = re.sub(r"\s+", " ", text)
    return text
