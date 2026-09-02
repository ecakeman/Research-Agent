from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _encoding():
    import tiktoken

    return tiktoken.get_encoding("cl100k_base")


def count_tokens(text: str) -> int:
    if not text:
        return 0
    return len(_encoding().encode(text))


def take_tokens(text: str, n: int) -> str:
    enc = _encoding()
    ids = enc.encode(text)
    return enc.decode(ids[:n])
