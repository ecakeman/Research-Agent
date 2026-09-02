from __future__ import annotations

import re
from dataclasses import dataclass

from app.config import settings
from app.ingestion.tokens import count_tokens, take_tokens
from app.models.schemas import ChunkRecord

HEADING_RE = re.compile(r"^(#{1,4})\s+(.*)$")
FENCE_RE = re.compile(r"^```([\w+-]*)\s*$")
TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
QUOTE_RE = re.compile(r"^>\s?")
LIST_RE = re.compile(r"^(\s*)([-*+]|\d+\.)\s+")


@dataclass
class Block:
    kind: str
    text: str
    heading_level: int | None = None
    heading_title: str | None = None
    language: str | None = None
    start: int = 0
    end: int = 0


def _split_lines_with_offset(content: str) -> list[tuple[int, str]]:
    lines: list[tuple[int, str]] = []
    offset = 0
    for line in content.splitlines(keepends=True):
        lines.append((offset, line))
        offset += len(line)
    return lines


def parse_blocks(content: str) -> list[Block]:
    rows = _split_lines_with_offset(content)
    blocks: list[Block] = []
    i = 0
    n = len(rows)
    while i < n:
        off, line = rows[i]
        stripped = line.rstrip("\n")
        fence = FENCE_RE.match(stripped)
        if fence:
            lang = fence.group(1) or None
            start = off
            buf = [line]
            i += 1
            while i < n:
                buf.append(rows[i][1])
                if FENCE_RE.match(rows[i][1].rstrip("\n")):
                    end = rows[i][0] + len(rows[i][1])
                    i += 1
                    break
                i += 1
            else:
                end = rows[-1][0] + len(rows[-1][1])
            blocks.append(Block("code", "".join(buf), language=lang, start=start, end=end))
            continue
        hm = HEADING_RE.match(stripped)
        if hm:
            level = len(hm.group(1))
            title = hm.group(2).strip()
            end = off + len(line)
            blocks.append(
                Block("heading", stripped, heading_level=level, heading_title=title, start=off, end=end)
            )
            i += 1
            continue
        if TABLE_ROW_RE.match(stripped):
            start = off
            buf = [line]
            i += 1
            while i < n and TABLE_ROW_RE.match(rows[i][1].rstrip("\n")):
                buf.append(rows[i][1])
                i += 1
            end = rows[i - 1][0] + len(rows[i - 1][1])
            blocks.append(Block("table", "".join(buf), start=start, end=end))
            continue
        if LIST_RE.match(stripped):
            start = off
            buf = [line]
            i += 1
            while i < n:
                nxt = rows[i][1].rstrip("\n")
                if not nxt.strip():
                    buf.append(rows[i][1])
                    i += 1
                    if i < n and LIST_RE.match(rows[i][1].rstrip("\n")):
                        continue
                    break
                if LIST_RE.match(nxt) or (nxt.startswith(" ") and buf):
                    buf.append(rows[i][1])
                    i += 1
                    continue
                break
            end = rows[i - 1][0] + len(rows[i - 1][1])
            blocks.append(Block("list", "".join(buf).rstrip() + "\n", start=start, end=end))
            continue
        if QUOTE_RE.match(stripped):
            start = off
            buf = [line]
            i += 1
            while i < n and QUOTE_RE.match(rows[i][1].rstrip("\n")):
                buf.append(rows[i][1])
                i += 1
            end = rows[i - 1][0] + len(rows[i - 1][1])
            blocks.append(Block("quote", "".join(buf), start=start, end=end))
            continue
        if not stripped.strip():
            i += 1
            continue
        start = off
        buf = [line]
        i += 1
        while i < n:
            nxt = rows[i][1]
            ns = nxt.rstrip("\n")
            if not ns.strip():
                break
            if (
                HEADING_RE.match(ns)
                or FENCE_RE.match(ns)
                or TABLE_ROW_RE.match(ns)
                or LIST_RE.match(ns)
                or QUOTE_RE.match(ns)
            ):
                break
            buf.append(nxt)
            i += 1
        end = rows[i - 1][0] + len(rows[i - 1][1])
        blocks.append(Block("paragraph", "".join(buf).rstrip() + "\n", start=start, end=end))
    return blocks


def _context_prefix(title: str, heading_path: list[str]) -> str:
    section = " > ".join(heading_path) if heading_path else ""
    lines = [f"Document: {title}"]
    if section:
        lines.append(f"Section: {section}")
    return "\n".join(lines) + "\n\n"


def _chunk_type(kinds: list[str]) -> str:
    unique = {k for k in kinds if k != "heading"}
    if unique == {"code"}:
        return "code"
    if unique == {"table"}:
        return "table"
    if unique == {"list"}:
        return "list"
    if unique == {"quote"}:
        return "quote"
    if unique <= {"paragraph", "text"} or unique == {"paragraph"}:
        return "text"
    if len(unique) <= 1:
        return next(iter(unique), "text")
    return "mixed"


def _pack_text(text: str, max_tokens: int, overlap: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    parts: list[str] = []
    remaining = text
    while remaining:
        piece = take_tokens(remaining, max_tokens)
        if not piece:
            break
        parts.append(piece)
        if piece == remaining:
            break
        overlap_text = take_tokens(piece, overlap) if overlap else ""
        remaining = remaining[len(piece) :]
        if overlap_text and remaining:
            remaining = overlap_text + remaining
    return parts


def _split_code(text: str, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    inner = text
    fence_open = ""
    fence_close = ""
    lines = text.splitlines(keepends=True)
    if lines and lines[0].startswith("```"):
        fence_open = lines[0]
        if lines[-1].strip().startswith("```"):
            fence_close = lines[-1]
            inner = "".join(lines[1:-1])
        else:
            inner = "".join(lines[1:])
    groups: list[str] = []
    buf: list[str] = []
    for line in inner.splitlines(keepends=True):
        stripped = line.lstrip()
        boundary = stripped.startswith(("def ", "class ", "function ", "func ", "pub fn ", "fn "))
        if boundary and buf and count_tokens("".join(buf)) > 40:
            groups.append("".join(buf))
            buf = [line]
        else:
            buf.append(line)
            if count_tokens("".join(buf)) >= max_tokens:
                groups.append("".join(buf))
                buf = []
    if buf:
        groups.append("".join(buf))
    if not groups:
        groups = _pack_text(inner, max_tokens, 0)
    wrapped = []
    for g in groups:
        body = g
        if fence_open:
            body = fence_open + g
            if fence_close and not g.rstrip().endswith("```"):
                body = body.rstrip() + "\n" + fence_close
        if count_tokens(body) > max_tokens:
            wrapped.extend(_pack_text(body, max_tokens, 0))
        else:
            wrapped.append(body)
    return wrapped


def _split_table(text: str, max_tokens: int) -> list[str]:
    if count_tokens(text) <= max_tokens:
        return [text]
    lines = [ln for ln in text.splitlines() if ln.strip()]
    if len(lines) < 3:
        return _pack_text(text, max_tokens, 0)
    header = lines[0] + "\n" + lines[1] + "\n"
    parts: list[str] = []
    buf = header
    for row in lines[2:]:
        candidate = buf + row + "\n"
        if count_tokens(candidate) > max_tokens and buf != header:
            parts.append(buf)
            buf = header + row + "\n"
        else:
            buf = candidate
    if buf.strip():
        parts.append(buf)
    return parts


def split_document(
    content: str,
    *,
    title: str,
    target_tokens: int | None = None,
    max_tokens: int | None = None,
    min_tokens: int | None = None,
    overlap_tokens: int | None = None,
) -> list[ChunkRecord]:
    target = target_tokens or settings.target_tokens
    max_tok = max_tokens or settings.max_tokens
    min_tok = min_tokens or settings.min_tokens
    overlap = overlap_tokens or settings.overlap_tokens
    blocks = parse_blocks(content)
    heading_path: list[str] = []
    sections: list[dict] = []
    current: dict | None = None

    def flush():
        nonlocal current
        if current and current["items"]:
            sections.append(current)
        current = None

    for block in blocks:
        if block.kind == "heading":
            level = block.heading_level or 1
            heading_path = heading_path[: level - 1] + [block.heading_title or ""]
            flush()
            current = {
                "path": list(heading_path),
                "level": level,
                "items": [block],
            }
            continue
        if current is None:
            current = {"path": list(heading_path), "level": 99, "items": []}
        current["items"].append(block)
    flush()

    raw_chunks: list[ChunkRecord] = []
    for sec in sections:
        path: list[str] = sec["path"]
        parent = path[-1] if path else None
        items: list[Block] = [b for b in sec["items"] if b.kind != "heading"]
        i = 0
        while i < len(items):
            block = items[i]
            if block.kind == "code":
                pieces = _split_code(block.text, max_tok)
                for p in pieces:
                    raw_chunks.append(_make_chunk(p, title, path, parent, "code", block.start, block.end, {"language": block.language}))
                i += 1
                continue
            if block.kind == "table":
                pieces = _split_table(block.text, max_tok)
                for p in pieces:
                    raw_chunks.append(_make_chunk(p, title, path, parent, "table", block.start, block.end, {}))
                i += 1
                continue
            buf_text = ""
            kinds = []
            start = items[i].start
            end = items[i].end
            while i < len(items) and items[i].kind not in {"code", "table"}:
                candidate = buf_text + items[i].text
                if buf_text and count_tokens(candidate) > max_tok:
                    break
                buf_text = candidate
                kinds.append(items[i].kind)
                end = items[i].end
                i += 1
                if count_tokens(buf_text) >= target:
                    break
            if not buf_text.strip():
                i += 1
                continue
            pieces = _pack_text(buf_text, max_tok, overlap) if count_tokens(buf_text) > max_tok else [buf_text]
            ctype = _chunk_type(kinds)
            for p in pieces:
                raw_chunks.append(_make_chunk(p, title, path, parent, ctype, start, end, {}))

    merged = _merge_short(raw_chunks, min_tok)
    for idx, ch in enumerate(merged):
        ch.chunk_index = idx
    return merged


def _make_chunk(
    content: str,
    title: str,
    heading_path: list[str],
    parent: str | None,
    chunk_type: str,
    start: int,
    end: int,
    extra_meta: dict,
) -> ChunkRecord:
    content = content.strip() + "\n"
    ctx = _context_prefix(title, heading_path) + content
    meta = dict(extra_meta)
    if extra_meta.get("language"):
        meta["language"] = extra_meta["language"]
    return ChunkRecord(
        chunk_index=0,
        parent_section=parent,
        heading_path=heading_path,
        content=content,
        content_with_context=ctx,
        token_count=count_tokens(ctx),
        char_count=len(content),
        start_offset=start,
        end_offset=end,
        chunk_type=chunk_type,
        metadata=meta,
    )


def _merge_short(chunks: list[ChunkRecord], min_tok: int) -> list[ChunkRecord]:
    if not chunks:
        return []
    out: list[ChunkRecord] = []
    for ch in chunks:
        if (
            out
            and ch.token_count < min_tok
            and out[-1].heading_path == ch.heading_path
            and out[-1].chunk_type == ch.chunk_type
            and count_tokens(out[-1].content + ch.content) <= settings.max_tokens
        ):
            prev = out[-1]
            combined = prev.content.rstrip() + "\n" + ch.content
            prefix = _context_prefix("Document", prev.heading_path)
            # 保留原 title 前缀：从 content_with_context 抽 Document 行
            doc_line = prev.content_with_context.splitlines()[0]
            title = doc_line.split(":", 1)[-1].strip() if doc_line.startswith("Document:") else ""
            ctx = _context_prefix(title, prev.heading_path) + combined
            out[-1] = ChunkRecord(
                chunk_index=0,
                parent_section=prev.parent_section,
                heading_path=prev.heading_path,
                content=combined,
                content_with_context=ctx,
                token_count=count_tokens(ctx),
                char_count=len(combined),
                start_offset=prev.start_offset,
                end_offset=ch.end_offset,
                chunk_type=prev.chunk_type if prev.chunk_type == ch.chunk_type else "mixed",
                metadata=prev.metadata,
            )
        else:
            out.append(ch)
    return out
