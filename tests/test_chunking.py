from app.ingestion.splitter import parse_blocks, split_document
from app.ingestion.tokens import count_tokens


def test_heading_stays_with_section_body():
    md = """# LangGraph

## Core Concepts

### State

State stores the shared structure that nodes read and write.
It is typically a TypedDict.

### Control Flow

Control flow is a graph of nodes and edges.
"""
    chunks = split_document(md, title="LangGraph")
    state = [c for c in chunks if c.parent_section == "State"]
    assert state
    assert "State stores" in state[0].content
    assert "Document: LangGraph" in state[0].content_with_context
    assert "Core Concepts > State" in state[0].content_with_context
    assert not any(c.content.strip() == "State stores the shared structure that nodes read and write." and "TypedDict" not in c.content for c in state)


def test_does_not_cross_h2_when_merging_short():
    md = """## Alpha

Short.

## Beta

Also short but different topic.
"""
    chunks = split_document(md, title="Doc", min_tokens=120)
    sections = {tuple(c.heading_path) for c in chunks}
    assert ("Alpha",) in sections or any("Alpha" in c.heading_path for c in chunks)
    assert any("Beta" in c.heading_path for c in chunks)
    mixed = [c for c in chunks if "Short." in c.content and "Also short" in c.content]
    assert mixed == []


def test_code_block_kept_whole_under_limit():
    md = """## Tools

```python
def add(a, b):
    return a + b
```
"""
    chunks = split_document(md, title="Doc")
    code = [c for c in chunks if c.chunk_type == "code"]
    assert code
    assert "def add" in code[0].content
    assert count_tokens(code[0].content) <= 600


def test_table_split_keeps_header():
    header = "| col1 | col2 |\n| --- | --- |\n"
    rows = "".join(f"| v{i} | x{i} |\n" for i in range(80))
    md = "## Table\n\n" + header + rows
    chunks = split_document(md, title="Doc", max_tokens=80, target_tokens=40, min_tokens=10)
    tables = [c for c in chunks if c.chunk_type == "table"]
    assert tables
    if len(tables) > 1:
        for t in tables:
            assert "col1" in t.content
            assert "col2" in t.content


def test_parse_blocks_kinds():
    md = """# T

Para.

- a
- b

> q

```js
ok
```
"""
    kinds = [b.kind for b in parse_blocks(md)]
    assert "heading" in kinds
    assert "paragraph" in kinds
    assert "list" in kinds
    assert "quote" in kinds
    assert "code" in kinds
