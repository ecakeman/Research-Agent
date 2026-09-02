# Official sources fetched 2026-09-02. Ingest reads this directory; this file is skipped.

## Downloaded

| Local file | Canonical URL | Fetch URL | Notes |
|---|---|---|---|
| langgraph-overview.md | https://docs.langchain.com/oss/python/langgraph/overview | .../overview.md | Official markdown index page |
| langgraph-graph-api.md | https://docs.langchain.com/oss/python/langgraph/graph-api | .../graph-api.md | State / Node / Edge |
| langgraph-use-graph-api.md | https://docs.langchain.com/oss/python/langgraph/use-graph-api | .../use-graph-api.md | |
| langgraph-quickstart.md | https://docs.langchain.com/oss/python/langgraph/quickstart | .../quickstart.md | |
| langchain-overview.md | https://docs.langchain.com/oss/python/langchain/overview | .../overview.md | |
| langchain-quickstart.md | https://docs.langchain.com/oss/python/langchain/quickstart | .../quickstart.md | |
| pgvector-readme.md | https://github.com/pgvector/pgvector | raw GitHub README | |
| qwen-function-calling.md | https://qwen.readthedocs.io/en/stable/framework/function_call.html | GitHub Qwen2.5 docs `function_call.md` | |

## Fallback

- LangChain / LangGraph HTML at docs.langchain.com is a JS app (large shell). Used the site's official `.md` export instead of HTML scrape.
- Qwen: readthedocs HTML exists, but body is mixed with Sphinx chrome. Used official GitHub source `QwenLM/Qwen2.5/docs/source/framework/function_call.md` (same Function Calling doc).

## Skipped

- None of the requested pages after fallback.

`make ingest` reads `data/documents/*.md` and skips this file.
