# Research Agent

面向技术资料的可验证研究助手。

用户提出技术研究问题后，Agent 做结构化任务分析、混合检索、证据判断、有限次 Query Rewrite 和 Citation Validation，生成可追溯的研究结果；证据不足时主动 Abstain。

这不是普通 RAG QA。核心是研究流程：分析 → 检索 → 证据判定 → 必要时重检索 → 引用校验。

## Why

技术问题需要可核对的来源，而不是流畅的猜测。LangGraph 适合用 State / Node / Edge / conditional workflow 编排这条有状态研究环；LangChain 负责 Prompt、Document 与 Retriever 抽象；检索融合与 sufficiency 判定留在明确的代码里。

## Architecture

```text
Query
  ↓
Analyze (Pro)
  ↓
Hybrid Retrieval (BM25 + Vector)
  ↓
RRF
  ↓
Rerank
  ↓
Evidence Grade (Fast)
  ↓
Sufficient?   (direct 覆盖 sub_questions ≥ 2/3，代码判定)
 ├─ Yes → Compress (Fast) → Answer (Pro)
 └─ No  → Rewrite (Fast) → Retrieve again → Grade
                              ↓
                         Max 2 rounds
  ↓
Citation Validation（确定性）
  ↓
Final / Abstain
```

最多 2 轮 retrieval。非法 citation：Fast 再生成一次，仍非法则 Pro 再生成一次，再失败则 `failed`，不伪造来源。

## Model Routing

`MODEL_ROUTING=single|dual`。Node 只声明 `ModelRole`，不读环境变量。

**Single**：Analyze / Grade / Rewrite / Compress / Answer / Citation retry 共用 `LLM_*`。

**Dual**：

```text
Analyze            → Pro
Grade              → Fast
Rewrite            → Fast
Compress           → Fast
Answer (首次)      → Pro
Citation retry 1   → Fast
Citation retry 2   → Pro
```

Dual 缺 Fast 或 Pro 配置会直接报错，不静默 fallback。

## Retrieval

Keyword BM25（Postgres FTS 候选 + Okapi）与 Vector（pgvector）各 top 20，RRF（k=60）融合 top 20，再 Rerank 到 8。

## Evidence & Citation

- Sufficiency：被 `direct` 覆盖的 sub_question 比例 ≥ 2/3
- Citation：`chunk_id` 必须属于当前 run 的 evidence；quote 必须是 chunk 子串
- 两轮后仍不足：Abstain（`insufficient_evidence`）

## Evaluation

Baseline：Vector / Hybrid / Hybrid+Rerank / Agentic+Single / Agentic+Dual

```bash
make eval BASELINE=vector
make eval BASELINE=hybrid
make eval BASELINE=rerank
make eval BASELINE=agentic MODEL_ROUTING=single
make eval BASELINE=agentic MODEL_ROUTING=dual
make eval BASELINE=agentic MODEL_ROUTING=dual CONCURRENCY=8
```

Evaluation supports controlled concurrent execution across independent queries.

A single Research Run remains sequential because its graph nodes have data dependencies.
Different evaluation queries may execute concurrently.

Example:

```bash
make eval BASELINE=agentic MODEL_ROUTING=dual CONCURRENCY=8
```

指标接口：Recall@5 / Recall@10 / MRR / nDCG@10（**source-level**：`expected_sources` 对 `documents.source_name`）；Generation：Groundedness、Citation Precision/Recall 仅 **completed**；Abstention 与 Failed 分列；Agentic rewrite recovery 用 Graph 记录的 `first_pass_evidence_sufficient`。

非 live 的 `make eval` 对已 ingest 的库做检索层打分（`expected_sources` 对 `documents.source_name`），不调 LLM。Retrieval metrics are evaluated at source level. Generation 与 rewrite_* 在未跑 `--live` 时为 **N/A**。`BASELINE=agentic` 非 live 与 hybrid+rerank 检索相同。Rerank HTTP 失败则退出，不编 Recall。空数据集时打印 `Evaluation dataset not ready`。

数字追溯：`eval/results/vector.json`、`hybrid.json`、`rerank.json`、`live_dual.json`、`live_single.json`。并发 `CONCURRENCY=8`。先前几次手工 `research ask` 与未完成的串行 live 只留下 `research_runs` 历史行，不改变 Graph 或知识库。

### Retrieval Evaluation

Pipeline | Recall@5 | Recall@10 | MRR | nDCG@10 | wall s
---|---:|---:|---:|---:|---:
Vector | 0.95 | 0.95 | 0.99 | 0.95 | 3.43
Hybrid | 0.99 | 0.99 | 0.99 | 0.98 | 4.09
Hybrid + Rerank | 0.93 | 0.93 | 0.99 | 0.94 | 6.77
Agentic + Dual (live, 成功题的检索) | 0.94 | 0.94 | 0.99 | 0.94 | 1052.78
Agentic + Single (live) | N/A | N/A | N/A | N/A | 6.61

### Generation Evaluation（Agentic live）

| | Dual | Single |
|---|---:|---:|
| total / completed / abstained / failed | 60 / 24 / 33 / 3 | 60 / 0 / 0 / 60 |
| Abstention rate / Failure rate | 0.55 / 0.05 | N/A / 1.00 |
| Groundedness (completed) | 1.00 | N/A |
| Citation Precision (completed) | 1.00 | N/A |
| Citation Recall (completed) | 0.98 | N/A |
| Correct Abstention | 1.00 | N/A |
| first_pass_insufficient | 34 | N/A |
| eligible_for_recovery | 25 | N/A |
| rewrite_attempted / recovered / rate | 34 / 1 / 0.04 | N/A |
| Total Tokens | 953148 | 0 |
| Pro Calls / Fast Calls | 81 / 150 | 0 / 0 |

Single 60/60 为上游 `402 Payment Required`，**不按 0.00 记分**。Dual citation 只在 completed 上平均。rewrite recovery = recovered / eligible（首轮不足且 gold 可答）。Dual 失败 3 题未计入平均分。

## Tech Stack

Python, FastAPI, LangChain, LangGraph, PostgreSQL, pgvector, BM25, Rerank HTTP, tiktoken

## Project Structure

```text
app/graph          LangGraph workflow
app/models         HTTP clients + ModelRouter
app/retrieval      BM25 / vector / RRF / rerank
app/ingestion      markdown chunking
app/generation     prompts / citations
app/evaluation     metrics / runner
data/raw           早期样本（ingest 不再读这里）
data/documents     官方资料；`make ingest` 读这里，跳过 SOURCES.md
eval/questions.jsonl  60 条 gold（fact/comparison/multi-hop/insufficient/ambiguous）
tests/golden       G1–G7
```

## Quick Start

```bash
uv sync
cp .env.example .env
make db-up
make migrate
make ingest
research ask "What is LangGraph State?"
```

`make ingest` 读 `data/documents/`（跳过 `SOURCES.md`）。同 source+version+path 再次 ingest 会 skip。

## Testing

```bash
uv run pytest -q
make golden
```

Golden 不调用真实 API。

## Failure Cases

- 两轮检索仍无足够证据 → abstain（live Dual：Desk HITL 问句 `abstained`，步骤含 rewrite）
- 幻觉 citation → Fast retry → Pro retry → failed（Golden G3；本次 Dual live 未自然打到该路径）
- 第一轮不足、rewrite 后 recovered（F4 / G4）；Dual live `rewrite_recovery_rate=0.03`
- live 单题 JSON 非对象 / 上游 402 → case `failed`，整批继续

## Status

- Core implementation complete
- Dual routing complete
- Official docs ingested
- Evaluation dataset: 60 questions
- Evaluation runner: controlled concurrency (default 8)
- Retrieval eval recorded
- Agentic Dual live recorded; Agentic Single live blocked by API 402
# Research-Agent
