# Research Agent

面向技术资料的可验证研究助手。

Research Agent 将 **Hybrid Retrieval、Evidence Grading、Query Rewrite、Citation Validation** 组织成一个有状态的 Agentic RAG Workflow，针对技术研究问题进行检索、证据判断、有限恢复与可追溯回答。

核心目标不是“生成一个看起来合理的答案”，而是：

> **基于检索到的证据进行回答；证据不足时停止生成。**

---

## Overview

普通 RAG：

```text
Query
  ↓
Retrieve
  ↓
Generate
```

Research Agent：

```text
Query
  ↓
Analyze
  ↓
Retrieve
  ↓
Rerank
  ↓
Grade Evidence
  ↓
Sufficient?
 ├─ Yes → Compress → Answer
 └─ No  → Rewrite → Retrieve again
                    ↓
                  Grade
  ↓
Citation Validation
  ↓
Final / Abstain
```

整个研究过程由 LangGraph 编排，并通过显式 State 在节点之间传递研究上下文。

---

## Core Workflow

### 1. Query Analysis

首先对研究问题进行结构化分析，提取：

* `intent`
* `entities`
* `sub_questions`

复杂问题会被拆分成多个可验证的子问题，为后续 evidence coverage 提供依据。

### 2. Hybrid Retrieval

同时使用：

```text
BM25 + Vector Search
```

BM25 基于 PostgreSQL FTS 与 Okapi BM25。

Vector Search 使用 PostgreSQL + pgvector。

两路结果通过 **Reciprocal Rank Fusion (RRF)** 合并，再交给 Reranker 进行最终候选筛选。

```text
BM25 Top 20
        +
Vector Top 20
        ↓
RRF (k=60)
        ↓
Top 20
        ↓
Rerank
        ↓
Top 8
```

### 3. Evidence Grading

Rerank 后的候选证据由 Fast Model 判断其对具体 sub-question 的支持程度：

```text
direct
partial
weak
```

模型只负责判断 evidence，是否满足整体回答条件由代码决定。

### 4. Evidence Sufficiency

系统不会让模型直接决定“证据够不够”。

当前策略：

```text
directly covered sub_questions
────────────────────────────── ≥ 2 / 3
      total sub_questions
```

达到阈值后进入回答，否则进入有限次 Recovery。

### 5. Query Rewrite

证据不足时：

```text
Evidence Gap
     ↓
Query Rewrite
     ↓
Retrieve Again
     ↓
Rerank
     ↓
Grade Again
```

最多执行两轮 Retrieval，避免 Agent 无限循环。

### 6. Evidence Compression

通过 Fast Model 将最终证据整理成更适合 Answer 阶段消费的上下文，减少无关内容进入最终生成。

### 7. Citation Validation

回答生成后进行确定性 Citation Validation。

Citation 必须满足：

```text
chunk_id ∈ current run evidence
quote ∈ corresponding chunk
```

非法 citation 会触发有限次重新生成；仍无法通过则终止，而不是伪造来源。

### 8. Abstention

两轮检索后仍无法获得充分证据：

```text
→ insufficient_evidence
→ Abstain
```

系统选择停止，而不是补充未经证实的信息。

---

## Model Routing

支持两种模式：

```text
MODEL_ROUTING=single
MODEL_ROUTING=dual
```

### Single

所有节点使用同一模型：

```text
Analyze
Grade
Rewrite
Compress
Answer
Citation Retry
```

### Dual

根据任务复杂度进行模型分工：

```text
Analyze            → Pro
Grade              → Fast
Rewrite            → Fast
Compress           → Fast
Answer             → Pro
Citation Retry 1   → Fast
Citation Retry 2   → Pro
```

Graph Node 只依赖 `ModelRole`，模型配置由统一的 `ModelRouter` 管理。

Dual 模式缺失必要模型配置时直接报错，不静默 fallback。

---

## Architecture

```text
                    ┌──────────────┐
                    │    Query     │
                    └──────┬───────┘
                           ↓
                    ┌──────────────┐
                    │    Analyze   │
                    └──────┬───────┘
                           ↓
                ┌─────────────────────┐
                │   Hybrid Retrieval  │
                │                     │
                │   BM25 + Vector     │
                └──────────┬──────────┘
                           ↓
                         RRF
                           ↓
                       Rerank
                           ↓
                    Evidence Grade
                           ↓
                   ┌───────┴────────┐
                   │   Sufficient?  │
                   └───────┬────────┘
                     Yes   │   No
                      ↓    │
                  Compress │ Rewrite
                      ↓    │
                    Answer │ Retrieve
                      ↓    │
                Citation   │ Grade
                Validation │
                     ↑     │
                     └─────┘
```

关键控制逻辑由应用代码负责：

```text
Retrieval Fusion
Evidence Sufficiency
Retry Limit
Citation Validation
Abstention
Model Routing
```

模型负责理解、判断和生成，不承担整个 Workflow 的控制权。

---

## Evaluation

项目提供独立 Retrieval Evaluation 与 Agentic Evaluation。

### Retrieval Baseline

在当前 evaluation set 上：

| Pipeline        | Recall@5 | Recall@10 |  MRR |  nDCG@10 |
| --------------- | -------: | --------: | ---: | -------: |
| Vector          |     0.95 |      0.95 | 0.99 |     0.95 |
| Hybrid          | **0.99** |  **0.99** | 0.99 | **0.98** |
| Hybrid + Rerank |     0.93 |      0.93 | 0.99 |     0.94 |

### Agentic Quality

Live evaluation 中，已完成回答表现为：

```text
Groundedness
    1.00

Citation Precision
    1.00
```

Single routing 的 evidence recovery：

```text
4 / 11
36.36%
```

该指标表示首轮 evidence 不足、且问题具备有效答案时，经 Query Rewrite 后成功恢复的比例。

---

## Quick Start

### Requirements

```text
Python 3.12+
PostgreSQL
pgvector
uv
```

### Install

```bash
uv sync
```

### Configure

```bash
cp .env.example .env
```

配置数据库、Embedding、Rerank 与模型服务。

### Start Database

```bash
make db-up
make migrate
```

### Ingest Documents

```bash
make ingest
```

默认从：

```text
data/documents/
```

读取技术资料。

同一 `source + version + path` 的文档重复 ingest 时会自动跳过。

### Ask

```bash
research ask "What is LangGraph State?"
```

---

## Evaluation

Retrieval：

```bash
make eval BASELINE=vector
make eval BASELINE=hybrid
make eval BASELINE=rerank
```

Agentic：

```bash
make eval BASELINE=agentic MODEL_ROUTING=single
make eval BASELINE=agentic MODEL_ROUTING=dual
```

独立 Query 支持受控并发：

```bash
make eval BASELINE=agentic MODEL_ROUTING=dual CONCURRENCY=8
```

单个 Research Run 仍按照 Graph 的数据依赖顺序执行。

---

## Testing

```bash
uv run pytest -q
make golden
```

Golden tests 不调用真实模型 API，用于验证核心 Workflow 与控制逻辑。

---

## Tech Stack

```text
Python
FastAPI
LangChain
LangGraph
PostgreSQL
pgvector
BM25
RRF
Rerank HTTP
tiktoken
```

---

## Project Structure

```text
app/
├── graph/          LangGraph workflow
├── models/         HTTP clients + ModelRouter
├── retrieval/      BM25 / Vector / RRF / Rerank
├── ingestion/      Markdown ingestion + chunking
├── generation/     prompts / compression / citations
└── evaluation/     metrics / runner / live evaluation

data/
└── documents/      Technical documents

eval/
└── questions.jsonl Evaluation questions

tests/
└── golden/         Deterministic golden cases
```

---

## Design Principles

### Evidence over Fluency

回答必须建立在检索证据之上，而不是依赖模型自身知识补全。

### Code over Guessing

关键控制逻辑由确定性代码执行：

```text
Sufficiency
Citation Validation
Retry Limits
Routing
Abstention
```

### Bounded Agent Loop

Recovery 有明确上限：

```text
Maximum Retrieval Rounds = 2
```

避免不可控的 Agent 循环与成本增长。

### Explicit Model Routing

不同任务根据复杂度选择不同模型，降低不必要的高成本调用。

---

## Status

```text
Core Workflow        ✓
Hybrid Retrieval     ✓
RRF                  ✓
Rerank               ✓
Evidence Grading     ✓
Sufficiency Gate     ✓
Query Rewrite        ✓
Evidence Compression ✓
Citation Validation  ✓
Abstention           ✓
Single / Dual        ✓
Evaluation           ✓
Golden Tests         ✓
```

**Research Agent V1**
