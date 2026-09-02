---
title: RAG Hybrid Retrieval Notes
source_type: official_docs
source_name: rag-notes
version: "0.1"
url: https://example.local/rag-notes
project: rag
---

# Retrieval Augmented Generation

## Hybrid Retrieval

BM25 is strong on exact tokens such as API names and error codes. Vector search is strong on paraphrases. Reciprocal Rank Fusion (RRF) merges ranked lists without calibrating scores. A common pattern is BM25 top 20 plus vector top 20, fused with rrf_k=60.

## Rerank

A reranker scores query-document pairs and keeps a smaller top-k, for example 8. Rerank is not the same as evidence grading: rerank is a relevance sort, grading asks whether a chunk can support a specific question.

## Abstention

If the knowledge base does not contain enough evidence after a bounded rewrite loop, the system should abstain instead of inventing facts.
