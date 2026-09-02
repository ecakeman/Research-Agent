---
title: LangGraph Core Concepts
source_type: official_docs
source_name: langgraph
version: "0.1"
url: https://langchain-ai.github.io/langgraph/
project: langgraph
---

# LangGraph

LangGraph is a library for building stateful, multi-actor applications with LLMs.

## Core Concepts

### State

State stores the shared structure that nodes read and write. In LangGraph, state is typically a TypedDict or dataclass that is passed through the graph. Each node returns a partial update that is merged into the current state. Checkpointing can persist this state between steps so an interrupted run can resume.

State handling is the main difference from a traditional chain: a chain usually passes a single string or messages list forward, while LangGraph keeps a named, updatable state object.

### Control Flow

Control flow in LangGraph is expressed as a graph of nodes and edges. Conditional edges choose the next node from state. Loops are allowed. A traditional chain workflow is usually a linear sequence: prompt then model then parser, without explicit looping or branching unless you write the loop yourself.

### Checkpointing

Checkpointing saves graph state after each step. It enables pause, resume, and time-travel. Chain workflows typically do not checkpoint intermediate node state unless the application adds its own storage.

## Tool Calling

LangGraph nodes can call tools and write tool results back into state. This is useful for agent runtimes that must keep tool traces aligned with conversation state.
