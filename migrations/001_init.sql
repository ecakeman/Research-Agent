CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE IF NOT EXISTS documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    source_type TEXT NOT NULL,
    source_name TEXT NOT NULL,
    title TEXT NOT NULL,
    section TEXT,
    url TEXT,
    version TEXT,
    doc_key TEXT NOT NULL,
    published_at TIMESTAMPTZ,
    content TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (source_name, version, doc_key)
);

CREATE TABLE IF NOT EXISTS chunks (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    document_id UUID NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_index INT NOT NULL,
    parent_section TEXT,
    heading_path JSONB NOT NULL,
    content TEXT NOT NULL,
    content_with_context TEXT NOT NULL,
    token_count INT NOT NULL,
    char_count INT NOT NULL,
    start_offset INT NOT NULL,
    end_offset INT NOT NULL,
    chunk_type TEXT NOT NULL,
    metadata JSONB NOT NULL DEFAULT '{}',
    search_tsv tsvector,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS chunks_search_tsv_idx ON chunks USING GIN (search_tsv);
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);

CREATE TABLE IF NOT EXISTS chunk_embeddings (
    chunk_id UUID PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    embedding vector,
    model TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research_runs (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    query TEXT NOT NULL,
    status TEXT NOT NULL,
    retrieval_rounds INT NOT NULL DEFAULT 0,
    final_answer TEXT,
    failure_reason TEXT,
    citations JSONB NOT NULL DEFAULT '[]',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    completed_at TIMESTAMPTZ
);

CREATE TABLE IF NOT EXISTS research_steps (
    id BIGSERIAL PRIMARY KEY,
    run_id UUID NOT NULL REFERENCES research_runs(id) ON DELETE CASCADE,
    step_index INT NOT NULL,
    node TEXT NOT NULL,
    input JSONB NOT NULL,
    output JSONB NOT NULL,
    duration_ms INT,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (run_id, step_index)
);

CREATE TABLE IF NOT EXISTS evaluation_cases (
    id TEXT PRIMARY KEY,
    category TEXT NOT NULL,
    question TEXT NOT NULL,
    expected_sources JSONB NOT NULL DEFAULT '[]',
    expected_claims JSONB NOT NULL DEFAULT '[]',
    metadata JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS evaluation_results (
    id BIGSERIAL PRIMARY KEY,
    case_id TEXT NOT NULL,
    baseline TEXT NOT NULL,
    retrieval_recall_5 DOUBLE PRECISION,
    retrieval_recall_10 DOUBLE PRECISION,
    mrr DOUBLE PRECISION,
    ndcg_10 DOUBLE PRECISION,
    groundedness DOUBLE PRECISION,
    citation_precision DOUBLE PRECISION,
    citation_recall DOUBLE PRECISION,
    abstention_expected BOOLEAN,
    abstention_actual BOOLEAN,
    retrieval_rounds INT,
    latency_ms INT,
    input_tokens INT,
    output_tokens INT,
    model_calls INT,
    status TEXT NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);
