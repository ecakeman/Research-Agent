from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    database_url: str = "postgres://research:research@127.0.0.1:5433/research"

    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_model: str = ""

    llm_fast_base_url: str = ""
    llm_fast_api_key: str = ""
    llm_fast_model: str = ""

    llm_pro_base_url: str = ""
    llm_pro_api_key: str = ""
    llm_pro_model: str = ""

    model_routing: str = "single"

    embedding_base_url: str = ""
    embedding_api_key: str = ""
    embedding_model: str = ""
    embedding_dim: int = 1024

    rerank_base_url: str = ""
    rerank_api_key: str = ""
    rerank_model: str = ""

    eval_concurrency: int = 8

    max_retrieval_rounds: int = 2
    bm25_top_k: int = 20
    vector_top_k: int = 20
    rrf_k: int = 60
    fusion_top_k: int = 20
    rerank_top_k: int = 8
    max_evidence_items: int = 8
    max_evidence_tokens: int = 3000
    token_budget: int = 8000

    target_tokens: int = 400
    max_tokens: int = 600
    min_tokens: int = 120
    overlap_tokens: int = 60


settings = Settings()
