from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    memory_api_key: str = Field(min_length=16)
    supabase_url: str = "https://example.supabase.co"
    supabase_key: str = "anon-key"
    storage_rpc_token: str = Field(
        default="storage-token-for-default-config-123456", min_length=32
    )
    dashscope_api_key: str = ""
    llm_api_key: str = ""
    sqlite_path: str = "dandelion_v2.sqlite3"
    embedding_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    embedding_model: str = "text-embedding-v4"
    embedding_dimensions: int = 1024
    embedding_batch_size: int = Field(default=10, ge=1, le=10)
    llm_base_url: str = "https://dashscope.aliyuncs.com/compatible-mode/v1"
    llm_model: str = "qwen3.5-flash"
    llm_timeout: float = 35.0
    add_extract_budget: float = 35.0
    add_total_budget: float = 75.0
    rerank_base_url: str = (
        "https://dashscope.aliyuncs.com/api/v1/services/rerank/text-rerank/text-rerank"
    )
    rerank_model: str = "qwen3-rerank"
    rerank_timeout: float = 4.0
    semantic_candidates: int = Field(default=150, ge=1, le=1000)
    lexical_candidates: int = Field(default=150, ge=1, le=1000)
    neighbor_anchors: int = Field(default=20, ge=0, le=100)
    neighbor_radius: int = Field(default=1, ge=0, le=3)
    answer_token_budget: int = Field(default=100_000, ge=1)
    chunk_target_chars: int = Field(default=1200, ge=100)
    chunk_max_chars: int = Field(default=1800, ge=200)


@lru_cache
def get_settings() -> Settings:
    return Settings()  # type: ignore[call-arg]
