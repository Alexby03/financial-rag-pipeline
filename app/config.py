from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    # Vision model used for parsing documents during ingest.
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"
    # Reasoning model used to answer queries, so the UI can show its reasoning summary.
    openai_answer_model: str = "gpt-5-mini"
    openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"
    openai_reasoning_summary: Literal["auto", "concise", "detailed"] = "auto"

    # Cohere Rerank on Azure AI Foundry. Reranking stays off until both values are set.
    cohere_api_base: str = ""
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    rerank_candidate_multiplier: int = Field(default=4, ge=1, le=10)
    rerank_timeout_seconds: float = 10.0

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ma_documents"

    minio_endpoint: str = "localhost:9000"
    minio_access_key: str = "minioadmin"
    minio_secret_key: str = "minioadmin"
    minio_bucket: str = "rag-documents"
    minio_secure: bool = False
    # Host the browser resolves; differs from minio_endpoint once the API runs inside Docker.
    minio_public_endpoint: str = "localhost:9000"
    presigned_url_ttl_seconds: int = 900

    default_tenant_id: str = "default_tenant_test_corp_123"
    log_level: str = "INFO"

    chunk_size: int = 800
    chunk_overlap: int = 100
    embedding_dimensions: int = 1536

    @property
    def rerank_configured(self) -> bool:
        return bool(self.cohere_api_base and self.cohere_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
