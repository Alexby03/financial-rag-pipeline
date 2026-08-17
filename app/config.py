from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_embedding_model: str = "text-embedding-3-small"
    # Reasoning model used to answer queries, so the UI can show its reasoning summary.
    openai_answer_model: str = "gpt-5-mini"
    openai_reasoning_effort: Literal["minimal", "low", "medium", "high"] = "medium"
    openai_reasoning_summary: Literal["auto", "concise", "detailed"] = "auto"
    # Reads retrieved figure crops before the reasoning model answers. Blank disables.
    # Prefer gpt-4o over mini for stacked charts and axis interpolation.
    openai_vision_model: str = "gpt-4o"

    # Cohere Rerank on Azure AI Foundry. Reranking stays off until both values are set.
    cohere_api_base: str = ""
    cohere_api_key: str = ""
    cohere_rerank_model: str = "rerank-multilingual-v3.0"
    rerank_candidate_multiplier: int = Field(default=4, ge=1, le=10)
    rerank_timeout_seconds: float = 10.0
    # Late lexical BM25 over the dense candidate pool, fused with RRF (and Cohere when on).
    hybrid_lexical: bool = True
    rrf_k: int = Field(default=60, ge=1, le=200)

    # Mistral OCR 4 on Azure AI Foundry. Required for PDF/Office/image ingest.
    mistral_ocr_endpoint: str = ""
    mistral_ocr_api_key: str = ""
    mistral_ocr_model: str = "mistral-ocr-4-0"
    mistral_ocr_timeout_seconds: float = Field(default=300.0, ge=30.0, le=1800.0)
    # Foundry OCR 4 rejects the whole PDF above this page count (error 3730).
    mistral_ocr_max_pages: int = Field(default=30, ge=1, le=1000)

    qdrant_host: str = "localhost"
    qdrant_port: int = 6333
    qdrant_collection: str = "ma_documents"

    redis_url: str = "redis://localhost:6379/0"
    redis_ingest_queue_key: str = "ingest:jobs"

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

    @property
    def mistral_ocr_configured(self) -> bool:
        return bool(self.mistral_ocr_endpoint and self.mistral_ocr_api_key)


@lru_cache
def get_settings() -> Settings:
    return Settings()
