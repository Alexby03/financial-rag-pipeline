from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    openai_api_key: str
    openai_chat_model: str = "gpt-4o"
    openai_embedding_model: str = "text-embedding-3-small"

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


@lru_cache
def get_settings() -> Settings:
    return Settings()
