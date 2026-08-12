from functools import lru_cache

from app.config import Settings, get_settings
from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.ingest import IngestService
from app.services.minio_client import MinioStorage
from app.services.openai_client import OpenAIService
from app.services.parser import DocumentParser
from app.services.qdrant_service import QdrantService
from app.services.rag import RagService
from app.services.reranker import CohereReranker


@lru_cache
def get_minio_storage() -> MinioStorage:
    return MinioStorage(get_settings())


@lru_cache
def get_qdrant_service() -> QdrantService:
    return QdrantService(get_settings())


@lru_cache
def get_openai_service() -> OpenAIService:
    return OpenAIService(get_settings())


@lru_cache
def get_embedder() -> Embedder:
    return Embedder(get_openai_service())


@lru_cache
def get_parser() -> DocumentParser:
    return DocumentParser(get_openai_service())


@lru_cache
def get_chunker() -> Chunker:
    return Chunker(get_settings())


@lru_cache
def get_ingest_service() -> IngestService:
    return IngestService(
        storage=get_minio_storage(),
        parser=get_parser(),
        chunker=get_chunker(),
        embedder=get_embedder(),
        qdrant=get_qdrant_service(),
    )


@lru_cache
def get_reranker() -> CohereReranker | None:
    settings = get_settings()
    return CohereReranker(settings) if settings.rerank_configured else None


@lru_cache
def get_rag_service() -> RagService:
    return RagService(
        embedder=get_embedder(),
        qdrant=get_qdrant_service(),
        openai_service=get_openai_service(),
        settings=get_settings(),
        reranker=get_reranker(),
    )


def get_app_settings() -> Settings:
    return get_settings()
