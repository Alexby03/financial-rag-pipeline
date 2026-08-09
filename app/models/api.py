from pydantic import BaseModel, Field

from app.models.enums import Category, DocType
from app.models.metadata import ChunkMetadata


class IngestMetadata(BaseModel):
    tenant_id: str
    company_name: str
    doc_type: DocType
    category: Category
    year: int
    quarter: int | None = Field(default=None, ge=1, le=4)
    language: str = "sv"


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    object_key: str
    status: str = "completed"


class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    doc_type: DocType | None = None
    category: Category | None = None
    year: int | None = None
    quarter: int | None = Field(default=None, ge=1, le=4)
    top_k: int = Field(default=5, ge=1, le=20)
    expand_neighbors: bool = False


class SourceChunk(BaseModel):
    raw_text: str
    metadata: ChunkMetadata
    score: float | None = None


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    minio: str
