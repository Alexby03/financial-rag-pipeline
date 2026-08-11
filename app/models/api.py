from typing import Literal

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


class RetrievedChunkTrace(BaseModel):
    chunk_id: str
    source_filename: str
    page_number: int
    score: float
    preview: str


class RetrievalTrace(BaseModel):
    """What the agent actually did to answer, surfaced to the user for transparency."""

    embedding_model: str
    answer_model: str
    filters: dict[str, str | int] = Field(default_factory=dict)
    top_k: int
    hits: list[RetrievedChunkTrace] = Field(default_factory=list)
    neighbor_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_count: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    reasoning_summary: str = ""


class QueryResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]
    trace: RetrievalTrace


class SourceUrlResponse(BaseModel):
    url: str
    filename: str
    disposition: Literal["inline", "attachment"]
    expires_in_seconds: int


class HealthResponse(BaseModel):
    status: str
    qdrant: str
    minio: str
