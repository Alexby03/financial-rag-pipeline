from typing import Literal

from pydantic import BaseModel, Field

from app.models.enums import Category, DocType, Language
from app.models.metadata import ChunkMetadata

IngestJobStatus = Literal["queued", "running", "completed", "failed", "duplicate"]
IngestJobStage = Literal["checking", "uploading", "parsing", "embedding", "indexing"]


class IngestMetadata(BaseModel):
    tenant_id: str
    company_name: str | None = None
    doc_type: DocType | None = None
    category: Category | None = None
    year: int | None = None
    quarter: int | None = Field(default=None, ge=1, le=4)
    # None means Auto — filled from OCR / parser annotation at ingest time.
    language: Language | None = None


class IngestResponse(BaseModel):
    document_id: str
    chunk_count: int
    object_key: str
    status: str = "completed"


class IngestJobAccepted(BaseModel):
    """Immediate reply from POST /ingest — work continues in the lifespan worker."""

    job_id: str
    status: Literal["queued"] = "queued"


class IngestJobStatusResponse(BaseModel):
    job_id: str
    status: IngestJobStatus
    stage: IngestJobStage | None = None
    document_id: str = ""
    chunk_count: int = 0
    object_key: str = ""
    error: str | None = None


class ChatTurn(BaseModel):
    """One prior turn in the browser session. Not persisted server-side."""

    role: Literal["user", "assistant"]
    content: str


class QueryRequest(BaseModel):
    question: str
    tenant_id: str
    company_name: str | None = None
    doc_type: DocType | None = None
    category: Category | None = None
    year: int | None = None
    quarter: int | None = Field(default=None, ge=1, le=4)
    top_k: int = Field(default=5, ge=1, le=20)
    expand_neighbors: bool = False
    rerank: bool = True
    # Prior turns from the Streamlit tab; dropped when the tab reloads.
    chat_history: list[ChatTurn] = Field(default_factory=list)


class SourceChunk(BaseModel):
    raw_text: str
    metadata: ChunkMetadata
    score: float | None = None
    rerank_score: float | None = None


class RetrievedChunkTrace(BaseModel):
    chunk_id: str
    source_filename: str
    page_number: int
    score: float
    preview: str
    vector_rank: int
    rerank_score: float | None = None
    lexical_rank: int | None = None
    rrf_score: float | None = None


class RerankTrace(BaseModel):
    model: str
    candidates: int
    applied: bool
    fallback_reason: str = ""
    rerank_ms: int = 0


class FusionTrace(BaseModel):
    """How dense / lexical / Cohere rankings were combined."""

    method: str = "rrf"
    lexical_applied: bool = False
    cohere_applied: bool = False
    candidates: int = 0
    rrf_k: int = 60


class FigureReadingTrace(BaseModel):
    chunk_id: str
    source_filename: str
    page_number: int
    applied: bool
    reading: str = ""
    preview: str = ""
    error: str = ""


class RetrievalTrace(BaseModel):
    """What the agent actually did to answer, surfaced to the user for transparency."""

    embedding_model: str
    answer_model: str
    filters: dict[str, str | int] = Field(default_factory=dict)
    top_k: int
    # Question actually embedded / BM25'd / reranked (prior user turns + current).
    retrieval_question: str = ""
    hits: list[RetrievedChunkTrace] = Field(default_factory=list)
    neighbor_chunk_ids: list[str] = Field(default_factory=list)
    context_chunk_count: int = 0
    retrieval_ms: int = 0
    generation_ms: int = 0
    vision_ms: int = 0
    vision_model: str = ""
    reasoning_summary: str = ""
    rerank: RerankTrace | None = None
    fusion: FusionTrace | None = None
    figure_readings: list[FigureReadingTrace] = Field(default_factory=list)


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
