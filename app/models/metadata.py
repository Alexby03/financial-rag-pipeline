from datetime import datetime

from pydantic import BaseModel, Field

from app.models.enums import Category, DocType


class ChunkMetadata(BaseModel):
    tenant_id: str
    company_name: str
    doc_type: DocType
    category: Category
    year: int
    quarter: int | None = Field(default=None, ge=1, le=4)
    source_filename: str
    object_key: str
    # Optional so chunks stored before content hashing keep validating on read.
    content_hash: str | None = None
    page_number: int
    chunk_id: str
    previous_chunk_id: str | None
    next_chunk_id: str | None
    language: str
    processed_timestamp: datetime


class QdrantPayload(BaseModel):
    raw_text: str
    metadata: ChunkMetadata

    def to_qdrant_dict(self) -> dict[str, object]:
        return {
            "raw_text": self.raw_text,
            "metadata": self.metadata.model_dump(mode="json"),
        }
