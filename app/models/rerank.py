from pydantic import BaseModel, Field


class RerankedDocument(BaseModel):
    index: int
    relevance_score: float


class RerankOutcome(BaseModel):
    """Result of a rerank attempt; `applied` is False when we fell back to vector order."""

    applied: bool
    documents: list[RerankedDocument] = Field(default_factory=list)
    fallback_reason: str = ""
