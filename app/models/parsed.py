from dataclasses import dataclass, field
from typing import Literal

from app.models.ocr import OcrDocumentAnnotation


@dataclass(frozen=True)
class ParsedPage:
    """One page of extracted content, numbered from 1 as the reader sees it."""

    text: str
    page_number: int


@dataclass(frozen=True)
class ParsedFigure:
    """A figure with caption text for its own retrieval chunk."""

    page_number: int
    caption: str
    image_bytes: bytes


@dataclass(frozen=True)
class ParsedContent:
    """Normalized parser output before chunking (see docs/ingestion_pipeline.md)."""

    pages: list[ParsedPage]
    content_format: Literal["markdown", "json"] = "markdown"
    figures: list[ParsedFigure] = field(default_factory=list)
    annotation: OcrDocumentAnnotation | None = None
    # OCR multi-page statements need sticky headings/years across pages. Excel sheets
    # and SIE sections must not inherit those — each page starts clean.
    carry_page_context: bool = True
