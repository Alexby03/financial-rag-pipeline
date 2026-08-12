from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ParsedPage:
    """One page of extracted content, numbered from 1 as the reader sees it."""

    text: str
    page_number: int


@dataclass(frozen=True)
class ParsedContent:
    """Normalized parser output before chunking (see docs/ingestion_pipeline.md)."""

    pages: list[ParsedPage]
    content_format: Literal["markdown", "json"]
