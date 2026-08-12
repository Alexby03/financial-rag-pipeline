from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ParsedContent:
    """Normalized parser output before chunking (see docs/ingestion_pipeline.md)."""

    text: str
    content_format: Literal["markdown", "json"]
