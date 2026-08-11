import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config import Settings
from app.models.api import IngestMetadata
from app.models.metadata import ChunkMetadata, QdrantPayload

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)


def strip_code_fence(text: str) -> str:
    """Remove a single fenced block wrapping the whole LLM response."""
    stripped = text.strip()
    match = _FENCE_PATTERN.match(stripped)
    return match.group(1).strip() if match else stripped


@dataclass
class TextChunk:
    text: str
    page_number: int


class Chunker:
    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def split_markdown(self, markdown: str, page_number: int = 1) -> list[TextChunk]:
        normalized = strip_code_fence(markdown)
        if not normalized:
            return []

        chunks: list[TextChunk] = []
        headings: dict[int, str] = {}
        pending: list[str] = []

        for block in self._split_markdown_blocks(normalized):
            level = self._heading_level(block)
            if level is not None:
                chunks.extend(self._flush(pending, headings, page_number))
                pending = []
                headings = {lvl: text for lvl, text in headings.items() if lvl < level}
                headings[level] = block.lstrip("#").strip()
                continue

            if self._is_markdown_table(block):
                chunks.extend(self._flush(pending, headings, page_number))
                pending = []
                chunks.append(
                    TextChunk(
                        text=self._with_context(block, headings),
                        page_number=page_number,
                    )
                )
                continue

            pending.append(block)

        chunks.extend(self._flush(pending, headings, page_number))
        return chunks

    def split_json(self, json_text: str, page_number: int = 1) -> list[TextChunk]:
        normalized = json_text.strip()
        if not normalized:
            return []
        if len(normalized) <= self._chunk_size:
            return [TextChunk(text=normalized, page_number=page_number)]
        return self._split_plain_text(normalized, page_number)

    def split_text(self, text: str, page_number: int = 1) -> list[TextChunk]:
        return self.split_markdown(text, page_number)

    def build_payloads(
        self,
        chunks: list[TextChunk],
        ingest_metadata: IngestMetadata,
        source_filename: str,
        object_key: str,
    ) -> list[QdrantPayload]:
        if not chunks:
            return []

        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        processed_at = datetime.now(UTC)

        payloads: list[QdrantPayload] = []
        for index, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                tenant_id=ingest_metadata.tenant_id,
                company_name=ingest_metadata.company_name,
                doc_type=ingest_metadata.doc_type,
                category=ingest_metadata.category,
                year=ingest_metadata.year,
                quarter=ingest_metadata.quarter,
                source_filename=source_filename,
                object_key=object_key,
                page_number=chunk.page_number,
                chunk_id=chunk_ids[index],
                previous_chunk_id=chunk_ids[index - 1] if index > 0 else None,
                next_chunk_id=chunk_ids[index + 1] if index < len(chunks) - 1 else None,
                language=ingest_metadata.language,
                processed_timestamp=processed_at,
            )
            payloads.append(QdrantPayload(raw_text=chunk.text, metadata=metadata))
        return payloads

    def _flush(
        self,
        pending: list[str],
        headings: dict[int, str],
        page_number: int,
    ) -> list[TextChunk]:
        if not pending:
            return []
        return [
            TextChunk(text=self._with_context(chunk.text, headings), page_number=page_number)
            for chunk in self._split_plain_text("\n\n".join(pending), page_number)
        ]

    @staticmethod
    def _with_context(text: str, headings: dict[int, str]) -> str:
        if not headings:
            return text
        breadcrumb = " > ".join(headings[level] for level in sorted(headings))
        return f"{breadcrumb}\n\n{text}"

    @staticmethod
    def _heading_level(block: str) -> int | None:
        if "\n" in block:
            return None
        match = re.match(r"^(#{1,6})\s+\S", block)
        return len(match.group(1)) if match else None

    def _split_plain_text(self, text: str, page_number: int) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        start = 0
        text_len = len(text)

        while start < text_len:
            end = min(start + self._chunk_size, text_len)
            if end < text_len:
                paragraph_break = text.rfind("\n\n", start, end)
                if paragraph_break > start + self._chunk_size // 2:
                    end = paragraph_break
            chunk_text = text[start:end].strip()
            if chunk_text:
                chunks.append(TextChunk(text=chunk_text, page_number=page_number))
            if end >= text_len:
                break
            start = max(end - self._chunk_overlap, start + 1)

        return chunks

    @staticmethod
    def _split_markdown_blocks(markdown: str) -> list[str]:
        lines = markdown.splitlines()
        blocks: list[str] = []
        current: list[str] = []
        in_table = False

        for line in lines:
            is_table_line = "|" in line and line.strip()
            if is_table_line:
                if not in_table and current:
                    blocks.append("\n".join(current).strip())
                    current = []
                in_table = True
                current.append(line)
                continue

            if in_table:
                blocks.append("\n".join(current).strip())
                current = []
                in_table = False

            if line.strip() == "":
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
                continue

            current.append(line)

        if current:
            blocks.append("\n".join(current).strip())

        return [block for block in blocks if block]

    @staticmethod
    def _is_markdown_table(block: str) -> bool:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        pipe_lines = sum(1 for line in lines if "|" in line)
        return pipe_lines >= 2 and pipe_lines / len(lines) >= 0.5

    @staticmethod
    def new_document_id() -> str:
        return str(uuid.uuid4())
