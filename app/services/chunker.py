import re
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime

from app.config import Settings
from app.models.api import IngestMetadata
from app.models.enums import Category, DocType
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.models.parsed import ParsedPage

_FENCE_PATTERN = re.compile(r"^```[a-zA-Z0-9_-]*\s*(.*?)\s*```$", re.DOTALL)
# Balance-sheet / income-statement column dates from Bolagsverket-style reports.
_ISO_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_AMOUNT_TOKEN = re.compile(
    r"^-?\d{1,3}(?:[ \u00a0]\d{3})*(?:[.,]\d+)?$|^-?\d+(?:[.,]\d+)?$"
)


def strip_code_fence(text: str) -> str:
    """Remove a single fenced block wrapping the whole LLM response."""
    stripped = text.strip()
    match = _FENCE_PATTERN.match(stripped)
    return match.group(1).strip() if match else stripped


@dataclass
class TextChunk:
    text: str
    page_number: int
    figure_object_key: str | None = None


class Chunker:
    def __init__(self, settings: Settings) -> None:
        self._chunk_size = settings.chunk_size
        self._chunk_overlap = settings.chunk_overlap

    def split_pages(self, pages: list[ParsedPage]) -> list[TextChunk]:
        """Chunk a document page by page, carrying headings across page breaks."""
        chunks: list[TextChunk] = []
        headings: dict[int, str] = {}
        column_periods: str | None = None
        for page in pages:
            page_chunks, headings, column_periods = self._split_page(
                page.text,
                page.page_number,
                headings,
                column_periods,
            )
            chunks.extend(page_chunks)
        return chunks

    def split_markdown(self, markdown: str, page_number: int = 1) -> list[TextChunk]:
        chunks, _, _ = self._split_page(markdown, page_number, {}, None)
        return chunks

    def _split_page(
        self,
        markdown: str,
        page_number: int,
        inherited_headings: dict[int, str],
        inherited_column_periods: str | None,
    ) -> tuple[list[TextChunk], dict[int, str], str | None]:
        normalized = strip_code_fence(markdown)
        if not normalized:
            return [], inherited_headings, inherited_column_periods

        chunks: list[TextChunk] = []
        headings = dict(inherited_headings)
        column_periods = inherited_column_periods
        pending: list[str] = []

        for block in self._split_markdown_blocks(normalized):
            periods = self._column_periods_from_block(block)
            if periods:
                column_periods = periods

            level = self._heading_level(block)
            if level is not None:
                chunks.extend(
                    self._flush(pending, headings, page_number, column_periods)
                )
                pending = []
                headings = {lvl: text for lvl, text in headings.items() if lvl < level}
                headings[level] = block.lstrip("#").strip()
                continue

            if self._is_markdown_table(block) or self._is_tsv_financial_table(block):
                chunks.extend(
                    self._flush(pending, headings, page_number, column_periods)
                )
                pending = []
                chunks.append(
                    TextChunk(
                        text=self._with_context(block, headings, column_periods),
                        page_number=page_number,
                    )
                )
                continue

            pending.append(block)

        chunks.extend(self._flush(pending, headings, page_number, column_periods))
        return chunks, headings, column_periods

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
        content_hash: str | None = None,
    ) -> list[QdrantPayload]:
        if not chunks:
            return []

        chunk_ids = [str(uuid.uuid4()) for _ in chunks]
        processed_at = datetime.now(UTC)

        payloads: list[QdrantPayload] = []
        for index, chunk in enumerate(chunks):
            metadata = ChunkMetadata(
                tenant_id=ingest_metadata.tenant_id,
                company_name=_required_company_name(ingest_metadata),
                doc_type=_required_doc_type(ingest_metadata),
                category=_required_category(ingest_metadata),
                year=_required_year(ingest_metadata),
                quarter=ingest_metadata.quarter,
                source_filename=source_filename,
                object_key=object_key,
                content_hash=content_hash,
                figure_object_key=chunk.figure_object_key,
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
        column_periods: str | None,
    ) -> list[TextChunk]:
        if not pending:
            return []
        return [
            TextChunk(
                text=self._with_context(chunk.text, headings, column_periods),
                page_number=page_number,
            )
            for chunk in self._split_plain_text("\n\n".join(pending), page_number)
        ]

    @staticmethod
    def _with_context(
        text: str,
        headings: dict[int, str],
        column_periods: str | None = None,
    ) -> str:
        parts: list[str] = []
        if headings:
            parts.append(" > ".join(headings[level] for level in sorted(headings)))
        if column_periods:
            parts.append(f"Kolumner: {column_periods}")
        if not parts:
            return text
        return "\n".join(parts) + "\n\n" + text

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
    def _column_periods_from_block(block: str) -> str | None:
        """Return '2025-06-30 | 2024-06-30' when a block exposes period columns."""
        for line in block.splitlines():
            dates = _ISO_DATE.findall(line)
            if len(dates) >= 2:
                ordered: list[str] = []
                for date in dates:
                    if date not in ordered:
                        ordered.append(date)
                return " | ".join(ordered)
        return None

    @staticmethod
    def _split_markdown_blocks(markdown: str) -> list[str]:
        lines = markdown.splitlines()
        blocks: list[str] = []
        current: list[str] = []
        in_table = False

        for line in lines:
            is_pipe_table_line = "|" in line and bool(line.strip())
            is_tsv_row = Chunker._is_tsv_amount_row(line)
            is_table_line = is_pipe_table_line or is_tsv_row
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
    def _is_tsv_financial_table(block: str) -> bool:
        lines = [line for line in block.splitlines() if line.strip()]
        if len(lines) < 2:
            return False
        amount_rows = sum(1 for line in lines if Chunker._is_tsv_amount_row(line))
        return amount_rows >= 2 and amount_rows / len(lines) >= 0.5

    @staticmethod
    def _is_tsv_amount_row(line: str) -> bool:
        """True for OCR rows like 'Aktiekapital\\t100 000\\t100 000'."""
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "|" in stripped:
            return False
        if "\t" in stripped:
            parts = [part.strip() for part in stripped.split("\t") if part.strip()]
        else:
            parts = re.split(r"\s{2,}|\t+", stripped)
            parts = [part.strip() for part in parts if part.strip()]
        if len(parts) < 3:
            return False
        amount_like = sum(
            1 for part in parts[1:] if _AMOUNT_TOKEN.match(part.replace("\u00a0", " "))
        )
        return amount_like >= 2

    @staticmethod
    def new_document_id() -> str:
        return str(uuid.uuid4())


def _required_company_name(metadata: IngestMetadata) -> str:
    if not metadata.company_name or not metadata.company_name.strip():
        raise ValueError("company_name is required before indexing")
    return metadata.company_name.strip()


def _required_doc_type(metadata: IngestMetadata) -> DocType:
    if metadata.doc_type is None:
        raise ValueError("doc_type is required before indexing")
    return metadata.doc_type


def _required_category(metadata: IngestMetadata) -> Category:
    if metadata.category is None:
        raise ValueError("category is required before indexing")
    return metadata.category


def _required_year(metadata: IngestMetadata) -> int:
    if metadata.year is None:
        raise ValueError("year is required before indexing")
    return metadata.year
