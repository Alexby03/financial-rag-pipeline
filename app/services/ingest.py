import hashlib
import logging
import mimetypes
import uuid
from collections.abc import Callable

from app.models.api import IngestJobStage, IngestMetadata, IngestResponse
from app.models.ocr import OcrDocumentAnnotation
from app.services.chunker import Chunker, TextChunk
from app.services.embedder import Embedder
from app.services.minio_client import MinioStorage
from app.services.parser import DocumentParser
from app.services.qdrant_service import QdrantService

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[IngestJobStage], None]


class IngestMetadataError(ValueError):
    """Raised when OCR and the upload form together cannot fill required metadata."""


class IngestService:
    def __init__(
        self,
        storage: MinioStorage,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        qdrant: QdrantService,
    ) -> None:
        self._storage = storage
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._qdrant = qdrant

    def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: IngestMetadata,
        on_progress: ProgressCallback | None = None,
    ) -> IngestResponse:
        def report(stage: IngestJobStage) -> None:
            if on_progress is not None:
                on_progress(stage)

        report("checking")
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        already_indexed = self._qdrant.find_by_content_hash(metadata.tenant_id, content_hash)
        if already_indexed:
            # Parsing is the expensive step, so identical bytes are never parsed twice, whatever
            # the file happens to be called this time.
            object_key = already_indexed[0].metadata.object_key
            return IngestResponse(
                document_id=self._document_id_from_key(object_key),
                chunk_count=len(already_indexed),
                object_key=object_key,
                status="duplicate",
            )

        document_id = Chunker.new_document_id()
        object_key = f"{metadata.tenant_id}/{document_id}/{filename}"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        uploaded_keys: list[str] = []

        try:
            report("uploading")
            self._storage.upload(object_key, file_bytes, content_type)
            uploaded_keys.append(object_key)

            report("parsing")
            parsed = self._parser.parse(file_bytes, filename)
            metadata = merge_ingest_metadata(metadata, parsed.annotation)

            if parsed.content_format == "json":
                text_chunks = [
                    chunk
                    for page in parsed.pages
                    for chunk in self._chunker.split_json(page.text, page.page_number)
                ]
            else:
                text_chunks = self._chunker.split_pages(
                    parsed.pages,
                    carry_context=parsed.carry_page_context,
                )

            figure_chunks: list[TextChunk] = []
            for figure in parsed.figures:
                figure_id = str(uuid.uuid4())
                extension, figure_type = _image_extension_and_type(figure.image_bytes)
                figure_key = f"{metadata.tenant_id}/{document_id}/figures/{figure_id}.{extension}"
                self._storage.upload(figure_key, figure.image_bytes, figure_type)
                uploaded_keys.append(figure_key)
                figure_chunks.append(
                    TextChunk(
                        text=figure.caption,
                        page_number=figure.page_number,
                        figure_object_key=figure_key,
                    )
                )

            payloads = self._chunker.build_payloads(
                [*text_chunks, *figure_chunks],
                metadata,
                filename,
                object_key,
                content_hash,
            )

            if payloads:
                self._qdrant.delete_by_document(metadata.tenant_id, filename, content_hash)
                report("embedding")
                vectors = self._embedder.embed_texts([payload.raw_text for payload in payloads])
                report("indexing")
                self._qdrant.upsert_chunks(vectors, payloads)
        except Exception:
            self._rollback_uploads(uploaded_keys)
            raise

        return IngestResponse(
            document_id=document_id,
            chunk_count=len(payloads),
            object_key=object_key,
            status="completed",
        )

    def _rollback_uploads(self, object_keys: list[str]) -> None:
        for key in reversed(object_keys):
            try:
                self._storage.delete(key)
            except Exception:
                logger.exception("Failed to roll back MinIO object %s", key)

    @staticmethod
    def _document_id_from_key(object_key: str) -> str:
        parts = object_key.split("/")
        return parts[1] if len(parts) > 2 else ""


def merge_ingest_metadata(
    submitted: IngestMetadata,
    annotation: OcrDocumentAnnotation | None,
) -> IngestMetadata:
    """Explicit form values win; blanks are filled from OCR document_annotation."""
    source = annotation or OcrDocumentAnnotation()
    company_name = _filled_str(submitted.company_name) or _filled_str(source.company_name)
    doc_type = submitted.doc_type or source.doc_type
    category = submitted.category or source.category
    year = submitted.year if submitted.year is not None else source.year
    quarter = submitted.quarter if submitted.quarter is not None else source.quarter

    missing: list[str] = []
    if company_name is None:
        missing.append("company_name")
    if doc_type is None:
        missing.append("doc_type")
    if category is None:
        missing.append("category")
    if year is None:
        missing.append("year")
    if missing:
        raise IngestMetadataError(
            "Could not determine "
            + ", ".join(missing)
            + ". Fill them on upload, or use a document OCR can read."
        )

    assert company_name is not None
    assert doc_type is not None
    assert category is not None
    assert year is not None

    return IngestMetadata(
        tenant_id=submitted.tenant_id,
        company_name=company_name,
        doc_type=doc_type,
        category=category,
        year=year,
        quarter=quarter,
        language=submitted.language,
    )


def _filled_str(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _image_extension_and_type(image_bytes: bytes) -> tuple[str, str]:
    if image_bytes.startswith(b"\x89PNG"):
        return "png", "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "jpg", "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "gif", "image/gif"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return "webp", "image/webp"
    return "png", "image/png"
