import hashlib
from datetime import UTC, datetime

import pytest

from app.config import Settings
from app.models.api import IngestMetadata
from app.models.enums import Category, DocType, Language
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.models.ocr import OcrDocumentAnnotation
from app.models.parsed import ParsedContent, ParsedFigure, ParsedPage
from app.services.chunker import Chunker
from app.services.ingest import IngestMetadataError, IngestService, merge_ingest_metadata

FILE_BYTES = b"%PDF-1.7 quarterly report"
CONTENT_HASH = hashlib.sha256(FILE_BYTES).hexdigest()
TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

INGEST_METADATA = IngestMetadata(
    tenant_id="tenant_1",
    company_name="SME Corp AB",
    doc_type=DocType.FINANCIAL_REPORT,
    category=Category.FINANCING,
    year=2024,
    quarter=3,
    language=Language.SV,
)


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[tuple[str, bytes, str]] = []
        self.deletes: list[str] = []

    def upload(self, object_key: str, data: bytes, content_type: str) -> str:
        self.uploads.append((object_key, data, content_type))
        return object_key

    def delete(self, object_key: str) -> None:
        self.deletes.append(object_key)
        self.uploads = [item for item in self.uploads if item[0] != object_key]


class FakeParser:
    def __init__(
        self,
        figures: list[ParsedFigure] | None = None,
        annotation: OcrDocumentAnnotation | None = None,
    ) -> None:
        self.calls = 0
        self._figures = figures or []
        self._annotation = annotation

    def parse(self, file_bytes: bytes, filename: str) -> ParsedContent:
        self.calls += 1
        return ParsedContent(
            pages=[
                ParsedPage(
                    text="# Kassaflöde\n\nKassaflöde från löpande verksamhet 18,3 MSEK.",
                    page_number=1,
                )
            ],
            content_format="markdown",
            figures=list(self._figures),
            annotation=self._annotation,
        )


class ExplodingParser:
    def parse(self, file_bytes: bytes, filename: str) -> ParsedContent:
        raise RuntimeError("OCR endpoint returned 400: too many pages")


class ExplodingEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        raise RuntimeError("embedding failed")


class FakeEmbedder:
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return [[0.1, 0.2, 0.3] for _ in texts]


class FakeQdrant:
    def __init__(self, existing: list[QdrantPayload] | None = None) -> None:
        self.existing = existing or []
        self.deletes: list[tuple[str, str, str | None]] = []
        self.upserted: list[QdrantPayload] = []

    def find_by_content_hash(self, tenant_id: str, content_hash: str) -> list[QdrantPayload]:
        return self.existing

    def delete_by_document(
        self, tenant_id: str, source_filename: str, content_hash: str | None = None
    ) -> None:
        self.deletes.append((tenant_id, source_filename, content_hash))

    def upsert_chunks(self, vectors: list[list[float]], payloads: list[QdrantPayload]) -> int:
        self.upserted.extend(payloads)
        return len(payloads)


def build_service(
    qdrant: FakeQdrant,
    parser: FakeParser | ExplodingParser,
    storage: FakeStorage | None = None,
    embedder: FakeEmbedder | ExplodingEmbedder | None = None,
) -> IngestService:
    return IngestService(
        storage=storage or FakeStorage(),
        parser=parser,
        chunker=Chunker(settings=Settings(openai_api_key="test-key")),
        embedder=embedder or FakeEmbedder(),
        qdrant=qdrant,
    )


def existing_chunk(object_key: str) -> QdrantPayload:
    return QdrantPayload(
        raw_text="Kassaflöde från löpande verksamhet 18,3 MSEK.",
        metadata=ChunkMetadata(
            tenant_id="tenant_1",
            company_name="SME Corp AB",
            doc_type=DocType.FINANCIAL_REPORT,
            category=Category.FINANCING,
            year=2024,
            quarter=3,
            source_filename="@q3_report_2024.pdf",
            object_key=object_key,
            content_hash=CONTENT_HASH,
            page_number=3,
            chunk_id="c1",
            previous_chunk_id=None,
            next_chunk_id=None,
            language="sv",
            processed_timestamp=datetime.now(UTC),
        ),
    )


def test_new_document_is_parsed_and_stamped_with_its_hash() -> None:
    qdrant = FakeQdrant()
    parser = FakeParser()
    service = build_service(qdrant, parser)
    stages: list[str] = []

    response = service.ingest(
        FILE_BYTES, "q3_report_2024.pdf", INGEST_METADATA, on_progress=stages.append
    )

    assert response.status == "completed"
    assert parser.calls == 1
    assert qdrant.upserted
    assert all(payload.metadata.content_hash == CONTENT_HASH for payload in qdrant.upserted)
    assert qdrant.deletes == [("tenant_1", "q3_report_2024.pdf", CONTENT_HASH)]
    assert stages == ["checking", "uploading", "parsing", "embedding", "indexing"]


def test_identical_bytes_under_a_new_name_are_not_parsed_again() -> None:
    qdrant = FakeQdrant(existing=[existing_chunk("tenant_1/doc-9/@q3_report_2024.pdf")])
    parser = FakeParser()
    service = build_service(qdrant, parser)
    stages: list[str] = []

    response = service.ingest(
        FILE_BYTES, "q3_report_2024.pdf", INGEST_METADATA, on_progress=stages.append
    )

    assert response.status == "duplicate"
    assert response.document_id == "doc-9"
    assert response.object_key == "tenant_1/doc-9/@q3_report_2024.pdf"
    assert response.chunk_count == 1
    assert parser.calls == 0
    assert not qdrant.upserted
    assert stages == ["checking"]


def test_figures_are_uploaded_as_separate_chunks_with_figure_keys() -> None:
    qdrant = FakeQdrant()
    storage = FakeStorage()
    parser = FakeParser(
        figures=[
            ParsedFigure(
                page_number=4,
                caption="Revenue trend Q3",
                image_bytes=TINY_PNG,
            )
        ]
    )
    service = build_service(qdrant, parser, storage=storage)

    service.ingest(FILE_BYTES, "q3_report_2024.pdf", INGEST_METADATA)

    figure_uploads = [
        (key, data, content_type)
        for key, data, content_type in storage.uploads
        if "/figures/" in key
    ]
    assert len(figure_uploads) == 1
    figure_key, data, content_type = figure_uploads[0]
    assert figure_key.endswith(".png")
    assert data == TINY_PNG
    assert content_type == "image/png"

    text_payloads = [p for p in qdrant.upserted if p.metadata.figure_object_key is None]
    figure_payloads = [p for p in qdrant.upserted if p.metadata.figure_object_key is not None]
    assert text_payloads
    assert all(p.metadata.figure_object_key is None for p in text_payloads)
    assert len(figure_payloads) == 1
    assert figure_payloads[0].raw_text == "Revenue trend Q3"
    assert figure_payloads[0].metadata.figure_object_key == figure_key
    assert figure_payloads[0].metadata.object_key.endswith("q3_report_2024.pdf")
    assert figure_payloads[0].metadata.page_number == 4


def test_ocr_annotation_fills_blank_form_fields() -> None:
    qdrant = FakeQdrant()
    parser = FakeParser(
        annotation=OcrDocumentAnnotation(
            company_name="SME Corp AB",
            doc_type=DocType.FINANCIAL_REPORT,
            category=Category.FINANCING,
            year=2024,
            quarter=3,
            language=Language.SV,
        )
    )
    service = build_service(qdrant, parser)
    submitted = IngestMetadata(tenant_id="tenant_1")

    service.ingest(FILE_BYTES, "q3_report_2024.pdf", submitted)

    assert qdrant.upserted
    meta = qdrant.upserted[0].metadata
    assert meta.company_name == "SME Corp AB"
    assert meta.doc_type == DocType.FINANCIAL_REPORT
    assert meta.year == 2024
    assert meta.quarter == 3
    assert meta.language == Language.SV


def test_submitted_form_fields_win_over_annotation() -> None:
    submitted = IngestMetadata(
        tenant_id="tenant_1",
        company_name="Override AB",
        doc_type=DocType.AGREEMENT,
        category=Category.LEGAL,
        year=2023,
        quarter=1,
        language=Language.EN,
    )
    annotation = OcrDocumentAnnotation(
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        quarter=3,
        language=Language.SV,
    )

    merged = merge_ingest_metadata(submitted, annotation)

    assert merged.company_name == "Override AB"
    assert merged.doc_type == DocType.AGREEMENT
    assert merged.year == 2023
    assert merged.quarter == 1
    assert merged.language == Language.EN


def test_missing_metadata_after_merge_keeps_minio_object() -> None:
    storage = FakeStorage()
    service = build_service(FakeQdrant(), FakeParser(), storage=storage)
    with pytest.raises(IngestMetadataError, match="company_name") as exc_info:
        service.ingest(FILE_BYTES, "notes.txt", IngestMetadata(tenant_id="tenant_1"))
    assert storage.deletes == []
    assert len(storage.uploads) == 1
    assert storage.uploads[0][0].endswith("notes.txt")
    assert "company_name" in exc_info.value.missing
    assert exc_info.value.object_key.endswith("notes.txt")
    assert exc_info.value.document_id


def test_ocr_failure_rolls_back_the_minio_object() -> None:
    storage = FakeStorage()
    service = build_service(FakeQdrant(), ExplodingParser(), storage=storage)
    with pytest.raises(RuntimeError, match="too many pages"):
        service.ingest(FILE_BYTES, "q3_report_2024.pdf", INGEST_METADATA)
    assert len(storage.deletes) == 1
    assert storage.deletes[0].endswith("q3_report_2024.pdf")
    assert storage.uploads == []


def test_embed_failure_rolls_back_source_and_figure_objects() -> None:
    storage = FakeStorage()
    parser = FakeParser(
        figures=[
            ParsedFigure(page_number=4, caption="Revenue trend Q3", image_bytes=TINY_PNG)
        ]
    )
    service = build_service(
        FakeQdrant(), parser, storage=storage, embedder=ExplodingEmbedder()
    )
    with pytest.raises(RuntimeError, match="embedding failed"):
        service.ingest(FILE_BYTES, "q3_report_2024.pdf", INGEST_METADATA)
    assert len(storage.deletes) == 2
    assert any(key.endswith("q3_report_2024.pdf") for key in storage.deletes)
    assert any("/figures/" in key for key in storage.deletes)
    assert storage.uploads == []
