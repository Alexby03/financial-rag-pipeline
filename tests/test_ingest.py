import hashlib
from datetime import UTC, datetime

from app.config import Settings
from app.models.api import IngestMetadata
from app.models.enums import Category, DocType
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.models.parsed import ParsedContent, ParsedPage
from app.services.chunker import Chunker
from app.services.ingest import IngestService

FILE_BYTES = b"%PDF-1.7 quarterly report"
CONTENT_HASH = hashlib.sha256(FILE_BYTES).hexdigest()

INGEST_METADATA = IngestMetadata(
    tenant_id="tenant_1",
    company_name="SME Corp AB",
    doc_type=DocType.FINANCIAL_REPORT,
    category=Category.FINANCING,
    year=2024,
    quarter=3,
    language="sv",
)


class FakeStorage:
    def __init__(self) -> None:
        self.uploads: list[str] = []

    def upload(self, object_key: str, data: bytes, content_type: str) -> str:
        self.uploads.append(object_key)
        return object_key


class FakeParser:
    def __init__(self) -> None:
        self.calls = 0

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedContent:
        self.calls += 1
        return ParsedContent(
            pages=[
                ParsedPage(
                    text="# Kassaflöde\n\nKassaflöde från löpande verksamhet 18,3 MSEK.",
                    page_number=1,
                )
            ],
            content_format="markdown",
        )


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


def build_service(qdrant: FakeQdrant, parser: FakeParser) -> IngestService:
    return IngestService(
        storage=FakeStorage(),
        parser=parser,
        chunker=Chunker(settings=Settings(openai_api_key="test-key")),
        embedder=FakeEmbedder(),
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
