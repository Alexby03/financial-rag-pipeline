from datetime import UTC, datetime

from app.config import Settings
from app.models.enums import Category, DocType, Language
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.services.qdrant_service import QDRANT_UPSERT_MAX_POINTS, QdrantService, _batched


def test_batched_splits_evenly() -> None:
    assert _batched([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]]
    assert _batched([], 10) == []


def test_upsert_chunks_sends_multiple_qdrant_batches() -> None:
    settings = Settings(openai_api_key="test-key")
    service = QdrantService(settings)
    calls: list[list[object]] = []

    def fake_upsert(*, collection_name: str, points: list[object]) -> None:
        calls.append(list(points))

    service._client.upsert = fake_upsert  # type: ignore[method-assign]

    count = QDRANT_UPSERT_MAX_POINTS + 3
    vectors = [[0.1] * 3 for _ in range(count)]
    payloads = [_sample_payload(i) for i in range(count)]
    written = service.upsert_chunks(vectors, payloads)

    assert written == count
    assert len(calls) == 2
    assert len(calls[0]) == QDRANT_UPSERT_MAX_POINTS
    assert len(calls[1]) == 3


def _sample_payload(index: int) -> QdrantPayload:
    return QdrantPayload(
        raw_text=f"chunk {index}",
        metadata=ChunkMetadata(
            tenant_id="tenant_1",
            company_name="SME Corp AB",
            doc_type=DocType.OTHER,
            category=Category.OTHER,
            year=2024,
            quarter=None,
            source_filename="doc.pdf",
            object_key="tenant_1/doc/doc.pdf",
            content_hash="abc",
            page_number=1,
            chunk_id=f"00000000-0000-4000-8000-{index:012d}",
            previous_chunk_id=None,
            next_chunk_id=None,
            language=Language.SV,
            processed_timestamp=datetime.now(UTC),
        ),
    )
