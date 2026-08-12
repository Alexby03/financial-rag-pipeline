from datetime import UTC, datetime

from app.config import Settings
from app.models.api import IngestMetadata
from app.models.enums import Category, DocType
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.models.parsed import ParsedPage
from app.services.chunker import Chunker
from app.services.rag import RagService


def test_build_payloads_links_chunks() -> None:
    settings = Settings(openai_api_key="test-key", chunk_size=40, chunk_overlap=5)
    chunker = Chunker(settings=settings)
    chunks = chunker.split_text(
        "First paragraph with enough text to chunk. "
        "Second paragraph with more details for linking."
    )
    metadata = IngestMetadata(
        tenant_id="tenant_1",
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        quarter=3,
        language="sv",
    )
    payloads = chunker.build_payloads(
        chunks, metadata, "report.pdf", "tenant_1/doc-1/report.pdf"
    )

    assert payloads
    assert payloads[0].metadata.previous_chunk_id is None
    assert payloads[0].metadata.next_chunk_id == payloads[1].metadata.chunk_id
    assert payloads[-1].metadata.next_chunk_id is None
    assert payloads[0].metadata.object_key == "tenant_1/doc-1/report.pdf"


def test_markdown_table_stays_intact() -> None:
    settings = Settings(openai_api_key="test-key", chunk_size=20, chunk_overlap=0)
    chunker = Chunker(settings=settings)
    markdown = (
        "Intro paragraph before the table.\n\n"
        "| Metric | Q3 |\n"
        "| --- | --- |\n"
        "| Revenue | 100 |\n"
        "| EBITDA | 20 |\n\n"
        "Closing paragraph after the table."
    )
    chunks = chunker.split_markdown(markdown)
    table_chunks = [chunk for chunk in chunks if "| Revenue |" in chunk.text]

    assert len(table_chunks) == 1
    assert "| EBITDA | 20 |" in table_chunks[0].text


def test_headings_are_merged_into_content_chunks() -> None:
    settings = Settings(openai_api_key="test-key")
    chunker = Chunker(settings=settings)
    markdown = (
        "```markdown\n"
        "# Kvartalsrapport Q3 2024\n\n"
        "## 2. Finansiell översikt\n\n"
        "| Nyckeltal | Värde (MSEK) |\n"
        "| --- | --- |\n"
        "| Nettoomsättning | 150,0 |\n"
        "```"
    )
    chunks = chunker.split_markdown(markdown)

    assert len(chunks) == 1
    assert "```" not in chunks[0].text
    assert chunks[0].text.startswith("Kvartalsrapport Q3 2024 > 2. Finansiell översikt")
    assert "| Nettoomsättning | 150,0 |" in chunks[0].text


def test_chunks_keep_their_page_and_inherit_headings_across_pages() -> None:
    settings = Settings(openai_api_key="test-key")
    chunker = Chunker(settings=settings)
    pages = [
        ParsedPage(text="# Kvartalsrapport Q3 2024\n\n## 2. Finansiell översikt", page_number=1),
        ParsedPage(
            text=("| Nyckeltal | Värde (MSEK) |\n| --- | --- |\n| Nettoomsättning | 150,0 |"),
            page_number=2,
        ),
    ]
    chunks = chunker.split_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].page_number == 2
    assert chunks[0].text.startswith("Kvartalsrapport Q3 2024 > 2. Finansiell översikt")


def test_qdrant_payload_roundtrip() -> None:
    metadata = ChunkMetadata(
        tenant_id="tenant_1",
        company_name="SME Corp AB",
        doc_type=DocType.AGREEMENT,
        category=Category.LEGAL,
        year=2024,
        quarter=None,
        source_filename="contract.pdf",
        object_key="tenant_1/doc-1/contract.pdf",
        page_number=1,
        chunk_id="uuid-1",
        previous_chunk_id=None,
        next_chunk_id="uuid-2",
        language="sv",
        processed_timestamp=datetime.now(UTC),
    )
    payload = QdrantPayload(raw_text="Sample clause text.", metadata=metadata)
    restored = QdrantPayload.model_validate(payload.to_qdrant_dict())
    assert restored.raw_text == payload.raw_text
    assert restored.metadata.chunk_id == "uuid-1"


def test_rag_orders_linked_neighbors() -> None:
    base = ChunkMetadata(
        tenant_id="tenant_1",
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        quarter=3,
        source_filename="report.pdf",
        object_key="tenant_1/doc-1/report.pdf",
        page_number=1,
        chunk_id="c2",
        previous_chunk_id="c1",
        next_chunk_id="c3",
        language="sv",
        processed_timestamp=datetime.now(UTC),
    )
    payloads = [
        QdrantPayload(raw_text="middle", metadata=base),
        QdrantPayload(
            raw_text="prev",
            metadata=base.model_copy(
                update={"chunk_id": "c1", "previous_chunk_id": None, "next_chunk_id": "c2"}
            ),
        ),
        QdrantPayload(
            raw_text="next",
            metadata=base.model_copy(
                update={"chunk_id": "c3", "previous_chunk_id": "c2", "next_chunk_id": None}
            ),
        ),
    ]

    ordered = RagService._order_by_chunk_chain(payloads)
    assert [payload.metadata.chunk_id for payload in ordered] == ["c1", "c2", "c3"]
