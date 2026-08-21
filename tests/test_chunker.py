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


def test_section_three_does_not_nest_under_subsection_on_the_next_page() -> None:
    settings = Settings(openai_api_key="test-key")
    chunker = Chunker(settings=settings)
    pages = [
        ParsedPage(
            text=(
                "# Kvartalsrapport Q3 2024\n\n"
                "## 2. Finansiell översikt\n\n"
                "### 2.3 Utvalda nyckeltal"
            ),
            page_number=3,
        ),
        ParsedPage(
            text="## 3. Analys av finansiering\n\nKapitalstrukturen optimerades under kvartalet.",
            page_number=4,
        ),
    ]
    chunks = chunker.split_pages(pages)

    assert len(chunks) == 1
    assert chunks[0].page_number == 4
    assert chunks[0].text.startswith("Kvartalsrapport Q3 2024 > 3. Analys av finansiering")
    assert "2.3" not in chunks[0].text.split("\n\n", 1)[0]


def test_balance_sheet_column_years_stick_to_tsv_rows() -> None:
    """OCR often emits tab rows; period headers must ride along like breadcrumbs."""
    settings = Settings(openai_api_key="test-key", chunk_size=80, chunk_overlap=0)
    chunker = Chunker(settings=settings)
    markdown = (
        "# Balansräkning\n\n"
        "Not\t2025-06-30\t2024-06-30\n\n"
        "## EGET KAPITAL OCH SKULDER\n\n"
        "Fritt eget kapital\n\n"
        "Balanserat resultat\t67 646\t40 803\n"
        "Årets resultat\t52 601\t26 843\n"
        "Summa fritt eget kapital\t120 247\t67 646\n"
    )
    chunks = chunker.split_markdown(markdown)
    money_chunks = [chunk for chunk in chunks if "67 646" in chunk.text]

    assert money_chunks
    assert "| Post | 2025-06-30 | 2024-06-30 |" in money_chunks[0].text
    assert "EGET KAPITAL OCH SKULDER" in money_chunks[0].text
    assert "Årets resultat" in money_chunks[0].text


def test_column_periods_carry_across_page_break() -> None:
    settings = Settings(openai_api_key="test-key")
    chunker = Chunker(settings=settings)
    pages = [
        ParsedPage(
            text="# Balansräkning\n\n| Not | 2025-06-30 | 2024-06-30 |\n| --- | ---: | ---: |",
            page_number=3,
        ),
        ParsedPage(
            text=(
                "## EGET KAPITAL OCH SKULDER\n\n"
                "Aktiekapital\t100 000\t100 000\n"
                "Summa bundet eget kapital\t100 000\t100 000\n"
            ),
            page_number=4,
        ),
    ]
    chunks = chunker.split_pages(pages)
    equity = [chunk for chunk in chunks if "Aktiekapital" in chunk.text]

    assert len(equity) == 1
    assert equity[0].page_number == 4
    assert "| Post | 2025-06-30 | 2024-06-30 |" in equity[0].text


def test_column_periods_do_not_stamp_later_prose_slides() -> None:
    """Investor-deck cards must not inherit Kolumner from an earlier financial table."""
    settings = Settings(openai_api_key="test-key")
    chunker = Chunker(settings=settings)
    pages = [
        ParsedPage(
            text="# Balansräkning\n\n| Not | 2025-06-30 | 2024-06-30 |\n| --- | ---: | ---: |",
            page_number=3,
        ),
        ParsedPage(
            text=(
                "# Strategy to Lead Global AI Technology Company "
                "Serving Both AI Builders + AI Adopters.\n\n"
                "## Innovate\n\n"
                "New Capabilities + Platforms. Expand addressable market."
            ),
            page_number=12,
        ),
        ParsedPage(
            text=(
                "## EGET KAPITAL OCH SKULDER\n\n"
                "Aktiekapital\t100 000\t100 000\n"
                "Summa bundet eget kapital\t100 000\t100 000\n"
            ),
            page_number=13,
        ),
    ]
    chunks = chunker.split_pages(pages)
    strategy = [chunk for chunk in chunks if "New Capabilities" in chunk.text]
    equity = [chunk for chunk in chunks if "Aktiekapital" in chunk.text]

    assert len(strategy) == 1
    assert "Kolumner" not in strategy[0].text
    assert "1934" not in strategy[0].text
    assert strategy[0].text.startswith(
        "Strategy to Lead Global AI Technology Company"
    )
    assert equity
    assert "| Post | 2025-06-30 | 2024-06-30 |" in equity[0].text


def test_sec_act_years_are_not_column_periods() -> None:
    """Page-2 FLS cites the Exchange Act of 1934 and Securities Act of 1933."""
    settings = Settings(openai_api_key="test-key")
    chunker = Chunker(settings=settings)
    pages = [
        ParsedPage(
            text=(
                "# Forward-Looking Statements\n\n"
                "This presentation contains forward-looking statements within the meaning of "
                "Section 21E of the Securities Exchange Act of 1934, as amended, and "
                "Section 27A of the Securities Act of 1933, as amended. These forward-looking "
                "statements are based on current expectations."
            ),
            page_number=2,
        ),
        ParsedPage(
            text=(
                "# Strategy to Lead Global AI Technology Company "
                "Serving Both AI Builders + AI Adopters.\n\n"
                "## Innovate\n\n"
                "New Capabilities + Platforms. Expand addressable market."
            ),
            page_number=12,
        ),
    ]
    chunks = chunker.split_pages(pages)
    strategy = [chunk for chunk in chunks if "New Capabilities" in chunk.text]
    disclaimer = [chunk for chunk in chunks if "forward-looking" in chunk.text.lower()]

    assert disclaimer
    assert "1934" in disclaimer[0].text
    assert "1933" in disclaimer[0].text
    assert "Kolumner" not in disclaimer[0].text
    assert len(strategy) == 1
    assert "Kolumner" not in strategy[0].text
    assert "1934" not in strategy[0].text
    assert "1933" not in strategy[0].text


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
