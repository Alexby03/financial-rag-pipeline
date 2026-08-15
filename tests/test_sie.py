from pathlib import Path

import pytest

from app.models.enums import DocType
from app.services.mistral_ocr import OcrResult
from app.services.parser import DocumentParser
from app.services.sie import is_sie_filename, parse_sie_bytes

FIXTURE = Path(__file__).parent / "fixtures" / "sample.se"


class FakeOcrClient:
    def __init__(self) -> None:
        self.calls: list[tuple[bytes, str]] = []

    def process(self, file_bytes: bytes, filename: str) -> OcrResult:
        self.calls.append((file_bytes, filename))
        return OcrResult(pages=[])


def test_is_sie_filename() -> None:
    assert is_sie_filename("books.se")
    assert is_sie_filename("import.SI")
    assert is_sie_filename("ledger.SIE")
    assert not is_sie_filename("report.pdf")
    assert not is_sie_filename("notes.txt")


def test_parse_sie_bytes_builds_markdown_summaries() -> None:
    parsed = parse_sie_bytes(FIXTURE.read_bytes())

    assert parsed.content_format == "markdown"
    assert parsed.pages
    joined = "\n".join(page.text for page in parsed.pages)
    assert "SME Corp AB" in joined
    assert "Balance sheet" in joined or "Income statement" in joined
    assert parsed.annotation is not None
    assert parsed.annotation.company_name == "SME Corp AB"
    assert parsed.annotation.doc_type == DocType.ACCOUNTING_DATA
    assert parsed.annotation.year == 2024


def test_parse_sie_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_sie_bytes(b"")


def test_document_parser_routes_sie_without_ocr() -> None:
    client = FakeOcrClient()
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(FIXTURE.read_bytes(), "books.se")

    assert client.calls == []
    assert parsed.annotation is not None
    assert parsed.annotation.company_name == "SME Corp AB"
    assert any("SME Corp AB" in page.text for page in parsed.pages)


def test_document_parser_routes_si_extension() -> None:
    client = FakeOcrClient()
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(FIXTURE.read_bytes(), "import.si")

    assert client.calls == []
    assert parsed.pages
