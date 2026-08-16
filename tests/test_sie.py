from pathlib import Path

import pytest

from app.models.enums import Category, DocType, Language
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
    assert parsed.annotation.category == Category.FINANCING
    assert parsed.annotation.year == 2024
    assert parsed.annotation.language == Language.SV


def test_parse_sie_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        parse_sie_bytes(b"")


def test_sie_annotation_defaults_category_and_finds_year_in_period() -> None:
    from app.services.sie import _annotation_from_sie

    annotation = _annotation_from_sie(
        {
            "metadata": {"company_name": "Sparse AB"},
            "summary": {"company_name": "Sparse AB", "period": "20230101 to 20231231"},
        }
    )
    assert annotation.doc_type == DocType.ACCOUNTING_DATA
    assert annotation.category == Category.FINANCING
    assert annotation.year == 2023
    assert annotation.company_name == "Sparse AB"
    assert annotation.language == Language.SV


def test_norstedts_tab_separated_sie_fills_year_not_program_version() -> None:
    """Norstedts exports use tabs; #PROGRAM 2011.2.1 is software, not fiscal year."""
    fixture = Path(__file__).parent / "fixtures" / "norstedts_tabs.se"
    # Keep literal tabs from the fixture (do not normalize to spaces).
    parsed = parse_sie_bytes(fixture.read_bytes())

    assert parsed.annotation is not None
    assert parsed.annotation.company_name == "Testbolaget AB"
    assert parsed.annotation.year == 2011
    assert parsed.annotation.category == Category.FINANCING


def test_program_version_is_never_used_as_year() -> None:
    from app.services.sie import _annotation_from_sie, _year_from_value

    assert _year_from_value("2011.2.1") is None
    annotation = _annotation_from_sie(
        {
            "metadata": {
                "company_name": "X AB",
                "program": "Norstedts Bokslut",
                "program_version": "2011.2.1",
                "gen_date": "20120315",
            },
            "summary": {},
        }
    )
    assert annotation.year == 2012


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
