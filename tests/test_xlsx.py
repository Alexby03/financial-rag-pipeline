"""Excel → Markdown tables and chunker breadcrumbs for financial models."""

from __future__ import annotations

from io import BytesIO

from openpyxl import Workbook

from app.config import Settings
from app.services.chunker import Chunker
from app.services.parser import DocumentParser
from app.services.xlsx import parse_xlsx_bytes


class _UnusedOcr:
    def process(self, file_bytes: bytes, filename: str):  # noqa: ANN001
        raise AssertionError("xlsx must not call OCR")


def _two_table_workbook_bytes() -> bytes:
    """Mirrors the P&L + FCF layout from the financial-model screenshot."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Model"

    # Table 1 — Total expenses breakdown / P&L
    sheet["A1"] = "Total expenses breakdown / P&L"
    headers = ["Financials (SEKm)", 1991, 1992, "1993E", "2003E", "2004E"]
    for col, value in enumerate(headers, start=1):
        sheet.cell(2, col, value)
    sheet["A3"] = "Total sales"
    sheet["B3"] = 100
    sheet["C3"] = 110
    sheet["D3"] = 120
    sheet["E3"] = 200
    sheet["F3"] = 210
    sheet["A4"] = "EBITDA"
    sheet["B4"] = 40
    sheet["C4"] = 44
    sheet["D4"] = 48
    sheet["E4"] = 80
    sheet["F4"] = 84
    sheet["A5"] = "Net income"
    sheet["B5"] = 20
    sheet["C5"] = 22
    sheet["D5"] = 24
    sheet["E5"] = 55.67
    sheet["F5"] = 58
    sheet["G4"] = "Opex TBU"
    sheet["G5"] = "orimligt att ta interest expense som % av sales"

    # Blank row gap → second visual table
    sheet["A7"] = "FCF from Net income"
    fcf_headers = ["SEKm", 1991, 1992, "1993E", "2003E", "2004E"]
    for col, value in enumerate(fcf_headers, start=1):
        sheet.cell(8, col, value)
    sheet["A9"] = "Net income"
    sheet["B9"] = 20
    sheet["C9"] = 22
    sheet["D9"] = 24
    sheet["E9"] = 55.67
    sheet["F9"] = 58
    sheet["A10"] = "Unlevered FCF"
    sheet["B10"] = 15
    sheet["C10"] = 16
    sheet["D10"] = 17
    sheet["E10"] = 40
    sheet["F10"] = 42

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_parse_xlsx_splits_blank_row_tables() -> None:
    parsed = parse_xlsx_bytes(_two_table_workbook_bytes())

    assert len(parsed.pages) == 1
    text = parsed.pages[0].text
    assert text.startswith("# Model")
    assert "## Total expenses breakdown / P&L" in text
    assert "## FCF from Net income" in text
    assert "| Post | 1991 | 1992 | 1993E | 2003E | 2004E |" in text or (
        "| Financials (SEKm) | 1991 | 1992 | 1993E | 2003E | 2004E |" in text
    )
    assert "55.67" in text
    assert "Unlevered FCF" in text
    assert "Opex TBU" in text
    assert parsed.annotation is not None
    assert parsed.annotation.year == 2004


def test_document_parser_bypasses_ocr_for_xlsx() -> None:
    parser = DocumentParser(ocr_client=_UnusedOcr())  # type: ignore[arg-type]
    parsed = parser.parse(_two_table_workbook_bytes(), "model.xlsx")
    assert len(parsed.pages) == 1
    assert "EBITDA" in parsed.pages[0].text


def test_chunker_keeps_each_xlsx_table_atomic_with_year_breadcrumbs() -> None:
    parsed = parse_xlsx_bytes(_two_table_workbook_bytes())
    chunker = Chunker(settings=Settings(openai_api_key="test-key", chunk_size=80, chunk_overlap=0))
    chunks = chunker.split_pages(parsed.pages)

    money = [chunk for chunk in chunks if "55.67" in chunk.text]
    fcf = [chunk for chunk in chunks if "Unlevered FCF" in chunk.text]

    assert money
    assert fcf
    assert money[0] is not fcf[0] or "Unlevered FCF" in money[0].text
    # Year labels stick like ISO dates did for Bolagsverket tables.
    assert "Kolumner (vänster→höger efter radetikett): 1991 | 1992 | 1993E | 2003E | 2004E" in (
        money[0].text
    )
    assert "Model > Total expenses breakdown / P&L" in money[0].text
    assert "Model > FCF from Net income" in fcf[0].text
    # Tables must not be character-split despite tiny chunk_size.
    assert "Net income" in money[0].text
    assert "1991" in money[0].text and "2004E" in money[0].text


def _side_by_side_workbook_bytes() -> bytes:
    """Two year-grids on the same rows with a blank column between them."""
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet.title = "Parallel"

    # Left table
    sheet["A1"] = "Income statement"
    sheet["A2"] = "SEKm"
    sheet["B2"] = 2022
    sheet["C2"] = 2023
    sheet["A3"] = "Revenue"
    sheet["B3"] = 10
    sheet["C3"] = 12
    sheet["A4"] = "EBIT"
    sheet["B4"] = 2
    sheet["C4"] = 3

    # Blank column D, then right table
    sheet["F1"] = "Balance sheet"
    sheet["F2"] = "SEKm"
    sheet["G2"] = 2022
    sheet["H2"] = 2023
    sheet["F3"] = "Cash"
    sheet["G3"] = 5
    sheet["H3"] = 6
    sheet["F4"] = "Equity"
    sheet["G4"] = 8
    sheet["H4"] = 9

    buffer = BytesIO()
    workbook.save(buffer)
    return buffer.getvalue()


def test_parse_xlsx_splits_side_by_side_tables() -> None:
    parsed = parse_xlsx_bytes(_side_by_side_workbook_bytes())
    text = parsed.pages[0].text

    assert "## Income statement" in text
    assert "## Balance sheet" in text
    assert "Revenue" in text and "Cash" in text
    # Must not smash both grids into one header row.
    assert text.count("| 2022 | 2023 |") >= 2
    income_pos = text.index("Income statement")
    balance_pos = text.index("Balance sheet")
    revenue_pos = text.index("Revenue")
    cash_pos = text.index("Cash")
    assert income_pos < revenue_pos < balance_pos or income_pos < balance_pos
    assert "Cash" in text[balance_pos:]
    assert "Revenue" in text[income_pos:balance_pos]


def test_side_by_side_notes_without_year_grid_stay_with_left_table() -> None:
    workbook = Workbook()
    sheet = workbook.active
    assert sheet is not None
    sheet["A1"] = "P&L"
    sheet["A2"] = "Post"
    sheet["B2"] = 2023
    sheet["C2"] = 2024
    sheet["A3"] = "Sales"
    sheet["B3"] = 1
    sheet["C3"] = 2
    # Blank column D, then a note-only strip (no year headers)
    sheet["F3"] = "check vs budget"
    buffer = BytesIO()
    workbook.save(buffer)

    text = parse_xlsx_bytes(buffer.getvalue()).pages[0].text
    assert text.count("## ") == 1
    assert "check vs budget" in text
    assert "Sales" in text


def test_chunker_does_not_leak_sheet_breadcrumbs_across_xlsx_pages() -> None:
    """Regression: Financials H1 must not stick onto Exhibit 7 antibody rows."""
    workbook = Workbook()
    financials = workbook.active
    assert financials is not None
    financials.title = "Financials"
    financials["A1"] = "P&L"
    financials["A2"] = "Post"
    financials["B2"] = 1991
    financials["C2"] = 1992
    financials["A3"] = "EBIT"
    financials["B3"] = 1
    financials["C3"] = 2

    exhibit = workbook.create_sheet("Exhibit 7 (Rhomed forecast)")
    exhibit["A1"] = "Antibody Derivative Products"
    exhibit["A2"] = "Post"
    exhibit["B2"] = 1991
    exhibit["C2"] = 1992
    exhibit["A3"] = "LeukoScan"
    exhibit["B3"] = "n.a."
    exhibit["C3"] = "n.a."

    buffer = BytesIO()
    workbook.save(buffer)
    parsed = parse_xlsx_bytes(buffer.getvalue())
    assert parsed.carry_page_context is False

    chunks = Chunker(Settings(openai_api_key="test-key")).split_pages(
        parsed.pages,
        carry_context=parsed.carry_page_context,
    )
    antibody = next(chunk for chunk in chunks if "LeukoScan" in chunk.text)
    assert antibody.page_number == 2
    assert "Exhibit 7 (Rhomed forecast) > Antibody Derivative Products" in antibody.text
    assert "Financials >" not in antibody.text
    assert "Kolumner (vänster→höger efter radetikett): 1991 | 1992" in antibody.text
