"""Excel workbooks → Markdown tables for chunking (no OCR)."""

from __future__ import annotations

from pathlib import Path

from app.models.ocr import OcrDocumentAnnotation
from app.models.parsed import ParsedContent, ParsedPage
from app.services.spreadsheet.loader import load_sheet_grids
from app.services.spreadsheet.render import render_sheet
from app.services.spreadsheet.structure import detect_subtables

XLSX_EXTENSIONS = {".xlsx", ".xlsm"}


def is_xlsx_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in XLSX_EXTENSIONS


def parse_xlsx_bytes(file_bytes: bytes) -> ParsedContent:
    """Parse an .xlsx/.xlsm workbook into Markdown pages (one page per sheet)."""
    if not file_bytes:
        raise ValueError("Excel file is empty")

    sheets = load_sheet_grids(file_bytes)
    pages: list[ParsedPage] = []
    years: list[int] = []

    for sheet_index, (sheet_title, grid) in enumerate(sheets.items(), start=1):
        regions = detect_subtables(grid)
        markdown, sheet_years = render_sheet(sheet_title, regions)
        if not markdown.strip():
            continue
        pages.append(ParsedPage(text=markdown, page_number=sheet_index))
        years.extend(sheet_years)

    if not pages:
        raise ValueError("Excel file produced no table content")

    annotation = OcrDocumentAnnotation(year=max(years)) if years else None
    return ParsedContent(
        pages=pages,
        content_format="markdown",
        annotation=annotation,
        carry_page_context=False,
    )
