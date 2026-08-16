"""Excel workbooks → Markdown tables for chunking (no OCR).

Lessons from the XHTML / balance-sheet path:
- Keep each visual table as one unsplittable markdown table (chunker already
  treats pipe tables as atomic).
- Put period labels in the header row (``1991 | 1993E | …`` or ISO dates), not
  only as free text above the grid.
- Emit sheet + table titles as ``#`` / ``##`` breadcrumbs so chunks carry
  ``Sheet > Table`` context the way OCR pages carry section headings.
- Split on blank-row gaps (stacked tables) and blank-column gaps (side-by-side
  parallel tables), not on arbitrary row/column counts.
"""

from __future__ import annotations

import re
from datetime import date, datetime
from io import BytesIO
from pathlib import Path

from openpyxl import load_workbook
from openpyxl.cell.cell import Cell
from openpyxl.workbook.workbook import Workbook
from openpyxl.worksheet.worksheet import Worksheet

from app.models.ocr import OcrDocumentAnnotation
from app.models.parsed import ParsedContent, ParsedPage

XLSX_EXTENSIONS = {".xlsx", ".xlsm"}

# Calendar / fiscal year headers common in Financial models (1991, 2004E, FY2024).
_YEAR_LABEL = re.compile(
    r"(?i)\b(?:FY\s*)?((?:19|20)\d{2})([Ee])?\b"
)
_ISO_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")


def is_xlsx_filename(filename: str) -> bool:
    return Path(filename).suffix.lower() in XLSX_EXTENSIONS


def parse_xlsx_bytes(file_bytes: bytes) -> ParsedContent:
    """Parse an .xlsx/.xlsm workbook into Markdown pages (one page per sheet)."""
    if not file_bytes:
        raise ValueError("Excel file is empty")

    workbook = _load_workbook(file_bytes)
    pages: list[ParsedPage] = []
    years: list[int] = []
    try:
        for sheet_index, sheet in enumerate(workbook.worksheets, start=1):
            markdown, sheet_years = _sheet_to_markdown(sheet)
            if not markdown.strip():
                continue
            pages.append(ParsedPage(text=markdown, page_number=sheet_index))
            years.extend(sheet_years)
    finally:
        workbook.close()

    if not pages:
        raise ValueError("Excel file produced no table content")

    annotation = None
    if years:
        annotation = OcrDocumentAnnotation(year=max(years))
    return ParsedContent(
        pages=pages,
        content_format="markdown",
        annotation=annotation,
        carry_page_context=False,
    )


def _load_workbook(file_bytes: bytes) -> Workbook:
    workbook = load_workbook(BytesIO(file_bytes), data_only=True, read_only=False)
    if _workbook_has_values(workbook):
        return workbook
    workbook.close()
    return load_workbook(BytesIO(file_bytes), data_only=False, read_only=False)


def _workbook_has_values(workbook: Workbook) -> bool:
    for sheet in workbook.worksheets:
        max_row = min(sheet.max_row or 1, 40)
        max_col = min(sheet.max_column or 1, 40)
        for row in sheet.iter_rows(
            min_row=1,
            max_row=max_row,
            min_col=1,
            max_col=max_col,
            values_only=True,
        ):
            if any(value is not None and str(value).strip() for value in row):
                return True
    return False


def _sheet_to_markdown(sheet: Worksheet) -> tuple[str, list[int]]:
    grid = _sheet_grid(sheet)
    if not grid:
        return "", []

    blocks = _row_blocks(grid)
    if not blocks:
        return "", []

    sheet_title = (sheet.title or "Sheet").strip() or "Sheet"
    sections: list[str] = [f"# {sheet_title}", ""]
    years: list[int] = []
    table_index = 0

    for start, end in blocks:
        block_rows = grid[start : end + 1]
        for panel in _column_panels(block_rows):
            table_index += 1
            rendered, block_years = _block_to_markdown(panel, block_index=table_index)
            if not rendered:
                continue
            sections.append(rendered)
            sections.append("")
            years.extend(block_years)

    return "\n".join(sections).strip() + "\n", years


def _sheet_grid(sheet: Worksheet) -> list[list[str]]:
    if not sheet.max_row or not sheet.max_column:
        return []

    grid: list[list[str]] = []
    for row in sheet.iter_rows(
        min_row=1,
        max_row=sheet.max_row,
        min_col=1,
        max_col=sheet.max_column,
    ):
        grid.append([_cell_to_text(cell) for cell in row])

    while grid and _row_blank(grid[-1]):
        grid.pop()
    if not grid:
        return []

    width = len(grid[0])
    while width > 0 and all(not row[width - 1].strip() for row in grid):
        for row in grid:
            row.pop()
        width -= 1
    return grid


def _row_blank(row: list[str]) -> bool:
    return all(not cell.strip() for cell in row)


def _row_blocks(grid: list[list[str]]) -> list[tuple[int, int]]:
    """Contiguous non-empty row runs separated by ≥1 blank rows."""
    blocks: list[tuple[int, int]] = []
    start: int | None = None
    for index, row in enumerate(grid):
        if _row_blank(row):
            if start is not None:
                blocks.append((start, index - 1))
                start = None
            continue
        if start is None:
            start = index
    if start is not None:
        blocks.append((start, len(grid) - 1))
    return blocks


def _column_panels(rows: list[list[str]]) -> list[list[list[str]]]:
    """Split a row-block on fully blank columns (side-by-side tables).

    Weak right-hand strips (notes without their own year grid) are glued back onto
    the previous panel so a spacer before margin comments does not invent a table.
    """
    trimmed = _trim_columns(rows)
    if not trimmed:
        return []

    raw = _split_on_blank_columns(trimmed)
    if len(raw) <= 1:
        return raw

    panels: list[list[list[str]]] = []
    for panel in raw:
        if _looks_like_table_panel(panel):
            panels.append(panel)
        elif panels:
            panels[-1] = _hstack_panels(panels[-1], panel)
        else:
            panels.append(panel)
    return panels


def _split_on_blank_columns(rows: list[list[str]]) -> list[list[list[str]]]:
    width = len(rows[0])
    blank = {
        index
        for index in range(width)
        if all(not row[index].strip() for row in rows)
    }
    if not blank:
        return [rows]

    panels: list[list[list[str]]] = []
    start: int | None = None
    for index in range(width):
        if index in blank:
            if start is not None:
                panels.append([row[start:index] for row in rows])
                start = None
            continue
        if start is None:
            start = index
    if start is not None:
        panels.append([row[start:width] for row in rows])
    return [panel for panel in panels if any(not _row_blank(row) for row in panel)]


def _looks_like_table_panel(rows: list[list[str]]) -> bool:
    for row in rows[:8]:
        if len(_period_labels_in_row(row)) >= 2:
            return True
    width = len(rows[0]) if rows else 0
    if width < 2:
        return False
    numeric_cols = sum(
        1 for index in range(1, width) if _column_looks_numeric(rows, index)
    )
    content_rows = sum(1 for row in rows if not _row_blank(row))
    return numeric_cols >= 2 and content_rows >= 2


def _hstack_panels(
    left: list[list[str]],
    right: list[list[str]],
) -> list[list[str]]:
    """Rejoin a weak right strip (usually notes) onto the table on its left."""
    height = max(len(left), len(right))
    left_width = len(left[0]) if left else 0
    right_width = len(right[0]) if right else 0
    combined: list[list[str]] = []
    for index in range(height):
        left_row = left[index] if index < len(left) else [""] * left_width
        right_row = right[index] if index < len(right) else [""] * right_width
        combined.append([*left_row, "", *right_row])
    return combined


def _block_to_markdown(rows: list[list[str]], *, block_index: int) -> tuple[str, list[int]]:
    trimmed = _trim_columns(rows)
    if not trimmed:
        return "", []

    title, body = _split_title_row(trimmed)
    if not body:
        return "", []

    header_index = _header_row_index(body)
    header = body[header_index]
    data_rows = body[header_index + 1 :]
    if not data_rows:
        # Single header-ish row — still emit as a one-row table if useful.
        data_rows = []

    period_cols = _period_column_indexes(header)
    label_cols, value_cols, note_cols = _column_roles(header, data_rows, period_cols)

    if not value_cols and not period_cols:
        # Prose / notes block without a numeric grid.
        text_lines = [" ".join(cell for cell in row if cell).strip() for row in body]
        text = "\n\n".join(line for line in text_lines if line)
        if not text:
            return "", []
        heading = title or f"Block {block_index}"
        return f"## {heading}\n\n{text}", []

    headers = _build_headers(header, label_cols, value_cols)
    years = [year for cell in (header[i] for i in value_cols) for year in _years_in_text(cell)]
    table_lines = [
        "| " + " | ".join(_escape_cell(h) for h in headers) + " |",
        "| " + " | ".join(["---"] + [":---:" for _ in value_cols]) + " |",
    ]
    notes: list[str] = []
    for row in data_rows:
        if _row_blank(row):
            continue
        label = " ".join(row[i] for i in label_cols if row[i]).strip() or ""
        values = [_escape_cell(row[i] if i < len(row) else "") for i in value_cols]
        if not label and not any(values):
            continue
        table_lines.append("| " + " | ".join([_escape_cell(label), *values]) + " |")
        for index in note_cols:
            note = row[index].strip() if index < len(row) else ""
            if note and note not in notes:
                notes.append(note)

    heading = title or _guess_title_from_labels(data_rows, label_cols) or f"Table {block_index}"
    parts = [f"## {heading}", ""]
    parts.extend(table_lines)
    if notes:
        parts.append("")
        parts.append("Notes: " + "; ".join(notes))
    return "\n".join(parts), years


def _trim_columns(rows: list[list[str]]) -> list[list[str]]:
    if not rows:
        return []
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    left = 0
    while left < width and all(not row[left].strip() for row in normalized):
        left += 1
    right = width - 1
    while right >= left and all(not row[right].strip() for row in normalized):
        right -= 1
    if left > right:
        return []
    return [row[left : right + 1] for row in normalized]


def _split_title_row(rows: list[list[str]]) -> tuple[str | None, list[list[str]]]:
    if len(rows) < 2:
        return None, rows
    first = rows[0]
    non_empty = [cell for cell in first if cell.strip()]
    if len(non_empty) != 1:
        return None, rows
    if _period_labels_in_row(first):
        return None, rows
    if _period_labels_in_row(rows[1]) or _header_row_index(rows[1:]) == 0:
        return non_empty[0].strip(), rows[1:]
    return None, rows


def _header_row_index(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows[:5]):
        if len(_period_labels_in_row(row)) >= 2:
            return index
    return 0


def _period_labels_in_row(row: list[str]) -> list[str]:
    labels: list[str] = []
    for cell in row:
        for label in _period_labels_in_text(cell):
            if label not in labels:
                labels.append(label)
    return labels


def _period_labels_in_text(text: str) -> list[str]:
    labels: list[str] = []
    for match in _ISO_DATE.finditer(text):
        label = match.group(1)
        if label not in labels:
            labels.append(label)
    for match in _YEAR_LABEL.finditer(text):
        year, estimate = match.group(1), match.group(2)
        label = f"{year}{estimate.upper() if estimate else ''}"
        if label not in labels:
            labels.append(label)
    return labels


def _period_column_indexes(header: list[str]) -> list[int]:
    return [index for index, cell in enumerate(header) if _period_labels_in_text(cell)]


def _column_roles(
    header: list[str],
    data_rows: list[list[str]],
    period_cols: list[int],
) -> tuple[list[int], list[int], list[int]]:
    width = len(header)
    if period_cols:
        first_period = min(period_cols)
        last_period = max(period_cols)
        label_cols = list(range(0, first_period)) or [0]
        value_cols = list(range(first_period, last_period + 1))
        note_cols = list(range(last_period + 1, width))
        return label_cols, value_cols, note_cols

    # No year headers: treat first column as label, numeric-ish columns as values.
    label_cols = [0]
    value_cols = [
        index
        for index in range(1, width)
        if _column_looks_numeric(data_rows, index)
    ]
    note_cols = [
        index
        for index in range(1, width)
        if index not in value_cols
    ]
    return label_cols, value_cols or list(range(1, width)), note_cols


def _column_looks_numeric(rows: list[list[str]], index: int) -> bool:
    samples = [row[index] for row in rows if index < len(row) and row[index].strip()]
    if not samples:
        return False
    numeric = sum(1 for sample in samples if _looks_number(sample))
    return numeric / len(samples) >= 0.5


def _looks_number(text: str) -> bool:
    cleaned = text.replace("\u00a0", " ").replace(" ", "").replace("%", "").replace(",", ".")
    if cleaned in {"n.a.", "n/a", "na", "-", "—"}:
        return True
    try:
        float(cleaned)
        return True
    except ValueError:
        return False


def _build_headers(
    header: list[str],
    label_cols: list[int],
    value_cols: list[int],
) -> list[str]:
    label_name = " ".join(header[i] for i in label_cols if header[i]).strip() or "Post"
    value_names: list[str] = []
    for index in value_cols:
        cell = header[index].strip() if index < len(header) else ""
        labels = _period_labels_in_text(cell)
        value_names.append(labels[0] if labels else (cell or f"Col{index + 1}"))
    return [label_name, *value_names]


def _guess_title_from_labels(rows: list[list[str]], label_cols: list[int]) -> str | None:
    for row in rows[:3]:
        label = " ".join(row[i] for i in label_cols if i < len(row) and row[i]).strip()
        if label:
            return label
    return None


def _years_in_text(text: str) -> list[int]:
    years: list[int] = []
    for label in _period_labels_in_text(text):
        match = re.match(r"((?:19|20)\d{2})", label)
        if match:
            years.append(int(match.group(1)))
    return years


def _cell_to_text(cell: Cell) -> str:
    value = cell.value
    if value is None:
        return ""
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, datetime):
        return value.date().isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if value.is_integer():
            return str(int(value))
        text = f"{value:.6f}".rstrip("0").rstrip(".")
        return text
    text = str(value).strip()
    if text.startswith("="):
        # Formula with no cached value — keep as a hint, not as an amount.
        return ""
    return text


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()
