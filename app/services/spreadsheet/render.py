"""Render detected subtables as Markdown + row narratives for RAG."""

from __future__ import annotations

from dataclasses import dataclass

from app.services.spreadsheet.periods import period_labels_in_row, period_labels_in_text, years_in_text
from app.services.spreadsheet.structure import SubtableRegion


@dataclass(frozen=True)
class RenderedTable:
    markdown: str
    years: list[int]


def render_sheet(sheet_title: str, regions: list[SubtableRegion]) -> tuple[str, list[int]]:
    sections: list[str] = [f"# {sheet_title}", ""]
    years: list[int] = []
    table_index = 0

    for region in regions:
        table_index += 1
        rendered = render_subtable(region, block_index=table_index)
        if not rendered:
            continue
        sections.append(rendered.markdown)
        sections.append("")
        years.extend(rendered.years)

    if len(sections) <= 2:
        return "", []
    return "\n".join(sections).strip() + "\n", years


def render_subtable(region: SubtableRegion, *, block_index: int) -> RenderedTable | None:
    title = _title_from_rows(region.title_rows)
    body = region.core_rows
    if not body:
        return None

    header_index = _header_row_index(body)
    header = body[header_index]
    data_rows = body[header_index + 1 :]

    period_cols = _period_column_indexes(header)
    if not period_cols:
        # Sensitivity matrices / key-value blocks — keep the grid literal.
        return _render_literal_grid(title, body, block_index=block_index)

    label_cols, value_cols, note_cols = _column_roles(header, data_rows, period_cols)
    period_confident = _period_header_confident(header, data_rows, period_cols, value_cols)

    if not value_cols and not period_cols:
        text_lines = [" ".join(cell for cell in row if cell).strip() for row in body]
        text = "\n\n".join(line for line in text_lines if line)
        if not text:
            return None
        heading = title or f"Block {block_index}"
        return RenderedTable(markdown=f"## {heading}\n\n{text}", years=[])

    headers = _build_headers(header, label_cols, value_cols)
    table_years = [
        year
        for cell in (header[i] for i in value_cols)
        for year in years_in_text(cell)
    ]
    table_lines = [
        "| " + " | ".join(_escape_cell(h) for h in headers) + " |",
        "| " + " | ".join(["---"] + [":---:" for _ in value_cols]) + " |",
    ]
    notes: list[str] = []
    narrative_lines: list[str] = []

    for row in data_rows:
        if _row_blank(row):
            continue
        label = " ".join(row[i] for i in label_cols if row[i]).strip() or ""
        values = [_escape_cell(row[i] if i < len(row) else "") for i in value_cols]
        if not label and not any(values):
            continue
        table_lines.append("| " + " | ".join([_escape_cell(label), *values]) + " |")
        if period_confident and label and any(values):
            pairs = [
                f"{col_name}={values[index]}"
                for index, col_name in enumerate(headers[1:])
                if index < len(values) and values[index]
            ]
            if pairs:
                narrative_lines.append(f"- {label}: " + ", ".join(pairs))
        for index in note_cols:
            note = row[index].strip() if index < len(row) else ""
            if note and note not in notes:
                notes.append(note)

    # Prefer an explicit banner title. Do not steal the first data-row label
    # (e.g. "Terminal tax") — that breaks retrieval for "General Assumptions".
    heading = title or f"Table {block_index}"
    parts = [f"## {heading}", ""]
    parts.extend(table_lines)
    if narrative_lines:
        parts.append("")
        parts.append("Rows:")
        parts.extend(narrative_lines)
    if notes:
        parts.append("")
        parts.append("Notes: " + "; ".join(notes))
    return RenderedTable(markdown="\n".join(parts), years=table_years)


def _render_literal_grid(
    title: str | None,
    rows: list[list[str]],
    *,
    block_index: int,
) -> RenderedTable | None:
    """Emit a cell grid as markdown without fiscal Post/year reshaping."""
    trimmed = [_trim_trailing_empties(row) for row in rows if not _row_blank(row)]
    if not trimmed:
        return None
    width = max(len(row) for row in trimmed)
    normalized = [row + [""] * (width - len(row)) for row in trimmed]
    # Drop fully empty columns.
    keep = [index for index in range(width) if any(row[index].strip() for row in normalized)]
    if not keep:
        return None
    grid = [[row[index] for index in keep] for row in normalized]
    col_count = len(grid[0])
    table_lines = [
        "| " + " | ".join(_escape_cell(cell) for cell in grid[0]) + " |",
        "| " + " | ".join(["---"] * col_count) + " |",
    ]
    for row in grid[1:]:
        table_lines.append("| " + " | ".join(_escape_cell(cell) for cell in row) + " |")
    heading = title or f"Table {block_index}"
    return RenderedTable(markdown=f"## {heading}\n\n" + "\n".join(table_lines), years=[])


def _trim_trailing_empties(row: list[str]) -> list[str]:
    end = len(row)
    while end > 0 and not row[end - 1].strip():
        end -= 1
    return row[:end]


def _period_header_confident(
    header: list[str],
    data_rows: list[list[str]],
    period_cols: list[int],
    value_cols: list[int],
) -> bool:
    """Only emit row narratives when period labels align with numeric columns."""
    if len(period_cols) < 2:
        return False
    if len(value_cols) < 2:
        return False
    numeric_under_periods = sum(
        1 for index in value_cols if _column_looks_numeric(data_rows, index)
    )
    return numeric_under_periods >= 2


def _title_from_rows(title_rows: list[list[str]]) -> str | None:
    for row in title_rows:
        non_empty = [cell.strip() for cell in row if cell.strip()]
        if len(non_empty) == 1:
            return non_empty[0]
    return None


def _header_row_index(rows: list[list[str]]) -> int:
    for index, row in enumerate(rows[:5]):
        if len(period_labels_in_row(row)) >= 2:
            return index
    return 0


def _period_column_indexes(header: list[str]) -> list[int]:
    return [index for index, cell in enumerate(header) if period_labels_in_text(cell)]


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

    label_cols = [0]
    value_cols = [
        index for index in range(1, width) if _column_looks_numeric(data_rows, index)
    ]
    note_cols = [index for index in range(1, width) if index not in value_cols]
    return label_cols, value_cols or list(range(1, width)), note_cols


def _build_headers(
    header: list[str],
    label_cols: list[int],
    value_cols: list[int],
) -> list[str]:
    label_name = " ".join(header[i] for i in label_cols if header[i]).strip() or "Post"
    value_names: list[str] = []
    for index in value_cols:
        cell = header[index].strip() if index < len(header) else ""
        labels = period_labels_in_text(cell)
        value_names.append(labels[0] if labels else (cell or f"Col{index + 1}"))
    return [label_name, *value_names]


def _row_blank(row: list[str]) -> bool:
    return all(not cell.strip() for cell in row)


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


def _escape_cell(text: str) -> str:
    return text.replace("|", "\\|").replace("\n", " ").strip()
