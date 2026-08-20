"""Key/value column roles for Damodaran-style form layouts.

Fiscal year grids and sensitivity matrices stay on the matrix merge path.
These helpers only classify and reshape repeating ``label → value → !note`` units.
"""

from __future__ import annotations

from enum import Enum

# How far a same-row label may look right for its value (blank columns between).
MAX_KV_FILL_GAP = 8

_CATEGORICAL = frozenset({"yes", "no", "y", "n", "true", "false", "ja", "nej"})


class ColumnRole(str, Enum):
    LABEL = "label"
    VALUE = "value"
    NOTE = "note"
    EMPTY = "empty"
    MIXED = "mixed"


def is_note(text: str) -> bool:
    return text.strip().startswith("!")


def is_categorical_answer(text: str) -> bool:
    return text.strip().lower() in _CATEGORICAL


def looks_number(text: str) -> bool:
    cleaned = text.replace("\u00a0", " ").replace(" ", "").replace("%", "").replace(",", ".")
    if cleaned.lower() in {"n.a.", "n/a", "na", "-", "—"}:
        return True
    try:
        float(cleaned)
    except ValueError:
        return False
    return True


def is_value_cell(text: str) -> bool:
    stripped = text.strip()
    if not stripped or is_note(stripped):
        return False
    return looks_number(stripped) or is_categorical_answer(stripped)


def is_label_cell(text: str) -> bool:
    stripped = text.strip()
    if not stripped or is_note(stripped) or is_value_cell(stripped):
        return False
    return True


def classify_column(rows: list[list[str]], col_index: int) -> ColumnRole:
    cells = [
        row[col_index].strip()
        for row in rows
        if col_index < len(row) and row[col_index].strip()
    ]
    if not cells:
        return ColumnRole.EMPTY
    notes = sum(1 for cell in cells if is_note(cell))
    values = sum(1 for cell in cells if is_value_cell(cell))
    labels = sum(1 for cell in cells if is_label_cell(cell))
    total = len(cells)
    if notes == total:
        return ColumnRole.NOTE
    if values == total:
        return ColumnRole.VALUE
    if labels == total:
        return ColumnRole.LABEL
    if notes + values == total and notes > 0:
        return ColumnRole.NOTE if notes >= values else ColumnRole.VALUE
    if labels >= max(values, notes) and labels >= (total + 1) // 2:
        return ColumnRole.LABEL
    if values >= max(labels, notes) and values >= (total + 1) // 2:
        return ColumnRole.VALUE
    return ColumnRole.MIXED


def column_roles(rows: list[list[str]]) -> list[ColumnRole]:
    width = max((len(row) for row in rows), default=0)
    return [classify_column(rows, index) for index in range(width)]


def find_kv_split_index(roles: list[ColumnRole]) -> int | None:
    """Return column index where a second KV unit starts, or None.

    Glued Options dual block: ``value | label | value`` → split at the label.
    Intact dual form: ``label | value | label | value`` → split at the second label.
    """
    seen_label = False
    seen_value = False
    for index, role in enumerate(roles):
        if role == ColumnRole.LABEL:
            if seen_value and not seen_label:
                # Left values glued onto the right-hand label column.
                return index
            if seen_label and seen_value:
                return index
            seen_label = True
        elif role == ColumnRole.VALUE:
            seen_value = True
    return None


def cells_are_value_or_note(cells: list[str]) -> bool:
    """True when every non-empty cell is a value or a ``!`` note."""
    populated = [cell for cell in cells if cell.strip()]
    if not populated:
        return False
    return all(is_value_cell(cell) or is_note(cell) for cell in populated)


def fill_row_local_kv(rows: list[list[str]], *, max_gap: int = MAX_KV_FILL_GAP) -> list[list[str]]:
    """Pull far-right values (and following ``!`` notes) next to same-row labels.

    Does not invent values across competing candidates in the look-ahead window.
    """
    if not rows:
        return rows
    width = max(len(row) for row in rows)
    grid = [row + [""] * (width - len(row)) for row in rows]
    for row in grid:
        _fill_one_row(row, max_gap=max_gap)
    return grid


def _fill_one_row(row: list[str], *, max_gap: int) -> None:
    width = len(row)
    col = 0
    while col < width:
        cell = row[col].strip()
        if not is_label_cell(cell):
            col += 1
            continue
        # Already has a near value (typical KV pair).
        if col + 1 < width and is_value_cell(row[col + 1]):
            col += 2
            continue
        value_col: int | None = None
        for look in range(col + 1, min(width, col + 1 + max_gap)):
            candidate = row[look].strip()
            if not candidate:
                continue
            if is_note(candidate):
                # Note before any value — stop; ambiguous.
                break
            if is_value_cell(candidate):
                value_col = look
                break
            # Another label in the window — leave this label alone.
            if is_label_cell(candidate):
                break
            break
        if value_col is None:
            col += 1
            continue
        # Competing second value between label and chosen value? nearest only; window clear.
        conflict = False
        for look in range(col + 1, value_col):
            if is_value_cell(row[look]):
                conflict = True
                break
        if conflict:
            col += 1
            continue
        note = ""
        note_col: int | None = None
        if value_col + 1 < width and is_note(row[value_col + 1]):
            note = row[value_col + 1]
            note_col = value_col + 1
        value = row[value_col]
        row[value_col] = ""
        if note_col is not None:
            row[note_col] = ""
        # Place immediately after the label when that slot is empty.
        target = col + 1
        if target < width and not row[target].strip():
            row[target] = value
            if note and target + 1 < width and not row[target + 1].strip():
                row[target + 1] = note
            elif note:
                # Keep note in original column if target+1 busy.
                if note_col is not None:
                    row[note_col] = note
        else:
            row[value_col] = value
            if note_col is not None:
                row[note_col] = note
        col = target + (2 if note else 1)
