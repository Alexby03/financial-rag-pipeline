"""Connected-component subtable detection on spreadsheet grids."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import pandas as pd

from app.services.spreadsheet.kv import (
    MAX_KV_FILL_GAP,
    cells_are_value_or_note,
    column_roles,
    fill_row_local_kv,
    find_kv_split_index,
    is_categorical_answer,
    is_label_cell,
    is_note,
    is_value_cell,
)
from app.services.spreadsheet.periods import period_labels_in_row

CellCoord = tuple[int, int]


@dataclass(frozen=True)
class SubtableRegion:
    """A contiguous populated region with optional title rows split out."""

    title_rows: list[list[str]]
    core_rows: list[list[str]]


@dataclass
class _Panel:
    rows: list[list[str]]
    start_row: int
    end_row: int
    start_col: int
    end_col: int


def detect_subtables(grid: list[list[str]]) -> list[SubtableRegion]:
    """Split a sheet grid into subtables via 4-connected components."""
    if not grid:
        return []

    df = _grid_to_dataframe(grid)
    panels = _panels_from_components(df)
    regions: list[SubtableRegion] = []
    for panel in panels:
        title_rows, core_rows = _split_title_and_core(panel.rows)
        body = core_rows if core_rows else title_rows
        # Drop answer-only islands (Yes/No/bare numbers) with no title and no labels.
        if not title_rows and _is_label_less_value_dump(body):
            continue
        if core_rows:
            regions.append(SubtableRegion(title_rows=title_rows, core_rows=core_rows))
        elif title_rows:
            regions.append(SubtableRegion(title_rows=[], core_rows=title_rows))
    return regions


def _is_label_less_value_dump(rows: list[list[str]]) -> bool:
    """True for answer-only islands like ``Yes/No/No`` with no question text."""
    cells = [cell for row in rows for cell in row if cell.strip()]
    if not cells:
        return True
    return all(is_value_cell(cell) or is_note(cell) for cell in cells)

def _grid_to_dataframe(grid: list[list[str]]) -> pd.DataFrame:
    width = max(len(row) for row in grid)
    normalized = [row + [""] * (width - len(row)) for row in grid]
    return pd.DataFrame(normalized, dtype=str)


def _is_populated(value: object) -> bool:
    return bool(str(value).strip())


def _connected_components(df: pd.DataFrame) -> list[set[CellCoord]]:
    rows, cols = df.shape
    visited: set[CellCoord] = set()
    components: list[set[CellCoord]] = []

    for row in range(rows):
        for col in range(cols):
            if (row, col) in visited or not _is_populated(df.iat[row, col]):
                continue
            component: set[CellCoord] = set()
            queue: deque[CellCoord] = deque([(row, col)])
            while queue:
                current = queue.popleft()
                if current in visited:
                    continue
                r, c = current
                if not (0 <= r < rows and 0 <= c < cols):
                    continue
                if not _is_populated(df.iat[r, c]):
                    continue
                visited.add(current)
                component.add(current)
                for nr, nc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                    if (nr, nc) not in visited:
                        queue.append((nr, nc))
            if component:
                components.append(component)
    return components


@dataclass(frozen=True)
class _ComponentBox:
    cells: set[CellCoord]
    min_row: int
    max_row: int
    min_col: int
    max_col: int

    @classmethod
    def from_cells(cls, cells: set[CellCoord]) -> _ComponentBox:
        rows = [row for row, _ in cells]
        cols = [col for _, col in cells]
        return cls(
            cells=cells,
            min_row=min(rows),
            max_row=max(rows),
            min_col=min(cols),
            max_col=max(cols),
        )


# Banner titles may sit a few blank rows above their table (DCF "General Assumptions").
_MAX_TITLE_GAP_ROWS = 5
# Allow titles one column left of the table (banner in A, Case/Aberlyn in B:C).
_MAX_TITLE_COL_GAP = 3
# Blank-row gaps inside one year-grid (revenue block then D&A block).
_MAX_VERTICAL_STITCH_GAP = 2


def _panels_from_components(df: pd.DataFrame) -> list[_Panel]:
    boxes = sorted(
        [_ComponentBox.from_cells(cells) for cells in _connected_components(df)],
        key=lambda box: (box.min_row, box.min_col),
    )
    panels = [_panel_from_box(df, box) for box in boxes if box.cells]
    panels = _stitch_vertical_fragments(panels)
    panels = _drop_contained_panels(panels)
    panels = _merge_side_by_side_matrices(panels)
    panels = _merge_title_panels(panels)
    panels = _attach_orphan_titles(panels)
    panels = _merge_side_by_side_matrices(panels)
    panels = _merge_note_strips(panels)
    panels = _split_adjacent_kv_panels(panels)
    panels = _fill_row_local_kv_panels(panels)
    # Hosts may grow during merges; re-absorb crumbs that are now inside / adjacent.
    return _drop_contained_panels(panels)


def _panel_from_box(df: pd.DataFrame, box: _ComponentBox) -> _Panel:
    rows: list[list[str]] = []
    for row in range(box.min_row, box.max_row + 1):
        rows.append([str(df.iat[row, col]).strip() for col in range(box.min_col, box.max_col + 1)])
    trimmed = _trim_columns(rows)
    if not trimmed:
        return _Panel(
            rows=[],
            start_row=box.min_row,
            end_row=box.min_row,
            start_col=box.min_col,
            end_col=box.max_col,
        )
    # Keep absolute start_col aligned with the first retained column after left-trim.
    width = max(len(row) for row in rows)
    normalized = [row + [""] * (width - len(row)) for row in rows]
    left = 0
    while left < width and all(not row[left].strip() for row in normalized):
        left += 1
    start_col = box.min_col + left
    return _Panel(
        rows=trimmed,
        start_row=box.min_row,
        end_row=box.max_row,
        start_col=start_col,
        end_col=start_col + len(trimmed[0]) - 1,
    )


def _columns_overlap(left: _Panel, right: _Panel) -> bool:
    return left.start_col <= right.end_col and right.start_col <= left.end_col


def _rows_overlap(left: _Panel, right: _Panel) -> bool:
    return left.start_row <= right.end_row and right.start_row <= left.end_row


def _is_directly_below(upper: _Panel, lower: _Panel) -> bool:
    return lower.start_row == upper.end_row + 1


def _merge_title_panels(panels: list[_Panel]) -> list[_Panel]:
    """Merge a lone title row into the table on the next row (same column band)."""
    if len(panels) < 2:
        return panels

    merged: list[_Panel] = []
    index = 0
    while index < len(panels):
        current = panels[index]
        if index + 1 < len(panels):
            below = panels[index + 1]
            if (
                _is_title_only_panel(current)
                and _is_directly_below(current, below)
                and _columns_overlap(current, below)
            ):
                merged.append(
                    _Panel(
                        rows=[*current.rows, *below.rows],
                        start_row=current.start_row,
                        end_row=below.end_row,
                        start_col=min(current.start_col, below.start_col),
                        end_col=max(current.end_col, below.end_col),
                    )
                )
                index += 2
                continue
        merged.append(current)
        index += 1
    return merged


def _is_title_only_panel(panel: _Panel) -> bool:
    if not panel.rows:
        return False
    if len(panel.rows) > 2:
        return False
    if _looks_like_table_panel(panel.rows):
        return False
    return all(_is_single_cell_row(row) or _row_blank(row) for row in panel.rows if row)


def _column_gap(left: _Panel, right: _Panel) -> int:
    if _columns_overlap(left, right):
        return 0
    if left.end_col < right.start_col:
        return right.start_col - left.end_col - 1
    return left.start_col - right.end_col - 1


def _strongly_column_aligned(left: _Panel, right: _Panel) -> bool:
    """True when panels share essentially the same horizontal band."""
    if not _columns_overlap(left, right):
        return False
    overlap = min(left.end_col, right.end_col) - max(left.start_col, right.start_col) + 1
    left_width = left.end_col - left.start_col + 1
    right_width = right.end_col - right.start_col + 1
    return overlap >= 0.8 * min(left_width, right_width)


def _vstack_panels(upper: _Panel, lower: _Panel) -> _Panel:
    start_col = min(upper.start_col, lower.start_col)
    end_col = max(upper.end_col, lower.end_col)
    width = end_col - start_col + 1

    def expand(panel: _Panel) -> list[list[str]]:
        left_pad = panel.start_col - start_col
        right_pad = end_col - panel.end_col
        return [[""] * left_pad + row + [""] * right_pad for row in panel.rows]

    gap = max(0, lower.start_row - upper.end_row - 1)
    blank = [[""] * width for _ in range(gap)]
    return _Panel(
        rows=[*expand(upper), *blank, *expand(lower)],
        start_row=upper.start_row,
        end_row=lower.end_row,
        start_col=start_col,
        end_col=end_col,
    )


def _stitch_vertical_fragments(panels: list[_Panel]) -> list[_Panel]:
    """Rejoin year-grid slices split by blank rows (revenue then D&A then CAPEX)."""
    if len(panels) < 2:
        return panels

    remaining = sorted(panels, key=lambda panel: (panel.start_col, panel.start_row))
    stitched: list[_Panel] = []
    while remaining:
        current = remaining.pop(0)
        progressed = True
        while progressed:
            progressed = False
            for index, other in enumerate(remaining):
                if not _strongly_column_aligned(current, other):
                    continue
                if other.start_row > current.end_row:
                    gap = other.start_row - current.end_row - 1
                    if gap <= _MAX_VERTICAL_STITCH_GAP and _is_vertical_continuation(
                        current, other
                    ):
                        current = _vstack_panels(current, other)
                        remaining.pop(index)
                        progressed = True
                        break
                elif current.start_row > other.end_row:
                    gap = current.start_row - other.end_row - 1
                    if gap <= _MAX_VERTICAL_STITCH_GAP and _is_vertical_continuation(
                        other, current
                    ):
                        current = _vstack_panels(other, current)
                        remaining.pop(index)
                        progressed = True
                        break
        stitched.append(current)
    return sorted(stitched, key=lambda panel: (panel.start_row, panel.start_col))


def _is_vertical_continuation(upper: _Panel, lower: _Panel) -> bool:
    """True when lower continues upper after blank rows — not a new stacked table.

    Year grids: D&A under revenue shares columns but has no new year header.
    Sensitivity cubes: Discount Rate / Revenue adj headers sit above the value matrix.
    A second P&L/FCF block with its own ``1991 | 1992`` header must stay separate.
    """
    lower_has_periods = any(len(period_labels_in_row(row)) >= 2 for row in lower.rows[:3])
    if lower_has_periods:
        return False
    if any(len(period_labels_in_row(row)) >= 2 for row in upper.rows[:5]):
        return True
    if _has_year_period_header(upper) or _has_year_period_header(lower):
        return False
    return _looks_like_table_panel(upper.rows) or _looks_like_table_panel(lower.rows)


def _panel_cell_count(panel: _Panel) -> int:
    return sum(1 for row in panel.rows for cell in row if cell.strip())


def _title_applies_to_table(title: _Panel, table: _Panel) -> bool:
    """Banner must label its own column band — never a table entirely to its left.

    ``Equity Value Sensitivity`` sits to the right of the DCF year grid; attaching it
    to a left-hand D&A fragment produced ``Equity Value Sensitivity > Depreciation``.
    ``General Assumptions`` (left of Case/Aberlyn) remains allowed.
    """
    if title.start_col > table.end_col:
        return False
    if _columns_overlap(title, table):
        return True
    if title.end_col < table.start_col:
        return _column_gap(title, table) <= _MAX_TITLE_COL_GAP
    return False


def _attach_orphan_titles(panels: list[_Panel]) -> list[_Panel]:
    """Attach floating banner titles to the nearest table below in the same column band.

    DCF models put ``General Assumptions`` a few blank rows above the Case/Aberlyn
    block; adjacent-only title merge misses that. Do not attach a right-hand banner
    onto a left-hand year grid.
    """
    if len(panels) < 2:
        return panels

    used_targets: set[int] = set()
    attach: dict[int, int] = {}  # title_index -> table_index

    for title_index, title in enumerate(panels):
        if not _is_title_only_panel(title):
            continue
        best_index: int | None = None
        best_distance: int | None = None
        best_score = -1
        for table_index, table in enumerate(panels):
            if table_index == title_index or table_index in used_targets:
                continue
            if not table.rows or _is_title_only_panel(table):
                continue
            if table.start_row <= title.end_row:
                continue
            gap = table.start_row - title.end_row - 1
            if gap > _MAX_TITLE_GAP_ROWS:
                continue
            if not _title_applies_to_table(title, table):
                continue
            score = _panel_cell_count(table)
            # Prefer nearer tables; if equally near, prefer the denser matrix so a
            # banner does not latch onto a header-only fragment above the real cube.
            if (
                best_distance is None
                or gap < best_distance
                or (gap == best_distance and score > best_score)
                or (best_distance is not None and gap <= best_distance + 2 and score > best_score * 2)
            ):
                best_distance = gap
                best_index = table_index
                best_score = score
        if best_index is not None:
            attach[title_index] = best_index
            used_targets.add(best_index)

    if not attach:
        return panels

    merged: list[_Panel] = []
    skip: set[int] = set()
    for index, panel in enumerate(panels):
        if index in skip:
            continue
        if index in attach:
            table_index = attach[index]
            table = panels[table_index]
            skip.add(table_index)
            merged.append(_prepend_title_rows(panel, table))
            continue
        if index in used_targets:
            continue
        merged.append(panel)
    return merged


def _prepend_title_rows(title: _Panel, table: _Panel) -> _Panel:
    """Put banner rows above a table, padded to the table width so it stays a table."""
    table_width = max((len(row) for row in table.rows), default=1)
    title_rows: list[list[str]] = []
    for row in title.rows:
        non_empty = [cell for cell in row if cell.strip()]
        label = non_empty[0] if non_empty else ""
        title_rows.append([label] + [""] * (table_width - 1))
    return _Panel(
        rows=[*title_rows, *table.rows],
        start_row=title.start_row,
        end_row=table.end_row,
        start_col=min(title.start_col, table.start_col),
        end_col=max(title.end_col, table.end_col),
    )


def _has_year_period_header(panel: _Panel) -> bool:
    return any(len(period_labels_in_row(row)) >= 2 for row in panel.rows[:5])


def _drop_contained_panels(panels: list[_Panel]) -> list[_Panel]:
    """Paint tiny interior panels into their host, then remove the crumbs.

    Discount-rate scalars (``12.8%``) sit inside the sensitivity matrix bbox next to
    ``Discount Rate:``. Dropping them without painting made the model guess the rate
    while Revenue adj. columns survived.
    """
    if len(panels) < 2:
        return panels

    hosts: list[_Panel] = []
    for panel in panels:
        container = _find_container(panel, panels)
        if container is None:
            hosts.append(panel)
            continue
        _paint_panel_into(container, panel)
    return hosts


def _find_container(panel: _Panel, panels: list[_Panel]) -> _Panel | None:
    panel_area = max(
        1, (panel.end_row - panel.start_row + 1) * (panel.end_col - panel.start_col + 1)
    )
    best: _Panel | None = None
    best_area = panel_area
    for other in panels:
        if other is panel:
            continue
        if not (
            panel.start_row >= other.start_row
            and panel.end_row <= other.end_row
            and panel.start_col >= other.start_col
            and panel.end_col <= other.end_col
        ):
            continue
        other_area = (other.end_row - other.start_row + 1) * (other.end_col - other.start_col + 1)
        if other_area > best_area:
            best = other
            best_area = other_area
    return best


def _paint_panel_into(host: _Panel, guest: _Panel) -> None:
    """Write guest cells into the host grid at absolute sheet coordinates."""
    for row_offset, guest_row in enumerate(guest.rows):
        abs_row = guest.start_row + row_offset
        host_row = abs_row - host.start_row
        if not (0 <= host_row < len(host.rows)):
            continue
        for col_offset, value in enumerate(guest_row):
            if not value.strip():
                continue
            abs_col = guest.start_col + col_offset
            host_col = abs_col - host.start_col
            if not (0 <= host_col < len(host.rows[host_row])):
                continue
            existing = host.rows[host_row][host_col].strip()
            if not existing:
                if _fold_rate_into_discount_label(host.rows[host_row], host_col, value):
                    continue
                host.rows[host_row][host_col] = value
            elif existing.rstrip(":").strip().lower() == "discount rate" and _looks_like_rate_value(
                value
            ):
                host.rows[host_row][host_col] = f"{existing.rstrip()} {value.strip()}"


def _fold_rate_into_discount_label(row: list[str], value_col: int, value: str) -> bool:
    """Attach ``12.8%`` to a nearby ``Discount Rate:`` cell on the same row."""
    if not _looks_like_rate_value(value):
        return False
    for index in range(max(0, value_col - 3), value_col):
        label = row[index].strip()
        if label.rstrip(":").strip().lower() == "discount rate":
            row[index] = f"{label.rstrip()} {value.strip()}"
            return True
    return False


def _looks_like_rate_value(text: str) -> bool:
    cleaned = text.strip().replace("%", "").replace(",", ".")
    try:
        number = float(cleaned)
    except ValueError:
        return False
    return 0 < number < 100


def _merge_side_by_side_matrices(panels: list[_Panel]) -> list[_Panel]:
    """HStack adjacent non-year matrices that share a row band (sensitivity cubes).

    Also absorb a left-hand label strip (``Terminal growth``) into the matrix on its right.
    Never merge into a fiscal year grid.
    """
    if len(panels) < 2:
        return panels

    ordered = sorted(panels, key=lambda panel: (panel.start_row, panel.start_col))
    merged: list[_Panel] = []
    for panel in ordered:
        if not panel.rows:
            continue
        attached = False
        for index in range(len(merged) - 1, -1, -1):
            host = merged[index]
            if _can_hstack_matrices(host, panel):
                merged[index] = _hstack_panels(host, panel)
                attached = True
                break
            if _can_absorb_left_label(host, panel):
                merged[index] = _hstack_panels(host, panel)
                attached = True
                break
            if _can_absorb_left_label(panel, host):
                merged[index] = _hstack_panels(panel, host)
                attached = True
                break
            if _can_absorb_right_value_crumb(host, panel):
                merged[index] = _union_panels(host, panel)
                attached = True
                break
            # Stop looking further back once we pass a different row band.
            if host.end_row < panel.start_row and not _rows_overlap(host, panel):
                break
        if not attached:
            merged.append(panel)
    return merged


def _is_categorical_answer(text: str) -> bool:
    return is_categorical_answer(text)


def _is_value_crumb(panel: _Panel) -> bool:
    """Answer column island: numbers/Yes-No and optional ``!`` notes — no question labels."""
    cells = [cell for row in panel.rows for cell in row if cell.strip()]
    if not cells or len(cells) > 24:
        return False
    if _has_year_period_header(panel):
        return False
    width = panel.end_col - panel.start_col + 1
    if width > 3:
        return False
    if any(is_label_cell(cell) for cell in cells):
        return False
    return cells_are_value_or_note(cells)


def _is_numeric_orphan(panel: _Panel) -> bool:
    """Backward-compatible alias for numeric-only crumbs."""
    cells = [cell for row in panel.rows for cell in row if cell.strip()]
    return bool(cells) and _is_value_crumb(panel) and all(
        is_value_cell(cell) and not is_note(cell) for cell in cells
    )


def _is_text_label_block(panel: _Panel) -> bool:
    """Label column(s) with little/no numeric payload — key side of a key/value pair."""
    cells = [cell for row in panel.rows for cell in row if cell.strip()]
    if not cells:
        return False
    if _has_year_period_header(panel):
        return False
    textish = sum(1 for cell in cells if is_label_cell(cell))
    return textish == len(cells)


def _host_has_text_on_overlap(host: _Panel, crumb: _Panel) -> bool:
    start = max(host.start_row, crumb.start_row)
    end = min(host.end_row, crumb.end_row)
    for abs_row in range(start, end + 1):
        host_index = abs_row - host.start_row
        if not (0 <= host_index < len(host.rows)):
            continue
        for cell in host.rows[host_index]:
            if is_label_cell(cell):
                return True
    return False


def _can_absorb_right_value_crumb(host: _Panel, crumb: _Panel) -> bool:
    """Join labels with answer crumbs when blank columns break 4-connectivity.

    Covers Damodaran Options (``14.32``) and Input Sheet (``Yes``/``No`` two columns
    right of the question). Wider Restated gaps use ``_fill_row_local_kv_panels``.
    """
    if crumb.start_col <= host.end_col:
        return False
    if _column_gap(host, crumb) > _MAX_TITLE_COL_GAP:
        return False
    if not _rows_overlap(host, crumb):
        return False
    if not _is_value_crumb(crumb):
        return False
    if _has_year_period_header(host):
        return False
    if _is_label_value_strip(host) or _is_text_label_block(host):
        return True
    return _host_has_text_on_overlap(host, crumb)


def _split_adjacent_kv_panels(panels: list[_Panel]) -> list[_Panel]:
    """Break ``value | label | value`` glue from side-by-side Damodaran KV columns."""
    result: list[_Panel] = []
    for panel in panels:
        if (
            not panel.rows
            or _has_year_period_header(panel)
            or _looks_like_table_panel(panel.rows)
        ):
            result.append(panel)
            continue
        roles = column_roles(panel.rows)
        split_at = find_kv_split_index(roles)
        if split_at is None or split_at <= 0:
            result.append(panel)
            continue
        left_rows = [row[:split_at] for row in panel.rows]
        right_rows = [row[split_at:] for row in panel.rows]
        left_rows = _trim_columns(left_rows)
        right_rows = _trim_columns(right_rows)
        if left_rows:
            result.append(
                _Panel(
                    rows=left_rows,
                    start_row=panel.start_row,
                    end_row=panel.end_row,
                    start_col=panel.start_col,
                    end_col=panel.start_col + split_at - 1,
                )
            )
        if right_rows:
            raw_right = [row[split_at:] for row in panel.rows]
            width = max(len(row) for row in raw_right)
            normalized = [row + [""] * (width - len(row)) for row in raw_right]
            leading = 0
            while leading < width and all(not row[leading].strip() for row in normalized):
                leading += 1
            right_start = panel.start_col + split_at + leading
            result.append(
                _Panel(
                    rows=right_rows,
                    start_row=panel.start_row,
                    end_row=panel.end_row,
                    start_col=right_start,
                    end_col=right_start + len(right_rows[0]) - 1,
                )
            )
    return result


def _fill_row_local_kv_panels(panels: list[_Panel]) -> list[_Panel]:
    """Pair labels with far-right values (and ``!`` notes); then compact within panels."""
    if len(panels) < 1:
        return panels

    # First: union label hosts with value/note crumbs across a wide but capped gap.
    ordered = sorted(panels, key=lambda panel: (panel.start_row, panel.start_col))
    merged: list[_Panel] = []
    for panel in ordered:
        if not panel.rows:
            continue
        attached = False
        for index in range(len(merged) - 1, -1, -1):
            host = merged[index]
            if not _rows_overlap(host, panel):
                continue
            if _can_pair_kv_across_gap(host, panel):
                merged[index] = _union_panels(host, panel)
                attached = True
                break
        if not attached:
            merged.append(panel)

    filled: list[_Panel] = []
    for panel in merged:
        # Year grids / sensitivity matrices: keep cells put; only compact form panels.
        if (
            not panel.rows
            or _has_year_period_header(panel)
            or _looks_like_table_panel(panel.rows)
        ):
            filled.append(panel)
            continue
        new_rows = fill_row_local_kv(panel.rows, max_gap=MAX_KV_FILL_GAP)
        trimmed = _trim_columns(new_rows)
        if not trimmed:
            filled.append(panel)
            continue
        width = max(len(row) for row in new_rows)
        normalized = [row + [""] * (width - len(row)) for row in new_rows]
        left = 0
        while left < width and all(not row[left].strip() for row in normalized):
            left += 1
        filled.append(
            _Panel(
                rows=trimmed,
                start_row=panel.start_row,
                end_row=panel.end_row,
                start_col=panel.start_col + left,
                end_col=panel.start_col + left + len(trimmed[0]) - 1,
            )
        )
    return filled


def _can_pair_kv_across_gap(host: _Panel, crumb: _Panel) -> bool:
    """Label strip on the left + value/note island on the right (Restated Financials)."""
    if crumb.start_col <= host.end_col:
        return False
    if _column_gap(host, crumb) > MAX_KV_FILL_GAP:
        return False
    if not _rows_overlap(host, crumb):
        return False
    if _has_year_period_header(host) or _has_year_period_header(crumb):
        return False
    if not _is_value_crumb(crumb):
        return False
    return _is_text_label_block(host) or _is_label_value_strip(host) or _host_has_text_on_overlap(
        host, crumb
    )


def _union_panels(left: _Panel, right: _Panel) -> _Panel:
    """Merge panels onto a shared absolute grid (no extra spacer column)."""
    start_row = min(left.start_row, right.start_row)
    end_row = max(left.end_row, right.end_row)
    start_col = min(left.start_col, right.start_col)
    end_col = max(left.end_col, right.end_col)
    width = end_col - start_col + 1
    height = end_row - start_row + 1
    grid = [["" for _ in range(width)] for _ in range(height)]

    def _paint(panel: _Panel) -> None:
        for row_offset, row in enumerate(panel.rows):
            abs_row = panel.start_row + row_offset
            for col_offset, value in enumerate(row):
                if not value.strip():
                    continue
                abs_col = panel.start_col + col_offset
                grid[abs_row - start_row][abs_col - start_col] = value

    _paint(left)
    _paint(right)
    return _Panel(
        rows=grid,
        start_row=start_row,
        end_row=end_row,
        start_col=start_col,
        end_col=end_col,
    )


def _can_hstack_matrices(left: _Panel, right: _Panel) -> bool:
    if right.start_col <= left.end_col:
        return False
    if _column_gap(left, right) > _MAX_TITLE_COL_GAP:
        return False
    if not _rows_overlap(left, right):
        return False
    if _has_year_period_header(left) or _has_year_period_header(right):
        return False
    if _is_label_value_strip(left) or _is_label_value_strip(right):
        return False
    return _looks_like_table_panel(left.rows) and _looks_like_table_panel(right.rows)


def _is_label_value_strip(panel: _Panel) -> bool:
    """Narrow key/value strip like Terminal growth | 1.0% — not a year grid."""
    width = panel.end_col - panel.start_col + 1
    if width > 3 or width < 1:
        return False
    if _has_year_period_header(panel):
        return False
    if not panel.rows:
        return False
    first_col_text = sum(1 for row in panel.rows if row and row[0].strip() and not _looks_number(row[0]))
    return first_col_text >= 1 and len(panel.rows) >= 2


def _can_absorb_left_label(label: _Panel, matrix: _Panel) -> bool:
    """Narrow label strip to the left of a value matrix, same row band."""
    if matrix.start_col <= label.end_col:
        return False
    if _column_gap(label, matrix) > _MAX_TITLE_COL_GAP:
        return False
    if not _rows_overlap(label, matrix):
        return False
    if _has_year_period_header(label) or _has_year_period_header(matrix):
        return False
    return _is_label_value_strip(label) and _looks_like_table_panel(matrix.rows)


def _merge_note_strips(panels: list[_Panel]) -> list[_Panel]:
    """Keep side-by-side tables separate; glue note strips onto the table on their left."""
    merged: list[_Panel] = []
    for panel in panels:
        if not panel.rows:
            continue
        if _looks_like_table_panel(panel.rows) or not merged:
            merged.append(panel)
            continue
        left = merged[-1]
        # Do not glue two real tables together (year grid + sensitivity matrix).
        if _looks_like_table_panel(panel.rows) and (
            _has_year_period_header(left) or _has_year_period_header(panel)
        ):
            merged.append(panel)
            continue
        if (
            _looks_like_table_panel(left.rows)
            and panel.start_col > left.end_col
            and _rows_overlap(left, panel)
        ):
            merged[-1] = _hstack_panels(left, panel)
        else:
            merged.append(panel)
    return merged


def _hstack_panels(left: _Panel, right: _Panel) -> _Panel:
    """Rejoin a weak right strip (usually notes) onto the table on its left."""
    start = min(left.start_row, right.start_row)
    end = max(left.end_row, right.end_row)
    left_width = len(left.rows[0]) if left.rows else 0
    right_width = len(right.rows[0]) if right.rows else 0
    combined: list[list[str]] = []
    for row in range(start, end + 1):
        left_index = row - left.start_row
        right_index = row - right.start_row
        left_row = left.rows[left_index] if 0 <= left_index < len(left.rows) else [""] * left_width
        right_row = (
            right.rows[right_index] if 0 <= right_index < len(right.rows) else [""] * right_width
        )
        combined.append([*left_row, "", *right_row])
    return _Panel(
        rows=combined,
        start_row=start,
        end_row=end,
        start_col=left.start_col,
        end_col=right.end_col,
    )


def _looks_like_table_panel(rows: list[list[str]]) -> bool:
    for row in rows[:8]:
        if len(period_labels_in_row(row)) >= 2:
            return True
    width = max((len(row) for row in rows), default=0)
    if width < 2:
        return False
    numeric_cols = sum(1 for index in range(1, width) if _column_looks_numeric(rows, index))
    content_rows = sum(1 for row in rows if not _row_blank(row))
    return numeric_cols >= 2 and content_rows >= 2


def _split_title_and_core(rows: list[list[str]]) -> tuple[list[list[str]], list[list[str]]]:
    """Split leading banner title rows from the table body.

    Only peel true banners (``General Assumptions``, ``Equity Value Sensitivity``).
    Do not peel ``Discount Rate:`` / ``Revenue adj.`` — those belong in the grid.
    """
    leading: list[list[str]] = []
    index = 0
    while index < len(rows) and _is_banner_title_row(rows[index]):
        leading.append(rows[index])
        index += 1
    core = rows[index:]
    if leading and not core:
        return [], leading
    return leading, core


def _is_banner_title_row(row: list[str]) -> bool:
    if not _is_single_cell_row(row):
        return False
    text = next(cell.strip() for cell in row if cell.strip())
    lowered = text.lower().rstrip(":")
    if lowered in {"revenue adj", "revenue adj.", "discount rate", "terminal growth"}:
        return False
    if text.endswith(":") and len(text) <= 24:
        return False
    return True


def _is_single_cell_row(row: list[str]) -> bool:
    non_empty = [cell for cell in row if cell.strip()]
    return len(non_empty) == 1


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
