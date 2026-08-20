"""Period / fiscal-year label detection for spreadsheet tables."""

from __future__ import annotations

import re

_YEAR_LABEL = re.compile(r"(?i)\b(?:FY\s*)?((?:19|20)\d{2})([Ee])?\b")
_ISO_DATE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_MIN_FISCAL_YEAR = 1990


def period_labels_in_text(text: str) -> list[str]:
    labels: list[str] = []
    for match in _ISO_DATE.finditer(text):
        label = match.group(1)
        if label not in labels:
            labels.append(label)
    for match in _YEAR_LABEL.finditer(text):
        year, estimate = match.group(1), match.group(2)
        if int(year) < _MIN_FISCAL_YEAR:
            continue
        label = f"{year}{estimate.upper() if estimate else ''}"
        if label not in labels:
            labels.append(label)
    return labels


def period_labels_in_row(row: list[str]) -> list[str]:
    labels: list[str] = []
    for cell in row:
        for label in period_labels_in_text(cell):
            if label not in labels:
                labels.append(label)
    return labels


def years_in_text(text: str) -> list[int]:
    years: list[int] = []
    for label in period_labels_in_text(text):
        match = re.match(r"((?:19|20)\d{2})", label)
        if match:
            years.append(int(match.group(1)))
    return years
