"""Deterministic heading cleanup after per-page OCR.

OCR can mis-guess `#` depth (`### 3.` under an earlier `2.3`). Numbered sections in Financial
reports (2., 2.1., 3.) give a local signal: depth follows the decimal outline.
"""

import re

from app.models.parsed import ParsedPage

_MULTI_SECTION = re.compile(r"^(\d+(?:\.\d+)+)(?:\.\s+|\s+)(.+)$")
_TOP_SECTION = re.compile(r"^(\d+)\.\s+(.+)$")
_MARKDOWN_HEADING = re.compile(r"^(#{1,6})\s+(.+)$")


def _parse_numbered_heading(body: str) -> tuple[str, str] | None:
    multi = _MULTI_SECTION.match(body)
    if multi:
        return multi.group(1), multi.group(2)
    top = _TOP_SECTION.match(body)
    if top:
        return top.group(1), top.group(2)
    return None


def numbered_section_depth(section: str) -> int:
    """'3' -> 1, '2.3' -> 2, '2.3.1' -> 3."""
    return section.count(".") + 1


def markdown_level_for_section(section: str) -> int:
    """Map outline depth to markdown: top section '3.' -> ##, '2.1' -> ###."""
    return min(numbered_section_depth(section) + 1, 6)


def normalize_heading_line(line: str) -> str:
    stripped = line.strip()
    if not stripped:
        return line

    body = stripped
    md_match = _MARKDOWN_HEADING.match(stripped)
    if md_match:
        body = md_match.group(2).strip()

    num_match = _parse_numbered_heading(body)
    if num_match:
        section, title = num_match
        level = markdown_level_for_section(section)
        if "." in section:
            heading_text = f"{section} {title}"
        else:
            heading_text = f"{section}. {title}"
        return f"{'#' * level} {heading_text}"

    return stripped if md_match else line


def normalize_page_markdown(markdown: str) -> str:
    return "\n".join(normalize_heading_line(line) for line in markdown.splitlines())


def normalize_parsed_pages(pages: list[ParsedPage]) -> list[ParsedPage]:
    return [
        ParsedPage(text=normalize_page_markdown(page.text), page_number=page.page_number)
        for page in pages
    ]
