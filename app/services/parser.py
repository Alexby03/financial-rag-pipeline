"""Document parsing via Mistral OCR 4 → Markdown pages + figures."""

from __future__ import annotations

import base64
import mimetypes
import re

from app.models.ocr import FigureBBoxAnnotation
from app.models.parsed import ParsedContent, ParsedFigure, ParsedPage
from app.services.heading_normalizer import normalize_page_markdown
from app.services.mistral_ocr import MistralOcrClient, OcrPage, OcrResult

# OCR embeds figures as relative markdown images that are not MinIO objects.
_IMAGE_PLACEHOLDER = re.compile(r"!\[[^\]]*\]\(img-\d+[^)]*\)", re.IGNORECASE)
_TABLE_PLACEHOLDER = re.compile(r"\[tbl-\d+[^\]]*\]\([^)]+\)", re.IGNORECASE)

_PLAIN_TEXT_EXTENSIONS = {".txt", ".md"}
_PLAIN_TEXT_MIMES = {"text/plain", "text/markdown"}


class DocumentParser:
    def __init__(self, ocr_client: MistralOcrClient) -> None:
        self._ocr = ocr_client

    def parse(self, file_bytes: bytes, filename: str) -> ParsedContent:
        mime_type, _ = mimetypes.guess_type(filename)
        mime = mime_type or "application/octet-stream"
        if mime in _PLAIN_TEXT_MIMES or filename.lower().endswith(tuple(_PLAIN_TEXT_EXTENSIONS)):
            decoded = file_bytes.decode("utf-8", errors="replace")
            return ParsedContent(
                pages=[ParsedPage(text=decoded, page_number=1)],
                content_format="markdown",
            )

        result = self._ocr.process(file_bytes, filename)
        return self._from_ocr_result(result)

    def _from_ocr_result(self, result: OcrResult) -> ParsedContent:
        zero_based = any(page.index == 0 for page in result.pages)
        pages: list[ParsedPage] = []
        figures: list[ParsedFigure] = []
        for page in result.pages:
            page_number = page.index + 1 if zero_based else page.index
            if page_number < 1:
                page_number = 1
            markdown = _page_markdown(page)
            if markdown:
                pages.append(ParsedPage(text=markdown, page_number=page_number))
            figures.extend(_figures_from_page(page, page_number))
        return ParsedContent(
            pages=pages,
            content_format="markdown",
            figures=figures,
            annotation=result.annotation,
        )


def _page_markdown(page: OcrPage) -> str:
    stripped = _IMAGE_PLACEHOLDER.sub("", page.markdown)
    stripped = _TABLE_PLACEHOLDER.sub("", stripped)
    stripped = re.sub(r"\n{3,}", "\n\n", stripped).strip()
    if not stripped:
        return ""
    return normalize_page_markdown(stripped)


def _figures_from_page(page: OcrPage, page_number: int) -> list[ParsedFigure]:
    figures: list[ParsedFigure] = []
    for image in page.images:
        image_bytes = _decode_image_base64(image.image_base64)
        if not image_bytes:
            continue
        figures.append(
            ParsedFigure(
                page_number=page_number,
                caption=_figure_caption(image.annotation, page_number),
                image_bytes=image_bytes,
            )
        )
    return figures


def _figure_caption(annotation: FigureBBoxAnnotation | None, page_number: int) -> str:
    if annotation is None:
        return f"Figure on page {page_number}"
    short = annotation.short_description.strip()
    summary = annotation.summary.strip()
    parts = [part for part in (short, summary) if part]
    if not parts:
        return f"Figure on page {page_number}"
    if len(parts) == 2 and parts[0] == parts[1]:
        return parts[0]
    return "\n\n".join(parts)


def _decode_image_base64(value: str) -> bytes | None:
    payload = value.strip()
    if not payload:
        return None
    if payload.startswith("data:") and "," in payload:
        payload = payload.split(",", 1)[1]
    try:
        decoded = base64.b64decode(payload, validate=False)
    except (ValueError, TypeError):
        return None
    return decoded or None
