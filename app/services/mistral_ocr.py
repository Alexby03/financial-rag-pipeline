"""Mistral OCR 4 client for Azure AI Foundry.

Foundry hands out several Target URI shapes, and the official SDK appends `/v1/ocr` in ways
that 404. This uses httpx against the normalized `/ocr` URL, the same pattern as Cohere rerank.
"""

from __future__ import annotations

import base64
import json
import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from io import BytesIO
from typing import Any, TypeVar

import httpx
from pydantic import BaseModel, ValidationError
from pypdf import PdfReader, PdfWriter

from app.config import Settings
from app.models.enums import Category, DocType, Language
from app.models.ocr import FigureBBoxAnnotation, OcrDocumentAnnotation

DOCUMENT_ANNOTATION_PROMPT = (
    "Extract metadata for an Financial document archive. "
    "company_name is the company the document is about. "
    "doc_type, category, and language must be one of the allowed enum values. "
    "language is the primary language of the document body "
    "(sv, en, dk, no, fi, de, fr, or other). "
    "year is the fiscal or calendar year. quarter is 1-4 only for quarterly documents."
)

EnumT = TypeVar("EnumT", bound=Enum)


class MistralOcrError(RuntimeError):
    """Raised when the OCR endpoint is missing, times out, or returns an error."""


def build_ocr_url(api_base: str) -> str:
    """Foundry hands out several base URL shapes depending on the deployment type."""
    base = api_base.rstrip("/")
    if base.endswith("/ocr"):
        return base
    if base.endswith("/v1"):
        return f"{base}/ocr"
    # Resource hosts look like https://<name>.services.ai.azure.com — same family as Cohere.
    if "services.ai.azure.com" in base and "/providers/" not in base:
        return f"{base}/providers/mistral/azure/ocr"
    return f"{base}/v1/ocr"


def json_schema_response_format(model: type[BaseModel], name: str) -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "schema": model.model_json_schema(),
        },
    }


@dataclass(frozen=True)
class OcrPageImage:
    image_id: str
    image_base64: str
    annotation: FigureBBoxAnnotation | None = None


@dataclass(frozen=True)
class OcrPage:
    index: int
    markdown: str
    images: list[OcrPageImage] = field(default_factory=list)


@dataclass(frozen=True)
class OcrResult:
    pages: list[OcrPage]
    annotation: OcrDocumentAnnotation | None = None


class MistralOcrClient:
    def __init__(self, settings: Settings) -> None:
        self._url = (
            build_ocr_url(settings.mistral_ocr_endpoint) if settings.mistral_ocr_endpoint else ""
        )
        self._api_key = settings.mistral_ocr_api_key
        self._model = settings.mistral_ocr_model
        self._timeout = settings.mistral_ocr_timeout_seconds
        self._max_pages = settings.mistral_ocr_max_pages

    def process(self, file_bytes: bytes, filename: str) -> OcrResult:
        if not self._url or not self._api_key:
            raise MistralOcrError(
                "Mistral OCR is not configured (MISTRAL_OCR_ENDPOINT / MISTRAL_OCR_API_KEY)"
            )

        mime_type, _ = mimetypes.guess_type(filename)
        mime = mime_type or "application/octet-stream"
        if _is_pdf(filename, mime):
            return self._process_pdf(file_bytes, filename)
        return self._ocr_once(file_bytes, filename, mime, annotate_document=True)

    def _process_pdf(self, file_bytes: bytes, filename: str) -> OcrResult:
        try:
            page_count = _pdf_page_count(file_bytes)
        except Exception:
            page_count = 0
        if page_count <= self._max_pages:
            return self._ocr_once(file_bytes, filename, "application/pdf", annotate_document=True)

        pages: list[OcrPage] = []
        annotation = None
        offset = 0
        while offset < page_count:
            end = min(offset + self._max_pages, page_count)
            slice_bytes = _pdf_slice(file_bytes, offset, end)
            result = self._ocr_once(
                slice_bytes,
                filename,
                "application/pdf",
                annotate_document=offset == 0,
            )
            pages.extend(_rebase_pages(result.pages, offset))
            if annotation is None:
                annotation = result.annotation
            offset = end
        return OcrResult(pages=pages, annotation=annotation)

    def _ocr_once(
        self,
        file_bytes: bytes,
        filename: str,
        mime: str,
        *,
        annotate_document: bool,
    ) -> OcrResult:
        encoded = base64.b64encode(file_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{encoded}"
        if mime.startswith("image/"):
            document: dict[str, str] = {"type": "image_url", "image_url": data_url}
        else:
            document = {
                "type": "document_url",
                "document_url": data_url,
                "document_name": filename,
            }
        body: dict[str, Any] = {
            "model": self._model,
            "document": document,
            "include_image_base64": True,
            "bbox_annotation_format": json_schema_response_format(
                FigureBBoxAnnotation, "figure_annotation"
            ),
        }
        if annotate_document:
            body["document_annotation_format"] = json_schema_response_format(
                OcrDocumentAnnotation, "document_annotation"
            )
            body["document_annotation_prompt"] = DOCUMENT_ANNOTATION_PROMPT

        try:
            response = httpx.post(
                self._url,
                headers={
                    "Authorization": f"Bearer {self._api_key}",
                    "Content-Type": "application/json",
                },
                json=body,
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            body_text = error.response.text[:500]
            raise MistralOcrError(
                f"OCR endpoint returned {error.response.status_code}: {body_text}"
            ) from error
        except httpx.HTTPError as error:
            raise MistralOcrError(f"OCR request failed: {error}") from error
        except ValueError as error:
            raise MistralOcrError(f"OCR response was not JSON: {error}") from error

        if not isinstance(payload, dict):
            raise MistralOcrError("OCR response was not a JSON object")
        return OcrResult(
            pages=_parse_pages(payload.get("pages") or []),
            annotation=parse_document_annotation(payload.get("document_annotation")),
        )


def parse_document_annotation(raw: object) -> OcrDocumentAnnotation | None:
    data = _as_json_object(raw)
    if data is None:
        return None
    parsed: dict[str, object] = {}
    company = data.get("company_name")
    if isinstance(company, str) and company.strip():
        parsed["company_name"] = company.strip()
    doc_type = _coerce_enum(data.get("doc_type"), DocType)
    if doc_type is not None:
        parsed["doc_type"] = doc_type
    category = _coerce_enum(data.get("category"), Category)
    if category is not None:
        parsed["category"] = category
    year = _coerce_int(data.get("year"))
    if year is not None:
        parsed["year"] = year
    quarter = _coerce_quarter(data.get("quarter"))
    if quarter is not None:
        parsed["quarter"] = quarter
    language = _coerce_language(data.get("language"))
    if language is not None:
        parsed["language"] = language
    if not parsed:
        return None
    try:
        return OcrDocumentAnnotation.model_validate(parsed)
    except ValidationError:
        return None


def parse_figure_annotation(raw: object) -> FigureBBoxAnnotation | None:
    data = _as_json_object(raw)
    if data is None:
        return None
    try:
        annotation = FigureBBoxAnnotation.model_validate(data)
    except ValidationError:
        return None
    if not annotation.short_description.strip() and not annotation.summary.strip():
        return None
    return annotation


def _parse_pages(raw_pages: object) -> list[OcrPage]:
    if not isinstance(raw_pages, list):
        return []
    pages: list[OcrPage] = []
    for item in raw_pages:
        if not isinstance(item, dict):
            continue
        index = _coerce_int(item.get("index"))
        if index is None:
            continue
        markdown = item.get("markdown")
        pages.append(
            OcrPage(
                index=index,
                markdown=str(markdown) if markdown is not None else "",
                images=_parse_images(item.get("images") or []),
            )
        )
    return pages


def _parse_images(raw_images: object) -> list[OcrPageImage]:
    if not isinstance(raw_images, list):
        return []
    images: list[OcrPageImage] = []
    for item in raw_images:
        if not isinstance(item, dict):
            continue
        encoded = item.get("image_base64")
        if not isinstance(encoded, str) or not encoded.strip():
            continue
        image_id = str(item.get("id") or item.get("image_id") or "")
        images.append(
            OcrPageImage(
                image_id=image_id,
                image_base64=encoded,
                annotation=parse_figure_annotation(item.get("image_annotation")),
            )
        )
    return images


def _as_json_object(raw: object) -> dict[str, Any] | None:
    if raw is None:
        return None
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str):
        stripped = raw.strip()
        if not stripped:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return None
        return parsed if isinstance(parsed, dict) else None
    return None


def _coerce_enum(value: object, enum_cls: type[EnumT]) -> EnumT | None:
    if value is None:
        return None
    try:
        return enum_cls(str(value).strip())
    except ValueError:
        return None


# OCR sometimes returns names or ISO codes we map onto Language.
_LANGUAGE_ALIASES: dict[str, str] = {
    "swedish": "sv",
    "svenska": "sv",
    "english": "en",
    "danish": "dk",
    "dansk": "dk",
    "da": "dk",
    "norwegian": "no",
    "norsk": "no",
    "nb": "no",
    "nn": "no",
    "finnish": "fi",
    "suomi": "fi",
    "german": "de",
    "deutsch": "de",
    "french": "fr",
    "francais": "fr",
    "français": "fr",
}


def _coerce_language(value: object) -> Language | None:
    if value is None:
        return None
    raw = str(value).strip().lower().replace("_", "-")
    if not raw:
        return None
    # Region tags like sv-SE / en_US → primary subtag.
    primary = raw.split("-", 1)[0]
    code = _LANGUAGE_ALIASES.get(raw) or _LANGUAGE_ALIASES.get(primary) or primary
    return _coerce_enum(code, Language)


def _coerce_int(value: object) -> int | None:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    if isinstance(value, str) and value.strip().lstrip("-").isdigit():
        return int(value.strip())
    return None


def _coerce_quarter(value: object) -> int | None:
    if value is None:
        return None
    parsed = _coerce_int(value)
    if parsed is not None:
        return parsed if 1 <= parsed <= 4 else None
    text = str(value).strip().upper().replace("QUARTER", "").replace("Q", "").strip()
    if text.isdigit() and 1 <= int(text) <= 4:
        return int(text)
    return None


def _is_pdf(filename: str, mime: str) -> bool:
    return mime == "application/pdf" or filename.lower().endswith(".pdf")


def _pdf_page_count(pdf_bytes: bytes) -> int:
    return len(PdfReader(BytesIO(pdf_bytes)).pages)


def _pdf_slice(pdf_bytes: bytes, start: int, end: int) -> bytes:
    reader = PdfReader(BytesIO(pdf_bytes))
    writer = PdfWriter()
    for index in range(start, min(end, len(reader.pages))):
        writer.add_page(reader.pages[index])
    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


def _rebase_pages(pages: list[OcrPage], page_offset: int) -> list[OcrPage]:
    """Map a slice's local page indexes onto the original 1-based document."""
    if not pages:
        return []
    zero_based = any(page.index == 0 for page in pages)
    rebased: list[OcrPage] = []
    for page in pages:
        local = page.index + 1 if zero_based else page.index
        if local < 1:
            local = 1
        rebased.append(
            OcrPage(
                index=page_offset + local,
                markdown=page.markdown,
                images=page.images,
            )
        )
    return rebased
