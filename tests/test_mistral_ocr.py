from io import BytesIO
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from pypdf import PdfWriter

from app.config import Settings
from app.models.enums import Category, DocType, Language
from app.services.mistral_ocr import (
    MistralOcrClient,
    MistralOcrError,
    build_ocr_url,
    parse_document_annotation,
    parse_figure_annotation,
)

FOUNDRY_OCR = "https://project.services.ai.azure.com/providers/mistral/azure/ocr"
SERVERLESS = "https://deployment.eastus.models.ai.azure.com"


@pytest.mark.parametrize(
    ("api_base", "expected"),
    [
        (FOUNDRY_OCR, FOUNDRY_OCR),
        (f"{FOUNDRY_OCR}/", FOUNDRY_OCR),
        ("https://project.services.ai.azure.com", FOUNDRY_OCR),
        ("https://project.services.ai.azure.com/", FOUNDRY_OCR),
        (f"{SERVERLESS}/v1", f"{SERVERLESS}/v1/ocr"),
        (f"{SERVERLESS}/v1/", f"{SERVERLESS}/v1/ocr"),
        (f"{SERVERLESS}/v1/ocr", f"{SERVERLESS}/v1/ocr"),
        (SERVERLESS, f"{SERVERLESS}/v1/ocr"),
    ],
)
def test_endpoint_url_is_built_for_every_foundry_shape(api_base: str, expected: str) -> None:
    assert build_ocr_url(api_base) == expected


def build_client() -> MistralOcrClient:
    return MistralOcrClient(
        Settings(
            openai_api_key="test-key",
            mistral_ocr_endpoint=FOUNDRY_OCR,
            mistral_ocr_api_key="ocr-key",
        )
    )


def test_process_posts_base64_document_and_annotation_schemas() -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["headers"] = kwargs["headers"]
        return httpx.Response(
            200,
            json={
                "pages": [
                    {
                        "index": 1,
                        "markdown": "# Cover\n\n![img-0.jpeg](img-0.jpeg)",
                        "images": [
                            {
                                "id": "img-0.jpeg",
                                "image_base64": "data:image/png;base64,aaa",
                                "image_annotation": {
                                    "short_description": "Org chart",
                                    "summary": "CEO reports to the board.",
                                },
                            }
                        ],
                    }
                ],
                "document_annotation": {
                    "company_name": "SME Corp AB",
                    "doc_type": "financial_report",
                    "category": "financing",
                    "year": 2024,
                    "quarter": 3,
                },
            },
            request=httpx.Request("POST", url),
        )

    with patch("httpx.post", side_effect=fake_post):
        result = build_client().process(b"%PDF-1.7", "report.pdf")

    assert captured["url"] == FOUNDRY_OCR
    assert captured["headers"]["Authorization"] == "Bearer ocr-key"
    assert captured["json"]["model"] == "mistral-ocr-4-0"
    assert captured["json"]["include_image_base64"] is True
    assert captured["json"]["document"]["type"] == "document_url"
    assert captured["json"]["document"]["document_url"].startswith("data:application/pdf;base64,")
    assert captured["json"]["bbox_annotation_format"]["type"] == "json_schema"
    assert captured["json"]["document_annotation_format"]["type"] == "json_schema"
    assert result.pages[0].markdown.startswith("# Cover")
    assert result.pages[0].images[0].annotation is not None
    assert result.pages[0].images[0].annotation.short_description == "Org chart"
    assert result.annotation is not None
    assert result.annotation.company_name == "SME Corp AB"
    assert result.annotation.doc_type == DocType.FINANCIAL_REPORT
    assert result.annotation.quarter == 3


def test_images_use_image_url_document_type() -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["json"] = kwargs["json"]
        return httpx.Response(
            200,
            json={"pages": [{"index": 1, "markdown": "chart", "images": []}]},
            request=httpx.Request("POST", url),
        )

    with patch("httpx.post", side_effect=fake_post):
        build_client().process(b"\x89PNG", "chart.png")

    assert captured["json"]["document"]["type"] == "image_url"
    assert captured["json"]["document"]["image_url"].startswith("data:image/png;base64,")


def test_http_error_becomes_ocr_error() -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(404, text="DeploymentNotFound", request=request)

    with patch("httpx.post", side_effect=fake_post), pytest.raises(MistralOcrError) as error:
        build_client().process(b"%PDF", "report.pdf")

    assert "404" in str(error.value)


def test_missing_config_raises_before_http() -> None:
    client = MistralOcrClient(
        Settings(
            openai_api_key="test-key",
            mistral_ocr_endpoint="",
            mistral_ocr_api_key="",
        )
    )
    with pytest.raises(MistralOcrError, match="not configured"):
        client.process(b"%PDF", "report.pdf")


def test_document_annotation_accepts_json_string_and_q_prefix() -> None:
    annotation = parse_document_annotation(
        '{"company_name": "SME Corp AB", "doc_type": "minutes", '
        '"category": "legal", "year": "2024", "quarter": "Q2", "language": "sv"}'
    )
    assert annotation is not None
    assert annotation.doc_type == DocType.MINUTES
    assert annotation.category == Category.LEGAL
    assert annotation.year == 2024
    assert annotation.quarter == 2
    assert annotation.language == Language.SV


def test_document_annotation_normalizes_language_aliases() -> None:
    swedish = parse_document_annotation({"language": "Swedish"})
    danish = parse_document_annotation({"language": "da"})
    region = parse_document_annotation({"language": "en-US"})
    assert swedish is not None and swedish.language == Language.SV
    assert danish is not None and danish.language == Language.DK
    assert region is not None and region.language == Language.EN


def test_unknown_enum_values_are_dropped() -> None:
    annotation = parse_document_annotation(
        {
            "company_name": "SME Corp AB",
            "doc_type": "annual_report",
            "year": 2024,
            "language": "klingon",
        }
    )
    assert annotation is not None
    assert annotation.doc_type is None
    assert annotation.language is None
    assert annotation.company_name == "SME Corp AB"
    assert annotation.year == 2024


def test_figure_annotation_parses_json_string() -> None:
    annotation = parse_figure_annotation(
        '{"short_description": "Scatter plot", "summary": "Cost vs accuracy."}'
    )
    assert annotation is not None
    assert annotation.short_description == "Scatter plot"


def _ocr_page_payload(index: int, markdown: str) -> dict[str, Any]:
    return {"index": index, "markdown": markdown, "images": []}


def test_long_pdf_is_split_into_max_page_slices() -> None:
    writer = PdfWriter()
    for _ in range(5):
        writer.add_blank_page(width=72, height=72)
    pdf = BytesIO()
    writer.write(pdf)
    pdf_bytes = pdf.getvalue()

    calls: list[dict[str, Any]] = []

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        call_number = len(calls)
        calls.append(kwargs["json"])
        pages = [
            _ocr_page_payload(1, f"slice-{call_number}-a"),
            _ocr_page_payload(2, f"slice-{call_number}-b"),
        ]
        if call_number == 2:
            pages = [_ocr_page_payload(1, "slice-2-a")]
        return httpx.Response(
            200,
            json={"pages": pages, "document_annotation": {"company_name": "SME Corp AB"}},
            request=httpx.Request("POST", url),
        )

    client = MistralOcrClient(
        Settings(
            openai_api_key="test-key",
            mistral_ocr_endpoint=FOUNDRY_OCR,
            mistral_ocr_api_key="ocr-key",
            mistral_ocr_max_pages=2,
        )
    )
    with patch("httpx.post", side_effect=fake_post):
        result = client.process(pdf_bytes, "report.pdf")

    assert len(calls) == 3
    assert "document_annotation_format" in calls[0]
    assert "document_annotation_format" not in calls[1]
    assert "document_annotation_format" not in calls[2]
    assert [page.index for page in result.pages] == [1, 2, 3, 4, 5]
    assert result.pages[0].markdown == "slice-0-a"
    assert result.pages[2].markdown == "slice-1-a"
    assert result.pages[4].markdown == "slice-2-a"
    assert result.annotation is not None
    assert result.annotation.company_name == "SME Corp AB"
