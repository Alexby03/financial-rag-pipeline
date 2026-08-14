import base64

from app.models.enums import Category, DocType
from app.models.ocr import FigureBBoxAnnotation, OcrDocumentAnnotation
from app.models.parsed import ParsedFigure, ParsedPage
from app.services.mistral_ocr import OcrPage, OcrPageImage, OcrResult
from app.services.parser import DocumentParser

TINY_PNG_B64 = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
TINY_PNG = base64.b64decode(TINY_PNG_B64)


class FakeOcrClient:
    """Stands in for MistralOcrClient so the parser can be exercised without network calls."""

    def __init__(self, result: OcrResult) -> None:
        self._result = result
        self.calls: list[tuple[bytes, str]] = []

    def process(self, file_bytes: bytes, filename: str) -> OcrResult:
        self.calls.append((file_bytes, filename))
        return self._result


def _ocr_page(
    index: int,
    markdown: str,
    images: list[OcrPageImage] | None = None,
) -> OcrPage:
    return OcrPage(index=index, markdown=markdown, images=images or [])


def test_ocr_pages_keep_their_numbers() -> None:
    client = FakeOcrClient(
        OcrResult(
            pages=[
                _ocr_page(1, "# Sammanfattning"),
                _ocr_page(2, "| Nettoomsättning | 150,0 |"),
                _ocr_page(3, "# Framtidsutsikter"),
            ]
        )
    )
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert parsed.content_format == "markdown"
    assert [page.page_number for page in parsed.pages] == [1, 2, 3]
    assert "150,0" in parsed.pages[1].text
    assert client.calls == [(b"%PDF-1.7", "report.pdf")]


def test_zero_based_page_indexes_become_one_based() -> None:
    client = FakeOcrClient(
        OcrResult(pages=[_ocr_page(0, "# Cover"), _ocr_page(1, "Body")])
    )
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert [page.page_number for page in parsed.pages] == [1, 2]


def test_blank_pages_do_not_shift_the_numbering() -> None:
    client = FakeOcrClient(
        OcrResult(
            pages=[
                _ocr_page(1, "# Sammanfattning"),
                _ocr_page(2, "   "),
                _ocr_page(3, "# Framtidsutsikter"),
            ]
        )
    )
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert [page.page_number for page in parsed.pages] == [1, 3]


def test_image_placeholders_are_stripped_from_page_markdown() -> None:
    client = FakeOcrClient(
        OcrResult(
            pages=[
                _ocr_page(
                    1,
                    "# Resultat\n\n![img-0.jpeg](img-0.jpeg)\n\nSe grafen ovan.",
                    images=[
                        OcrPageImage(
                            image_id="img-0.jpeg",
                            image_base64=TINY_PNG_B64,
                            annotation=FigureBBoxAnnotation(
                                short_description="Revenue chart",
                                summary="Bars for Q1-Q3 revenue in MSEK.",
                            ),
                        )
                    ],
                )
            ]
        )
    )
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert parsed.pages[0].text == "# Resultat\n\nSe grafen ovan."
    assert "img-0" not in parsed.pages[0].text
    assert parsed.figures == [
        ParsedFigure(
            page_number=1,
            caption="Revenue chart\n\nBars for Q1-Q3 revenue in MSEK.",
            image_bytes=TINY_PNG,
        )
    ]


def test_figures_without_bytes_are_skipped() -> None:
    client = FakeOcrClient(
        OcrResult(
            pages=[
                _ocr_page(
                    1,
                    "# Cover",
                    images=[OcrPageImage(image_id="img-0.jpeg", image_base64="")],
                )
            ]
        )
    )
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert parsed.figures == []
    assert parsed.pages == [ParsedPage(text="# Cover", page_number=1)]


def test_numbered_sections_are_normalized_after_parse() -> None:
    client = FakeOcrClient(
        OcrResult(
            pages=[
                _ocr_page(1, "## 2. Finansiell översikt"),
                _ocr_page(2, "### 3. Analys av finansiering"),
            ]
        )
    )
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert parsed.pages[1].text.startswith("## 3. Analys av finansiering")


def test_plain_text_is_a_single_page_without_ocr() -> None:
    client = FakeOcrClient(OcrResult(pages=[]))
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"Anteckningar", "notes.txt")

    assert client.calls == []
    assert [page.page_number for page in parsed.pages] == [1]
    assert parsed.pages[0].text == "Anteckningar"
    assert parsed.figures == []


def test_document_annotation_is_passed_through() -> None:
    annotation = OcrDocumentAnnotation(
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        quarter=3,
    )
    client = FakeOcrClient(OcrResult(pages=[_ocr_page(1, "# Report")], annotation=annotation))
    parser = DocumentParser(ocr_client=client)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf")

    assert parsed.annotation == annotation
