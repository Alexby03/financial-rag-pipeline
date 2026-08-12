from app.models.enums import DocType
from app.services.parser import DocumentParser


class FakeOpenAI:
    """Stands in for OpenAIService so the parser can be exercised without network calls."""

    def __init__(self, page_markdowns: list[str]) -> None:
        self._page_markdowns = page_markdowns
        self.parsed_images: list[str] = []

    def pdf_pages_to_png_base64(self, pdf_bytes: bytes, dpi: int = 150) -> list[str]:
        return [f"page-{index}" for index in range(1, len(self._page_markdowns) + 1)]

    def chat_completion_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        mime_type: str,
    ) -> str:
        self.parsed_images.append(image_base64)
        return self._page_markdowns[len(self.parsed_images) - 1]


def test_pdf_pages_are_parsed_one_by_one() -> None:
    openai = FakeOpenAI(["# Sammanfattning", "| Nettoomsättning | 150,0 |", "# Framtidsutsikter"])
    parser = DocumentParser(openai_service=openai)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf", DocType.FINANCIAL_REPORT)

    assert parsed.content_format == "markdown"
    assert [page.page_number for page in parsed.pages] == [1, 2, 3]
    assert openai.parsed_images == ["page-1", "page-2", "page-3"]
    assert "150,0" in parsed.pages[1].text


def test_blank_pages_do_not_shift_the_numbering() -> None:
    openai = FakeOpenAI(["# Sammanfattning", "   ", "# Framtidsutsikter"])
    parser = DocumentParser(openai_service=openai)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf", DocType.FINANCIAL_REPORT)

    assert [page.page_number for page in parsed.pages] == [1, 3]


def test_plain_text_is_a_single_page() -> None:
    openai = FakeOpenAI([])
    parser = DocumentParser(openai_service=openai)

    parsed = parser.parse(b"Anteckningar", "notes.txt", DocType.OTHER)

    assert [page.page_number for page in parsed.pages] == [1]
    assert parsed.pages[0].text == "Anteckningar"
