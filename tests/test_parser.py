from app.models.enums import DocType
from app.services.parser import DocumentParser, is_refusal


class FakeOpenAI:
    """Stands in for OpenAIService so the parser can be exercised without network calls."""

    def __init__(self, page_markdowns: list[str]) -> None:
        self._page_markdowns = page_markdowns
        self.parsed_images: list[str] = []
        self.user_prompts: list[str] = []

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
        self.user_prompts.append(user_prompt)
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


def test_refused_pages_are_not_stored_as_content() -> None:
    openai = FakeOpenAI(
        [
            "# Sammanfattning",
            "I'm sorry, I can't help with that.",
            "# Framtidsutsikter",
        ]
    )
    parser = DocumentParser(openai_service=openai)

    parsed = parser.parse(b"%PDF-1.7", "agreement.pdf", DocType.AGREEMENT)

    assert [page.page_number for page in parsed.pages] == [1, 3]
    assert all("sorry" not in page.text for page in parsed.pages)


def test_a_long_page_that_merely_mentions_an_apology_is_kept() -> None:
    apology_letter = (
        "# Brev till styrelsen\n\n"
        "I'm sorry to inform the board that the acquisition timeline has slipped by one quarter. "
        "The delay stems from the pending regulatory review, and management proposes the "
        "mitigations set out in the sections below."
    )
    assert not is_refusal(apology_letter)


def test_pdf_prompts_include_page_position() -> None:
    openai = FakeOpenAI(["# Titel", "Brödtext", "# Slut"])
    parser = DocumentParser(openai_service=openai)

    parser.parse(b"%PDF-1.7", "report.pdf", DocType.FINANCIAL_REPORT)

    assert "page 1 of 3" in openai.user_prompts[0]
    assert "page 2 of 3" in openai.user_prompts[1]
    assert "Continue the document's outline" in openai.user_prompts[2]


def test_numbered_sections_are_normalized_after_parse() -> None:
    openai = FakeOpenAI(["## 2. Finansiell översikt", "### 3. Analys av finansiering"])
    parser = DocumentParser(openai_service=openai)

    parsed = parser.parse(b"%PDF-1.7", "report.pdf", DocType.FINANCIAL_REPORT)

    assert parsed.pages[1].text.startswith("## 3. Analys av finansiering")


def test_plain_text_is_a_single_page() -> None:
    openai = FakeOpenAI([])
    parser = DocumentParser(openai_service=openai)

    parsed = parser.parse(b"Anteckningar", "notes.txt", DocType.OTHER)

    assert [page.page_number for page in parsed.pages] == [1]
    assert parsed.pages[0].text == "Anteckningar"
