import base64
import json
import mimetypes

from app.models.enums import DocType
from app.models.parsed import ParsedContent, ParsedPage
from app.services.chunker import strip_code_fence
from app.services.openai_client import OpenAIService

MARKDOWN_SYSTEM_PROMPT = """You extract content from Financial documents.
Return structured Markdown that preserves headings, paragraphs, and tables.
Financial tables MUST use Markdown table syntax with correct columns and rows.
Signature pages are ordinary content: transcribe the printed text around them and write
[signatur] where a handwritten signature appears, without naming or identifying the signatory.
Do not summarize — extract faithfully."""

MARKDOWN_USER_PROMPT = """Extract all text and tables from this document.
Output Markdown only."""

ORG_CHART_SYSTEM_PROMPT = """You describe organizational charts for retrieval.
Return valid JSON with keys: title, company, nodes (list of {name, role, reports_to}), summary.
Be precise about hierarchy and reporting lines.
Output JSON only, no markdown fences."""

ORG_CHART_USER_PROMPT = (
    "Describe this organizational chart as JSON for semantic search indexing."
)

FINANCIAL_DOC_TYPES = {DocType.FINANCIAL_REPORT, DocType.AGREEMENT, DocType.MINUTES}

REFUSAL_MARKERS = (
    "i'm sorry",
    "i am sorry",
    "i can't help",
    "i cannot help",
    "i can't assist",
    "i cannot assist",
    "i'm unable to",
    "i am unable to",
)
REFUSAL_MAX_LENGTH = 200


def is_refusal(markdown: str) -> bool:
    """Vision models decline signature pages. A refusal is short and apologetic; a page is not.

    Storing one would put "I'm sorry, I can't help with that." in the index as if it were
    content, where it is indistinguishable from a real chunk to both the embedder and the reader.
    """
    text = markdown.strip().lower()
    if len(text) > REFUSAL_MAX_LENGTH:
        return False
    return any(marker in text for marker in REFUSAL_MARKERS)


class DocumentParser:
    def __init__(self, openai_service: OpenAIService) -> None:
        self._openai = openai_service

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedContent:
        mime_type, _ = mimetypes.guess_type(filename)

        if doc_type == DocType.ORG_CHART:
            return self._parse_org_chart_image(file_bytes, mime_type or "image/png")

        pages = self._parse_to_markdown_pages(
            file_bytes,
            mime_type or "application/octet-stream",
            filename,
            doc_type,
        )
        return ParsedContent(pages=pages, content_format="markdown")

    def _parse_to_markdown_pages(
        self,
        file_bytes: bytes,
        mime_type: str,
        filename: str,
        doc_type: DocType,
    ) -> list[ParsedPage]:
        if mime_type.startswith("text/") and doc_type not in FINANCIAL_DOC_TYPES:
            decoded = file_bytes.decode("utf-8", errors="replace")
            return [ParsedPage(text=decoded, page_number=1)]

        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            return self._parse_pdf_pages(file_bytes)

        if mime_type.startswith("text/"):
            text = file_bytes.decode("utf-8", errors="replace")
            markdown = self._openai.chat_completion(
                MARKDOWN_SYSTEM_PROMPT,
                f"{MARKDOWN_USER_PROMPT}\n\nDocument:\n{text}",
            )
            return [ParsedPage(text=markdown, page_number=1)]

        encoded = base64.b64encode(file_bytes).decode("utf-8")
        markdown = self._openai.chat_completion_with_image(
            MARKDOWN_SYSTEM_PROMPT,
            MARKDOWN_USER_PROMPT,
            encoded,
            mime_type,
        )
        if is_refusal(markdown):
            return []
        return [ParsedPage(text=markdown, page_number=1)]

    def _parse_pdf_pages(self, file_bytes: bytes) -> list[ParsedPage]:
        """One request per page: the page number is only knowable while the pages are separate."""
        pages: list[ParsedPage] = []
        for index, page_png in enumerate(self._openai.pdf_pages_to_png_base64(file_bytes), start=1):
            markdown = self._openai.chat_completion_with_image(
                MARKDOWN_SYSTEM_PROMPT,
                MARKDOWN_USER_PROMPT,
                page_png,
                "image/png",
            )
            if markdown.strip() and not is_refusal(markdown):
                pages.append(ParsedPage(text=markdown, page_number=index))
        return pages

    def _parse_org_chart_image(self, file_bytes: bytes, mime_type: str) -> ParsedContent:
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        json_text = self._openai.chat_completion_with_image(
            ORG_CHART_SYSTEM_PROMPT,
            ORG_CHART_USER_PROMPT,
            encoded,
            mime_type,
        )
        normalized = self._normalize_json(json_text)
        return ParsedContent(
            pages=[ParsedPage(text=normalized, page_number=1)],
            content_format="json",
        )

    @staticmethod
    def _normalize_json(raw: str) -> str:
        parsed = json.loads(strip_code_fence(raw))
        return json.dumps(parsed, ensure_ascii=False, indent=2)
