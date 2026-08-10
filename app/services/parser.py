import base64
import json
import mimetypes

from app.models.enums import DocType
from app.models.parsed import ParsedContent
from app.services.chunker import strip_code_fence
from app.services.openai_client import OpenAIService

MARKDOWN_SYSTEM_PROMPT = """You extract content from Financial documents.
Return structured Markdown that preserves headings, paragraphs, and tables.
Financial tables MUST use Markdown table syntax with correct columns and rows.
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


class DocumentParser:
    def __init__(self, openai_service: OpenAIService) -> None:
        self._openai = openai_service

    def parse(self, file_bytes: bytes, filename: str, doc_type: DocType) -> ParsedContent:
        mime_type, _ = mimetypes.guess_type(filename)

        if doc_type == DocType.ORG_CHART:
            return self._parse_org_chart_image(file_bytes, mime_type or "image/png")

        markdown = self._parse_to_markdown(
            file_bytes,
            mime_type or "application/octet-stream",
            filename,
            doc_type,
        )
        return ParsedContent(text=markdown, content_format="markdown")

    def _parse_to_markdown(
        self,
        file_bytes: bytes,
        mime_type: str,
        filename: str,
        doc_type: DocType,
    ) -> str:
        if mime_type.startswith("text/") and doc_type not in FINANCIAL_DOC_TYPES:
            return file_bytes.decode("utf-8", errors="replace")

        encoded = base64.b64encode(file_bytes).decode("utf-8")
        if mime_type == "application/pdf" or filename.lower().endswith(".pdf"):
            return self._openai.chat_completion_with_image(
                MARKDOWN_SYSTEM_PROMPT,
                MARKDOWN_USER_PROMPT,
                encoded,
                "application/pdf",
            )

        if mime_type.startswith("text/"):
            text = file_bytes.decode("utf-8", errors="replace")
            return self._openai.chat_completion(
                MARKDOWN_SYSTEM_PROMPT,
                f"{MARKDOWN_USER_PROMPT}\n\nDocument:\n{text}",
            )

        return self._openai.chat_completion_with_image(
            MARKDOWN_SYSTEM_PROMPT,
            MARKDOWN_USER_PROMPT,
            encoded,
            mime_type,
        )

    def _parse_org_chart_image(self, file_bytes: bytes, mime_type: str) -> ParsedContent:
        encoded = base64.b64encode(file_bytes).decode("utf-8")
        json_text = self._openai.chat_completion_with_image(
            ORG_CHART_SYSTEM_PROMPT,
            ORG_CHART_USER_PROMPT,
            encoded,
            mime_type,
        )
        normalized = self._normalize_json(json_text)
        return ParsedContent(text=normalized, content_format="json")

    @staticmethod
    def _normalize_json(raw: str) -> str:
        parsed = json.loads(strip_code_fence(raw))
        return json.dumps(parsed, ensure_ascii=False, indent=2)
