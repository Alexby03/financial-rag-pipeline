import base64
from typing import Any

from openai import OpenAI

from app.config import Settings
from app.models.llm import ReasonedAnswer

FIGURE_VISION_PROMPT = """You extract quantitative answers from charts and figures in Financial documents.

Rules for charts:
- Read axis titles and units first (MSEK, mdkr, SEK, %, etc.).
- Locate the asked year or period on the x-axis. "Beginning of 2022" / "början av 2022" means
  the 2022 tick or the left edge of that year — not mid-year and not 2023.
- For stacked area or stacked bar charts, the TOTAL is the top of the full stack at that
  x position, not a single series and not the chart's y-axis maximum.
- Use labeled tick marks. Interpolate between ticks and say the value is approximate when you do.
- Do not invent a round number that conflicts with nearby ticks.
- Briefly cite the ticks you used (e.g. "y-axis between 5 000 000 and 6 000 000 at the 2022 mark").
- If the crop is too low-resolution to read reliably, say so instead of guessing.

Reply in the same language as the question. Lead with the answer value, then the brief method."""

# OpenAI embeddings.create caps: 2048 inputs and ~300k tokens per request.
EMBEDDING_MAX_BATCH = 512
# ~4 chars/token → stay well under 300k tokens with margin for non-English text.
EMBEDDING_MAX_CHARS = 600_000


def embedding_batches(
    texts: list[str],
    *,
    max_batch: int = EMBEDDING_MAX_BATCH,
    max_chars: int = EMBEDDING_MAX_CHARS,
) -> list[list[str]]:
    """Split texts so each OpenAI embeddings request stays under API limits."""
    if not texts:
        return []
    batches: list[list[str]] = []
    current: list[str] = []
    chars = 0
    for text in texts:
        text_chars = len(text)
        if current and (
            len(current) >= max_batch or chars + text_chars > max_chars
        ):
            batches.append(current)
            current = []
            chars = 0
        current.append(text)
        chars += text_chars
    if current:
        batches.append(current)
    return batches


def image_media_type(image_bytes: bytes) -> str:
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes.startswith((b"GIF87a", b"GIF89a")):
        return "image/gif"
    if image_bytes.startswith(b"RIFF") and b"WEBP" in image_bytes[:16]:
        return "image/webp"
    return "image/png"


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        vectors: list[list[float]] = []
        for batch in embedding_batches(texts):
            response = self._client.embeddings.create(
                model=self._settings.openai_embedding_model,
                input=batch,
            )
            ordered = sorted(response.data, key=lambda item: item.index)
            vectors.extend(item.embedding for item in ordered)
        return vectors

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def read_figure(
        self,
        question: str,
        caption: str,
        image_bytes: bytes,
        source: str,
    ) -> str:
        """Ask the vision model to read a stored crop against the user's question."""
        encoded = base64.b64encode(image_bytes).decode("ascii")
        media_type = image_media_type(image_bytes)
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    f"Question:\n{question}\n\nSource: {source}\n\n"
                    f"OCR caption (may be incomplete):\n{caption}"
                ),
            },
            {
                "type": "input_image",
                "detail": "high",
                "image_url": f"data:{media_type};base64,{encoded}",
            },
        ]
        response = self._client.responses.create(
            model=self._settings.openai_vision_model,
            instructions=FIGURE_VISION_PROMPT,
            input=[{"role": "user", "content": content}],
        )
        return response.output_text.strip()

    def answer_with_reasoning(self, system_prompt: str, user_prompt: str) -> ReasonedAnswer:
        response = self._client.responses.create(
            model=self._settings.openai_answer_model,
            instructions=system_prompt,
            input=user_prompt,
            reasoning={
                "effort": self._settings.openai_reasoning_effort,
                "summary": self._settings.openai_reasoning_summary,
            },
        )
        summaries = [
            part.text
            for item in response.output
            if item.type == "reasoning"
            for part in item.summary
        ]
        return ReasonedAnswer(
            answer=response.output_text,
            reasoning_summary="\n\n".join(summaries),
        )
