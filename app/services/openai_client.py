import base64

import pymupdf
from openai import OpenAI

from app.config import Settings


class OpenAIService:
    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._client = OpenAI(api_key=settings.openai_api_key)

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        response = self._client.embeddings.create(
            model=self._settings.openai_embedding_model,
            input=texts,
        )
        return [item.embedding for item in response.data]

    def embed_text(self, text: str) -> list[float]:
        return self.embed_texts([text])[0]

    def chat_completion(self, system_prompt: str, user_prompt: str) -> str:
        response = self._client.chat.completions.create(
            model=self._settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content
        return content or ""

    @staticmethod
    def _pdf_pages_to_png_base64(pdf_bytes: bytes, dpi: int = 150) -> list[str]:
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
        try:
            zoom = dpi / 72
            matrix = pymupdf.Matrix(zoom, zoom)
            images: list[str] = []
            for page in doc:
                pix = page.get_pixmap(matrix=matrix)
                images.append(base64.b64encode(pix.tobytes("png")).decode("utf-8"))
            return images
        finally:
            doc.close()

    def chat_completion_with_image(
        self,
        system_prompt: str,
        user_prompt: str,
        image_base64: str,
        mime_type: str,
    ) -> str:
        content: list[dict[str, object]] = [{"type": "text", "text": user_prompt}]

        if mime_type == "application/pdf":
            pdf_bytes = base64.b64decode(image_base64)
            for page_b64 in self._pdf_pages_to_png_base64(pdf_bytes):
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{page_b64}"},
                    }
                )
        else:
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime_type};base64,{image_base64}"},
                }
            )

        response = self._client.chat.completions.create(
            model=self._settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": content},
            ],
        )
        message_content = response.choices[0].message.content
        return message_content or ""
