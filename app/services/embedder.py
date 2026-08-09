from app.services.openai_client import OpenAIService


class Embedder:
    def __init__(self, openai_service: OpenAIService) -> None:
        self._openai = openai_service

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        return self._openai.embed_texts(texts)

    def embed_query(self, question: str) -> list[float]:
        return self._openai.embed_text(question)
