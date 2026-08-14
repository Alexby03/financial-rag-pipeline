from openai import OpenAI

from app.config import Settings
from app.models.llm import ReasonedAnswer


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
