"""Cohere Rerank on Azure AI Foundry.

The endpoint is a plain HTTP API, so this uses httpx directly instead of the Cohere SDK:
the SDK appends its own route to the base URL, which conflicts with the versioned base URL
that Foundry hands out. Any failure degrades to the original vector order so a missing or
revoked deployment never takes the query path down.
"""

import httpx

from app.config import Settings
from app.models.rerank import RerankedDocument, RerankOutcome


def build_rerank_url(api_base: str) -> str:
    """Foundry hands out several base URL shapes depending on the deployment type."""
    base = api_base.rstrip("/")
    if base.endswith("/rerank"):
        return base
    if base.endswith("/providers/cohere"):
        return f"{base}/v2/rerank"
    return f"{base}/rerank"


class CohereReranker:
    def __init__(self, settings: Settings) -> None:
        self._url = build_rerank_url(settings.cohere_api_base)
        self._api_key = settings.cohere_api_key
        self._model = settings.cohere_rerank_model
        self._timeout = settings.rerank_timeout_seconds

    @property
    def model(self) -> str:
        return self._model

    def rerank(self, query: str, documents: list[str], top_n: int) -> RerankOutcome:
        if not documents:
            return RerankOutcome(applied=True, documents=[])

        try:
            response = httpx.post(
                self._url,
                headers={"Authorization": f"Bearer {self._api_key}"},
                json={
                    "model": self._model,
                    "query": query,
                    "documents": documents,
                    "top_n": min(top_n, len(documents)),
                },
                timeout=self._timeout,
            )
            response.raise_for_status()
            payload = response.json()
        except httpx.HTTPStatusError as error:
            return RerankOutcome(
                applied=False,
                fallback_reason=f"Rerank endpoint returned {error.response.status_code}",
            )
        except httpx.HTTPError as error:
            return RerankOutcome(applied=False, fallback_reason=f"Rerank request failed: {error}")

        try:
            ranked = [
                RerankedDocument(index=item["index"], relevance_score=item["relevance_score"])
                for item in payload["results"]
            ]
        except (KeyError, TypeError) as error:
            return RerankOutcome(
                applied=False,
                fallback_reason=f"Unexpected rerank response shape: {error}",
            )

        if any(document.index >= len(documents) for document in ranked):
            return RerankOutcome(
                applied=False,
                fallback_reason="Rerank response referenced unknown documents",
            )
        return RerankOutcome(applied=True, documents=ranked)
