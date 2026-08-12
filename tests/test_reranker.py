from typing import Any
from unittest.mock import patch

import httpx

from app.config import Settings
from app.services.reranker import CohereReranker

DOCUMENTS = ["chunk a", "chunk b", "chunk c"]


def build_reranker() -> CohereReranker:
    settings = Settings(
        openai_api_key="test-key",
        cohere_api_base="https://endpoint.models.ai.azure.com/v1",
        cohere_api_key="azure-key",
    )
    return CohereReranker(settings)


def test_rerank_posts_to_versioned_endpoint_and_returns_order() -> None:
    captured: dict[str, Any] = {}

    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        captured["url"] = url
        captured["json"] = kwargs["json"]
        captured["headers"] = kwargs["headers"]
        return httpx.Response(
            200,
            json={
                "results": [
                    {"index": 2, "relevance_score": 0.91},
                    {"index": 0, "relevance_score": 0.42},
                ]
            },
            request=httpx.Request("POST", url),
        )

    with patch("httpx.post", side_effect=fake_post):
        outcome = build_reranker().rerank("fraga", DOCUMENTS, top_n=2)

    assert captured["url"] == "https://endpoint.models.ai.azure.com/v1/rerank"
    assert captured["headers"]["Authorization"] == "Bearer azure-key"
    assert captured["json"]["top_n"] == 2
    assert outcome.applied
    assert [document.index for document in outcome.documents] == [2, 0]


def test_missing_access_falls_back_to_vector_order() -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(401, json={"error": "unauthorized"}, request=request)

    with patch("httpx.post", side_effect=fake_post):
        outcome = build_reranker().rerank("fraga", DOCUMENTS, top_n=2)

    assert not outcome.applied
    assert "401" in outcome.fallback_reason
    assert outcome.documents == []


def test_network_failure_falls_back_to_vector_order() -> None:
    with patch("httpx.post", side_effect=httpx.ConnectError("no route")):
        outcome = build_reranker().rerank("fraga", DOCUMENTS, top_n=2)

    assert not outcome.applied
    assert "Rerank request failed" in outcome.fallback_reason


def test_unexpected_payload_falls_back_to_vector_order() -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(200, json={"unexpected": []}, request=request)

    with patch("httpx.post", side_effect=fake_post):
        outcome = build_reranker().rerank("fraga", DOCUMENTS, top_n=2)

    assert not outcome.applied
    assert "Unexpected rerank response shape" in outcome.fallback_reason


def test_out_of_range_index_is_rejected() -> None:
    def fake_post(url: str, **kwargs: Any) -> httpx.Response:
        request = httpx.Request("POST", url)
        return httpx.Response(
            200,
            json={"results": [{"index": 99, "relevance_score": 0.5}]},
            request=request,
        )

    with patch("httpx.post", side_effect=fake_post):
        outcome = build_reranker().rerank("fraga", DOCUMENTS, top_n=2)

    assert not outcome.applied
    assert "unknown documents" in outcome.fallback_reason
