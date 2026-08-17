from unittest.mock import MagicMock

from app.config import Settings
from app.services.openai_client import (
    EMBEDDING_MAX_BATCH,
    OpenAIService,
    embedding_batches,
    image_media_type,
)

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_image_media_type_sniffs_png() -> None:
    assert image_media_type(TINY_PNG) == "image/png"
    assert image_media_type(b"\xff\xd8\xff") == "image/jpeg"


def test_embedding_batches_respect_count_limit() -> None:
    texts = [f"chunk-{i}" for i in range(EMBEDDING_MAX_BATCH + 3)]
    batches = embedding_batches(texts)
    assert len(batches) == 2
    assert len(batches[0]) == EMBEDDING_MAX_BATCH
    assert len(batches[1]) == 3
    assert [text for batch in batches for text in batch] == texts


def test_embedding_batches_respect_char_budget() -> None:
    texts = ["a" * 400_000, "b" * 400_000]
    batches = embedding_batches(texts, max_batch=100, max_chars=600_000)
    assert len(batches) == 2
    assert batches[0] == [texts[0]]
    assert batches[1] == [texts[1]]


def test_embed_texts_issues_multiple_api_calls() -> None:
    service = OpenAIService(Settings(openai_api_key="test-key"))
    calls: list[list[str]] = []

    def fake_create(**kwargs: object) -> MagicMock:
        batch = list(kwargs["input"])  # type: ignore[arg-type]
        calls.append(batch)
        response = MagicMock()
        response.data = [
            MagicMock(index=i, embedding=[float(i), 0.0]) for i in range(len(batch))
        ]
        return response

    service._client.embeddings.create = fake_create  # type: ignore[method-assign]
    texts = [f"t{i}" for i in range(EMBEDDING_MAX_BATCH + 2)]
    vectors = service.embed_texts(texts)

    assert len(calls) == 2
    assert len(vectors) == len(texts)
    assert vectors[0] == [0.0, 0.0]
    assert vectors[EMBEDDING_MAX_BATCH] == [0.0, 0.0]


def test_read_figure_sends_high_detail_data_url() -> None:
    service = OpenAIService(Settings(openai_api_key="test-key", openai_vision_model="gpt-4o-mini"))
    captured: dict[str, object] = {}

    def fake_create(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        response = MagicMock()
        response.output_text = "  6 800 000 MSEK  "
        return response

    service._client.responses.create = fake_create  # type: ignore[method-assign]
    reading = service.read_figure(
        question="Fondförmögenhet 2022?",
        caption="Stacked area chart",
        image_bytes=TINY_PNG,
        source="prospekt.pdf p.19",
    )

    assert reading == "6 800 000 MSEK"
    assert captured["model"] == "gpt-4o-mini"
    payload = captured["input"]
    assert isinstance(payload, list)
    content = payload[0]["content"]
    assert content[1]["type"] == "input_image"
    assert content[1]["detail"] == "high"
    assert str(content[1]["image_url"]).startswith("data:image/png;base64,")
    assert "Fondförmögenhet" in content[0]["text"]
