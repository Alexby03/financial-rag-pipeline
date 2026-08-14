from unittest.mock import MagicMock

from app.config import Settings
from app.services.openai_client import OpenAIService, image_media_type

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)


def test_image_media_type_sniffs_png() -> None:
    assert image_media_type(TINY_PNG) == "image/png"
    assert image_media_type(b"\xff\xd8\xff") == "image/jpeg"


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
