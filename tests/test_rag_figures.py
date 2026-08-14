from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.models.api import ChatTurn, QueryRequest
from app.models.enums import Category, DocType
from app.models.llm import ReasonedAnswer
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.services.rag import RagService

TINY_PNG = (
    b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01"
    b"\x08\x06\x00\x00\x00\x1f\x15\xc4\x89\x00\x00\x00\nIDATx\x9cc\x00\x01"
    b"\x00\x00\x05\x00\x01\r\n-\xb4\x00\x00\x00\x00IEND\xaeB`\x82"
)

FIGURE_KEY = "tenant_1/doc-1/figures/chart.png"


def build_payload(
    chunk_id: str,
    text: str,
    *,
    figure_object_key: str | None = None,
    page_number: int = 19,
) -> QdrantPayload:
    return QdrantPayload(
        raw_text=text,
        metadata=ChunkMetadata(
            tenant_id="tenant_1",
            company_name="SME Corp AB",
            doc_type=DocType.FINANCIAL_REPORT,
            category=Category.FINANCING,
            year=2024,
            quarter=3,
            source_filename="prospekt.pdf",
            object_key="tenant_1/doc-1/prospekt.pdf",
            figure_object_key=figure_object_key,
            page_number=page_number,
            chunk_id=chunk_id,
            previous_chunk_id=None,
            next_chunk_id=None,
            language="sv",
            processed_timestamp=datetime.now(UTC),
        ),
    )


class FakeEmbedder:
    def embed_query(self, question: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class FakeQdrant:
    def __init__(self, hits: list[tuple[QdrantPayload, float]]) -> None:
        self._hits = hits

    def search(self, **kwargs: Any) -> list[tuple[QdrantPayload, float]]:
        return self._hits


class FakeStorage:
    def __init__(self, objects: dict[str, bytes]) -> None:
        self.objects = objects
        self.downloaded: list[str] = []

    def download(self, object_key: str) -> bytes:
        self.downloaded.append(object_key)
        if object_key not in self.objects:
            raise FileNotFoundError(object_key)
        return self.objects[object_key]


class FakeOpenAI:
    def __init__(self) -> None:
        self.figure_calls: list[dict[str, object]] = []
        self.last_prompt = ""

    def read_figure(
        self,
        question: str,
        caption: str,
        image_bytes: bytes,
        source: str,
    ) -> str:
        self.figure_calls.append(
            {
                "question": question,
                "caption": caption,
                "image_bytes": image_bytes,
                "source": source,
            }
        )
        return "Svenska fonder hade cirka 6 800 000 MSEK i början av 2022."

    def answer_with_reasoning(self, system_prompt: str, user_prompt: str) -> ReasonedAnswer:
        self.last_prompt = user_prompt
        return ReasonedAnswer(answer="cirka 6 800 000 MSEK", reasoning_summary="vision")


def make_request() -> QueryRequest:
    return QueryRequest(
        question="Vilken fondförmögenhet hade svenska fonder i början av 2022?",
        tenant_id="tenant_1",
        top_k=5,
        rerank=False,
    )


def build_service(
    openai: FakeOpenAI,
    storage: FakeStorage | None,
    hits: list[tuple[QdrantPayload, float]],
    *,
    vision_model: str = "gpt-4o-mini",
) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        qdrant=FakeQdrant(hits),
        openai_service=openai,  # type: ignore[arg-type]
        settings=Settings(openai_api_key="test-key", openai_vision_model=vision_model),
        storage=storage,  # type: ignore[arg-type]
    )


def test_query_reads_figure_crop_and_feeds_it_to_the_reasoner() -> None:
    openai = FakeOpenAI()
    storage = FakeStorage({FIGURE_KEY: TINY_PNG})
    figure = build_payload(
        "fig-1",
        "A stacked area chart showing fund growth 2019-2024.",
        figure_object_key=FIGURE_KEY,
    )
    prose = build_payload("t-1", "Prospekttext utan siffra för 2022.", page_number=3)
    service = build_service(openai, storage, [(figure, 0.9), (prose, 0.4)])

    response = service.query(make_request())

    assert storage.downloaded == [FIGURE_KEY]
    assert len(openai.figure_calls) == 1
    assert openai.figure_calls[0]["image_bytes"] == TINY_PNG
    assert "Authoritative figure readings" in openai.last_prompt
    assert "6 800 000 MSEK" in openai.last_prompt
    assert response.trace.figure_readings[0].applied
    assert response.trace.figure_readings[0].reading.startswith("Svenska fonder")
    assert response.trace.vision_model == "gpt-4o-mini"
    assert response.answer == "cirka 6 800 000 MSEK"


def test_query_skips_vision_when_no_figure_chunks() -> None:
    openai = FakeOpenAI()
    storage = FakeStorage({FIGURE_KEY: TINY_PNG})
    prose = build_payload("t-1", "Bara löptext.")
    service = build_service(openai, storage, [(prose, 0.8)])

    service.query(make_request())

    assert storage.downloaded == []
    assert openai.figure_calls == []
    assert "Authoritative figure readings" not in openai.last_prompt


def test_query_still_answers_when_figure_download_fails() -> None:
    openai = FakeOpenAI()
    storage = FakeStorage({})
    figure = build_payload(
        "fig-1",
        "Chart caption.",
        figure_object_key=FIGURE_KEY,
    )
    service = build_service(openai, storage, [(figure, 0.9)])

    response = service.query(make_request())

    assert openai.figure_calls == []
    assert "Authoritative figure readings" not in openai.last_prompt
    assert response.trace.figure_readings[0].applied is False
    assert response.trace.figure_readings[0].error
    assert response.answer == "cirka 6 800 000 MSEK"


def test_query_includes_chat_history_in_reasoner_prompt() -> None:
    openai = FakeOpenAI()
    prose = build_payload("t-1", "Bara löptext.")
    service = build_service(openai, None, [(prose, 0.8)])
    request = make_request()
    request.chat_history = [
        ChatTurn(role="user", content="Vad var siffran 2022?"),
        ChatTurn(role="assistant", content="Cirka 6 000 000 MSEK."),
    ]
    request.question = "Kan du uttrycka det i miljarder?"

    service.query(request)

    assert "Conversation so far:" in openai.last_prompt
    assert "User: Vad var siffran 2022?" in openai.last_prompt
    assert "Assistant: Cirka 6 000 000 MSEK." in openai.last_prompt
    assert "Current question:\nKan du uttrycka det i miljarder?" in openai.last_prompt


def test_query_skips_vision_when_model_is_blank() -> None:
    openai = FakeOpenAI()
    storage = FakeStorage({FIGURE_KEY: TINY_PNG})
    figure = build_payload("fig-1", "Chart caption.", figure_object_key=FIGURE_KEY)
    service = build_service(openai, storage, [(figure, 0.9)], vision_model="")

    service.query(make_request())

    assert storage.downloaded == []
    assert openai.figure_calls == []
