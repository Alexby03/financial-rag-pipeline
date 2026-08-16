"""Entity scoping keeps wrong-company / wrong-period chunks out of answers."""

from datetime import UTC, datetime

from app.config import Settings
from app.models.api import QueryRequest
from app.models.enums import Category, DocType
from app.models.llm import ReasonedAnswer
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.services.entity_scope import (
    company_mentioned_in_query,
    entity_rank_indices,
    parse_period_from_query,
)
from app.services.rag import RagService


def _payload(chunk_id: str, company: str, text: str, *, year: int = 2010) -> QdrantPayload:
    return QdrantPayload(
        raw_text=text,
        metadata=ChunkMetadata(
            tenant_id="tenant_1",
            company_name=company,
            doc_type=DocType.ACCOUNTING_DATA,
            category=Category.FINANCING,
            year=year,
            quarter=None,
            source_filename=f"{company}.si",
            object_key=f"tenant_1/{chunk_id}/{company}.si",
            page_number=1,
            chunk_id=chunk_id,
            previous_chunk_id=None,
            next_chunk_id=None,
            language="sv",
            processed_timestamp=datetime.now(UTC),
        ),
    )


class _Embedder:
    def embed_query(self, question: str) -> list[float]:
        return [0.1, 0.2, 0.3]


class _Qdrant:
    def __init__(self, hits: list[tuple[QdrantPayload, float]]) -> None:
        self._hits = hits
        self.last_company: str | None = None

    def search(self, **kwargs: object) -> list[tuple[QdrantPayload, float]]:
        self.last_company = kwargs.get("company_name")  # type: ignore[assignment]
        return self._hits


class _OpenAI:
    def answer_with_reasoning(self, system_prompt: str, user_prompt: str) -> ReasonedAnswer:
        return ReasonedAnswer(answer="ok", reasoning_summary="")


def test_company_mentioned_in_query_ignores_ab_suffix() -> None:
    assert company_mentioned_in_query(
        "Beskriv Testföretaget ABs konton lite ytligt",
        "Testföretaget AB",
    )
    assert not company_mentioned_in_query(
        "Beskriv Testföretaget ABs konton lite ytligt",
        "Datakonsulterna AB",
    )


def test_parse_quarter_year_from_swedish_question() -> None:
    year, quarter = parse_period_from_query("Vad var EBITDA i Q3 2024?")
    assert year == 2024
    assert quarter == 3


def test_entity_rank_puts_named_company_first() -> None:
    metas = [
        _payload("a", "Datakonsulterna AB", "konton").metadata,
        _payload("b", "Testföretaget AB", "konton").metadata,
    ]
    scope, order = entity_rank_indices(
        "Beskriv Testföretaget ABs konton",
        metas,
    )
    assert scope.company_names
    assert order is not None
    assert order[0] == 1


def test_query_drops_other_company_sie_chunks() -> None:
    hits = [
        (_payload("data", "Datakonsulterna AB", "Datakonsulterna kontoöversikt 1910"), 0.95),
        (_payload("test", "Testföretaget AB", "Testföretaget kontoöversikt 1510"), 0.55),
        (_payload("data2", "Datakonsulterna AB", "Datakonsulterna balans"), 0.90),
    ]
    service = RagService(
        embedder=_Embedder(),
        qdrant=_Qdrant(hits),
        openai_service=_OpenAI(),
        settings=Settings(openai_api_key="test-key", hybrid_lexical=True),
        reranker=None,
    )
    response = service.query(
        QueryRequest(
            question="Beskriv Testföretaget ABs konton lite ytligt",
            tenant_id="tenant_1",
            top_k=3,
            rerank=False,
        )
    )
    companies = {source.metadata.company_name for source in response.sources}
    assert companies == {"Testföretaget AB"}
