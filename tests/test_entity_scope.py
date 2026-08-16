"""Entity scoping keeps wrong-company / wrong-period chunks out of answers."""

from datetime import UTC, datetime

from app.config import Settings
from app.models.api import ChatTurn, QueryRequest
from app.models.enums import Category, DocType
from app.models.llm import ReasonedAnswer
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.services.entity_scope import (
    company_mentioned_in_query,
    entity_rank_indices,
    parse_period_from_query,
    resolve_entity_scope,
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
    def __init__(self) -> None:
        self.last_question = ""

    def embed_query(self, question: str) -> list[float]:
        self.last_question = question
        return [0.1, 0.2, 0.3]


class _Qdrant:
    def __init__(
        self,
        hits: list[tuple[QdrantPayload, float]],
        extra_by_company: dict[str, list[tuple[QdrantPayload, float]]] | None = None,
    ) -> None:
        self._hits = hits
        self._extra_by_company = extra_by_company or {}
        self.last_company: str | None = None
        self.search_count = 0

    def search(self, **kwargs: object) -> list[tuple[QdrantPayload, float]]:
        self.search_count += 1
        company = kwargs.get("company_name")
        self.last_company = company if isinstance(company, str) else None
        if not isinstance(company, str) or not company:
            return self._hits
        filtered = [
            (payload, score)
            for payload, score in self._hits
            if payload.metadata.company_name == company
        ]
        return [*filtered, *self._extra_by_company.get(company, [])]


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


def test_history_fills_company_scope_when_followup_omits_the_name() -> None:
    metas = [
        _payload("data", "Datakonsulterna AB", "konton").metadata,
        _payload("test", "Testföretaget AB", "konton").metadata,
    ]
    scope = resolve_entity_scope(
        "Berätta om de är väl positionerade i den växande marknaden",
        metas,
        history_text="main takeaways av Testföretaget Q4 presentation",
    )
    assert scope.company_names == ("Testföretaget AB",)
    assert scope.quarter is None


def test_followup_query_drops_other_company_using_chat_history() -> None:
    hits = [
        (
            _payload("data", "Datakonsulterna AB", "growing market well positioned strategy"),
            0.95,
        ),
        (
            _payload("test", "Testföretaget AB", "Testföretaget AI builders positioning"),
            0.40,
        ),
        (
            _payload("data2", "Datakonsulterna AB", "tillväxtmarknaden TAM"),
            0.90,
        ),
    ]
    embedder = _Embedder()
    service = RagService(
        embedder=embedder,
        qdrant=_Qdrant(hits),
        openai_service=_OpenAI(),
        settings=Settings(openai_api_key="test-key", hybrid_lexical=True),
        reranker=None,
    )
    response = service.query(
        QueryRequest(
            question="Berätta om de är väl positionerade i den växande marknaden",
            tenant_id="tenant_1",
            top_k=3,
            rerank=False,
            chat_history=[
                ChatTurn(
                    role="user",
                    content="Berätta the main takeaways av Testföretaget Q4 presentation",
                ),
                ChatTurn(role="assistant", content="De växte och har kassa."),
            ],
        )
    )
    companies = {source.metadata.company_name for source in response.sources}
    assert companies == {"Testföretaget AB"}
    assert "Testföretaget" in embedder.last_question
    assert "väl positionerade" in embedder.last_question
    assert response.trace.retrieval_question == embedder.last_question


def test_entity_scope_refills_top_k_from_company_search() -> None:
    """Hard-drop used to leave a single INOD hit when top_k=5 came from a mixed pool."""
    first_hits = [
        (_payload("other1", "Datakonsulterna AB", "growing market well positioned"), 0.99),
        (_payload("other2", "Datakonsulterna AB", "tillväxtmarknaden TAM"), 0.98),
        (_payload("other3", "Datakonsulterna AB", "strategy growth"), 0.97),
        (_payload("inod1", "Testföretaget AB", "Testföretaget Q4 highlights"), 0.40),
    ]
    extra = [
        (_payload("inod2", "Testföretaget AB", "Testföretaget kassa och net cash"), 0.88),
        (_payload("inod3", "Testföretaget AB", "Testföretaget AI builders"), 0.87),
        (_payload("inod4", "Testföretaget AB", "Testföretaget market position"), 0.86),
    ]
    qdrant = _Qdrant(first_hits, extra_by_company={"Testföretaget AB": extra})
    service = RagService(
        embedder=_Embedder(),
        qdrant=qdrant,
        openai_service=_OpenAI(),
        settings=Settings(openai_api_key="test-key", hybrid_lexical=True),
        reranker=None,
    )
    response = service.query(
        QueryRequest(
            question="Beskriv Testföretaget ABs position i marknaden",
            tenant_id="tenant_1",
            top_k=3,
            rerank=False,
        )
    )
    assert qdrant.search_count == 2
    assert qdrant.last_company == "Testföretaget AB"
    assert len(response.sources) == 3
    assert {source.metadata.company_name for source in response.sources} == {
        "Testföretaget AB"
    }
    assert {source.metadata.chunk_id for source in response.sources} <= {
        "inod1",
        "inod2",
        "inod3",
        "inod4",
    }

