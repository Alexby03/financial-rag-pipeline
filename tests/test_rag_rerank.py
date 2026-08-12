from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.models.api import QueryRequest
from app.models.enums import Category, DocType
from app.models.llm import ReasonedAnswer
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.models.rerank import RerankedDocument, RerankOutcome
from app.services.rag import RagService

RERANK_SETTINGS = Settings(
    openai_api_key="test-key",
    cohere_api_base="https://endpoint.models.ai.azure.com/v1",
    cohere_api_key="azure-key",
    rerank_candidate_multiplier=4,
)


def build_payload(chunk_id: str, text: str) -> QdrantPayload:
    return QdrantPayload(
        raw_text=text,
        metadata=ChunkMetadata(
            tenant_id="tenant_1",
            company_name="SME Corp AB",
            doc_type=DocType.FINANCIAL_REPORT,
            category=Category.FINANCING,
            year=2024,
            quarter=3,
            source_filename="report.pdf",
            object_key="tenant_1/doc-1/report.pdf",
            page_number=1,
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
    def __init__(self) -> None:
        self.requested_limit = 0

    def search(self, **kwargs: Any) -> list[tuple[QdrantPayload, float]]:
        self.requested_limit = kwargs["limit"]
        return [
            (build_payload("c1", "irrelevant prose"), 0.50),
            (build_payload("c2", "the financial table"), 0.45),
            (build_payload("c3", "more prose"), 0.40),
        ]


class FakeOpenAI:
    def answer_with_reasoning(self, system_prompt: str, user_prompt: str) -> ReasonedAnswer:
        return ReasonedAnswer(answer="svar", reasoning_summary="tankekedja")


class FakeReranker:
    def __init__(self, outcome: RerankOutcome) -> None:
        self._outcome = outcome
        self.calls = 0

    @property
    def model(self) -> str:
        return "rerank-multilingual-v3.0"

    def rerank(self, query: str, documents: list[str], top_n: int) -> RerankOutcome:
        self.calls += 1
        return self._outcome


def build_service(reranker: FakeReranker | None, qdrant: FakeQdrant) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        qdrant=qdrant,
        openai_service=FakeOpenAI(),
        settings=RERANK_SETTINGS,
        reranker=reranker,
    )


def make_request(top_k: int = 2, rerank: bool = True) -> QueryRequest:
    return QueryRequest(
        question="Vad var nettoomsattningen?",
        tenant_id="tenant_1",
        top_k=top_k,
        rerank=rerank,
    )


def test_rerank_reorders_sources_and_widens_candidate_pool() -> None:
    outcome = RerankOutcome(
        applied=True,
        documents=[
            RerankedDocument(index=1, relevance_score=0.97),
            RerankedDocument(index=0, relevance_score=0.31),
        ],
    )
    qdrant = FakeQdrant()
    response = build_service(FakeReranker(outcome), qdrant).query(make_request())

    assert qdrant.requested_limit == 8
    assert [source.metadata.chunk_id for source in response.sources] == ["c2", "c1"]
    assert response.sources[0].rerank_score == 0.97
    assert response.sources[0].score == 0.45
    assert response.trace.rerank is not None
    assert response.trace.rerank.applied
    assert response.trace.rerank.candidates == 3
    assert response.trace.hits[0].vector_rank == 2


def test_rerank_failure_keeps_vector_order() -> None:
    outcome = RerankOutcome(applied=False, fallback_reason="Rerank endpoint returned 401")
    response = build_service(FakeReranker(outcome), FakeQdrant()).query(make_request())

    assert [source.metadata.chunk_id for source in response.sources] == ["c1", "c2"]
    assert response.sources[0].rerank_score is None
    assert response.trace.rerank is not None
    assert not response.trace.rerank.applied
    assert "401" in response.trace.rerank.fallback_reason


def test_rerank_can_be_disabled_per_query() -> None:
    reranker = FakeReranker(RerankOutcome(applied=True, documents=[]))
    qdrant = FakeQdrant()
    response = build_service(reranker, qdrant).query(make_request(rerank=False))

    assert reranker.calls == 0
    assert qdrant.requested_limit == 2
    assert response.trace.rerank is None
    assert [source.metadata.chunk_id for source in response.sources] == ["c1", "c2"]


def test_service_without_reranker_is_unchanged() -> None:
    qdrant = FakeQdrant()
    response = build_service(None, qdrant).query(make_request())

    assert qdrant.requested_limit == 2
    assert response.trace.rerank is None
