from datetime import UTC, datetime
from typing import Any

from app.config import Settings
from app.models.api import ChatTurn, QueryRequest
from app.models.enums import Category, DocType
from app.models.llm import ReasonedAnswer
from app.models.metadata import ChunkMetadata, QdrantPayload
from app.models.rerank import RerankedDocument, RerankOutcome
from app.services.lexical import bm25_rank_indices
from app.services.rag import RagService
from app.services.rrf import reciprocal_rank_fusion

RERANK_SETTINGS = Settings(
    openai_api_key="test-key",
    cohere_api_base="https://endpoint.models.ai.azure.com/v1",
    cohere_api_key="azure-key",
    rerank_candidate_multiplier=4,
    hybrid_lexical=True,
)

NO_HYBRID_SETTINGS = Settings(
    openai_api_key="test-key",
    cohere_api_base="https://endpoint.models.ai.azure.com/v1",
    cohere_api_key="azure-key",
    rerank_candidate_multiplier=4,
    hybrid_lexical=False,
)


def build_payload(
    chunk_id: str,
    text: str,
    *,
    company_name: str = "SME Corp AB",
    source_filename: str = "report.pdf",
) -> QdrantPayload:
    return QdrantPayload(
        raw_text=text,
        metadata=ChunkMetadata(
            tenant_id="tenant_1",
            company_name=company_name,
            doc_type=DocType.FINANCIAL_REPORT,
            category=Category.FINANCING,
            year=2024,
            quarter=3,
            source_filename=source_filename,
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
    def __init__(self, hits: list[tuple[QdrantPayload, float]] | None = None) -> None:
        self.requested_limit = 0
        self._hits = hits or [
            (build_payload("c1", "irrelevant prose"), 0.50),
            (build_payload("c2", "the financial table nettoomsattningen"), 0.45),
            (build_payload("c3", "more prose"), 0.40),
        ]

    def search(self, **kwargs: Any) -> list[tuple[QdrantPayload, float]]:
        self.requested_limit = kwargs["limit"]
        return self._hits


class FakeOpenAI:
    def answer_with_reasoning(self, system_prompt: str, user_prompt: str) -> ReasonedAnswer:
        return ReasonedAnswer(answer="svar", reasoning_summary="tankekedja")


class FakeReranker:
    def __init__(self, outcome: RerankOutcome) -> None:
        self._outcome = outcome
        self.calls = 0
        self.last_top_n: int | None = None
        self.last_query = ""

    @property
    def model(self) -> str:
        return "rerank-multilingual-v3.0"

    def rerank(self, query: str, documents: list[str], top_n: int) -> RerankOutcome:
        self.calls += 1
        self.last_top_n = top_n
        self.last_query = query
        return self._outcome


def build_service(
    reranker: FakeReranker | None,
    qdrant: FakeQdrant,
    settings: Settings = RERANK_SETTINGS,
) -> RagService:
    return RagService(
        embedder=FakeEmbedder(),
        qdrant=qdrant,
        openai_service=FakeOpenAI(),
        settings=settings,
        reranker=reranker,
    )


def make_request(top_k: int = 2, rerank: bool = True) -> QueryRequest:
    return QueryRequest(
        question="Vad var nettoomsattningen?",
        tenant_id="tenant_1",
        top_k=top_k,
        rerank=rerank,
    )


def test_rrf_prefers_shared_top_ranks() -> None:
    fused = reciprocal_rank_fusion([[0, 1, 2], [1, 0, 2]], k=60)
    assert fused[0][0] == 0 or fused[0][0] == 1
    assert {index for index, _ in fused} == {0, 1, 2}


def test_bm25_promotes_exact_account_name() -> None:
    docs = [
        "Allmän text om bolagets likviditet och kassaflöde.",
        "EGET KAPITAL\nBalanserat resultat\t67 646\t40 803",
        "Noter till årsredovisningen utan belopp.",
    ]
    order = bm25_rank_indices("Balanserat resultat", docs)
    assert order[0] == 1


def test_content_phrases_split_long_swedish_question() -> None:
    from app.services.lexical import content_phrases

    phrases = content_phrases(
        "vad var Balanserat resultat i kr 2024 för RTP Radical Technology Products Aktiebolag"
    )
    assert "balanserat resultat" in phrases
    assert not any(phrase.startswith("balanserat resultat kr") for phrase in phrases)
    # Full-question join must never be treated as one phrase.
    assert all("2024" not in phrase for phrase in phrases)


def test_hybrid_rrf_promotes_account_line_over_company_cover() -> None:
    """Long questions used to join all content words into one non-matching phrase."""
    hits = [
        (
            build_payload(
                "cover",
                "Styrelsen för RTP Radical Technology Products Aktiebolag avger "
                "årsredovisning för räkenskapsåret 2024-07-01 - 2025-06-30.",
            ),
            0.92,
        ),
        (
            build_payload(
                "disposition",
                "Förvaltningsberättelse Resultatdisposition "
                "RTP Radical Technology Products Aktiebolag 556285-8117",
            ),
            0.90,
        ),
        (
            build_payload(
                "equity",
                "EGET KAPITAL OCH SKULDER\n"
                "| Post | 2025-06-30 | 2024-06-30 |\n"
                "| Balanserat resultat | 67 646 | 40 803 |",
            ),
            0.55,
        ),
    ]
    response = build_service(None, FakeQdrant(hits)).query(
        QueryRequest(
            question=(
                "vad var Balanserat resultat i kr 2024 för "
                "RTP Radical Technology Products Aktiebolag"
            ),
            tenant_id="tenant_1",
            top_k=3,
            rerank=False,
        )
    )

    assert response.sources[0].metadata.chunk_id == "equity"


def test_hybrid_rrf_promotes_exact_term_over_vector_leader() -> None:
    hits = [
        (build_payload("prose", "Likviditeten stärktes under året."), 0.91),
        (build_payload("equity", "Balanserat resultat 67 646 kronor"), 0.55),
        (build_payload("other", "Övriga noter utan relevans"), 0.50),
    ]
    qdrant = FakeQdrant(hits)
    response = build_service(None, qdrant).query(
        QueryRequest(
            question="Vad är Balanserat resultat?",
            tenant_id="tenant_1",
            top_k=2,
            rerank=False,
        )
    )

    assert qdrant.requested_limit == 8
    assert response.sources[0].metadata.chunk_id == "equity"
    assert response.trace.fusion is not None
    assert response.trace.fusion.lexical_applied
    assert response.trace.hits[0].lexical_rank == 1
    assert response.trace.hits[0].rrf_score is not None


def test_rerank_reorders_sources_and_widens_candidate_pool() -> None:
    outcome = RerankOutcome(
        applied=True,
        documents=[
            RerankedDocument(index=1, relevance_score=0.97),
            RerankedDocument(index=0, relevance_score=0.31),
            RerankedDocument(index=2, relevance_score=0.10),
        ],
    )
    qdrant = FakeQdrant()
    reranker = FakeReranker(outcome)
    response = build_service(reranker, qdrant).query(make_request())

    assert qdrant.requested_limit == 8
    assert reranker.last_top_n == 3
    assert [source.metadata.chunk_id for source in response.sources] == ["c2", "c1"]
    assert response.sources[0].rerank_score == 0.97
    assert response.sources[0].score == 0.45
    assert response.trace.rerank is not None
    assert response.trace.rerank.applied
    assert response.trace.fusion is not None
    assert response.trace.fusion.cohere_applied
    assert response.trace.fusion.method == "cohere"
    assert response.trace.hits[0].vector_rank == 2


def test_rerank_failure_keeps_hybrid_order() -> None:
    outcome = RerankOutcome(applied=False, fallback_reason="Rerank endpoint returned 401")
    response = build_service(FakeReranker(outcome), FakeQdrant()).query(make_request())

    assert response.trace.rerank is not None
    assert not response.trace.rerank.applied
    assert "401" in response.trace.rerank.fallback_reason
    assert response.trace.fusion is not None
    assert response.trace.fusion.lexical_applied
    assert not response.trace.fusion.cohere_applied


def test_rerank_can_be_disabled_per_query() -> None:
    reranker = FakeReranker(RerankOutcome(applied=True, documents=[]))
    qdrant = FakeQdrant()
    response = build_service(reranker, qdrant).query(make_request(rerank=False))

    assert reranker.calls == 0
    assert qdrant.requested_limit == 8  # hybrid still widens the pool
    assert response.trace.rerank is None
    assert response.trace.fusion is not None


def test_hybrid_can_be_disabled() -> None:
    qdrant = FakeQdrant()
    response = build_service(None, qdrant, settings=NO_HYBRID_SETTINGS).query(
        make_request(rerank=False)
    )

    assert qdrant.requested_limit == 2
    assert response.trace.fusion is None
    assert [source.metadata.chunk_id for source in response.sources] == ["c1", "c2"]


def test_cohere_order_beats_bm25_when_rerank_applied() -> None:
    """Generic 'growing market' copy can win BM25/RRF while Cohere scores INOD 0.96."""
    hits = [
        (
            build_payload(
                "generic",
                "De är väl positionerade i den växande marknaden. Strategy growth TAM.",
                company_name="Other Co AB",
                source_filename="q3_report_2024.pdf",
            ),
            0.91,
        ),
        (
            build_payload(
                "inod",
                "INOD Q4: AI builders and adopters, optimize positioning.",
                company_name="INOD AB",
                source_filename="inod_q4_investor_presentation.pdf",
            ),
            0.40,
        ),
        (
            build_payload(
                "other",
                "Unrelated notes without market language.",
                company_name="Other Co AB",
                source_filename="q3_report_2024.pdf",
            ),
            0.35,
        ),
    ]
    outcome = RerankOutcome(
        applied=True,
        documents=[
            RerankedDocument(index=1, relevance_score=0.96),
            RerankedDocument(index=0, relevance_score=0.62),
            RerankedDocument(index=2, relevance_score=0.10),
        ],
    )
    reranker = FakeReranker(outcome)
    response = build_service(reranker, FakeQdrant(hits)).query(
        QueryRequest(
            question="berätta om de är väl positionerade i den växande marknaden",
            tenant_id="tenant_1",
            top_k=2,
            rerank=True,
        )
    )

    assert [source.metadata.chunk_id for source in response.sources] == ["inod", "generic"]
    assert response.sources[0].rerank_score == 0.96
    assert response.trace.fusion is not None
    assert response.trace.fusion.method == "cohere"
    assert response.trace.hits[0].lexical_rank is not None
    assert response.trace.hits[0].lexical_rank > 1


def test_rerank_query_includes_prior_user_turn() -> None:
    outcome = RerankOutcome(
        applied=True,
        documents=[
            RerankedDocument(index=0, relevance_score=0.9),
            RerankedDocument(index=1, relevance_score=0.2),
            RerankedDocument(index=2, relevance_score=0.1),
        ],
    )
    reranker = FakeReranker(outcome)
    build_service(reranker, FakeQdrant()).query(
        QueryRequest(
            question="berätta om de är väl positionerade",
            tenant_id="tenant_1",
            top_k=2,
            rerank=True,
            chat_history=[
                ChatTurn(role="user", content="main takeaways av inods Q4 presentation"),
                ChatTurn(role="assistant", content="De växte."),
            ],
        )
    )
    assert "inods Q4" in reranker.last_query
    assert "väl positionerade" in reranker.last_query

