"""RAG query path — see docs/rag_strategy.md for linked-chunk retrieval rules."""

from dataclasses import dataclass
from time import perf_counter

from app.config import Settings
from app.models.api import (
    QueryRequest,
    QueryResponse,
    RerankTrace,
    RetrievalTrace,
    RetrievedChunkTrace,
    SourceChunk,
)
from app.models.metadata import QdrantPayload
from app.services.embedder import Embedder
from app.services.openai_client import OpenAIService
from app.services.qdrant_service import QdrantService
from app.services.reranker import CohereReranker

PREVIEW_LENGTH = 160


@dataclass
class Candidate:
    payload: QdrantPayload
    vector_score: float
    vector_rank: int
    rerank_score: float | None = None

RAG_SYSTEM_PROMPT = """You are an Financial analyst assistant.
Answer using only the provided raw_text context chunks.
Each chunk includes metadata with chunk_id, previous_chunk_id, and next_chunk_id.
These form a double-linked list within each document — adjacent chunks may continue
tables, clauses, or org-chart details. Use neighboring context only when the retrieved
chunk appears incomplete or the question spans multiple sections.
If the context is insufficient, say you do not have enough information.
Cite source filenames and page numbers when relevant."""


class RagService:
    def __init__(
        self,
        embedder: Embedder,
        qdrant: QdrantService,
        openai_service: OpenAIService,
        settings: Settings,
        reranker: CohereReranker | None = None,
    ) -> None:
        self._embedder = embedder
        self._qdrant = qdrant
        self._openai = openai_service
        self._settings = settings
        self._reranker = reranker

    def query(self, request: QueryRequest) -> QueryResponse:
        retrieval_started = perf_counter()
        reranker = self._reranker if request.rerank else None

        query_vector = self._embedder.embed_query(request.question)
        hits = self._qdrant.search(
            query_vector=query_vector,
            tenant_id=request.tenant_id,
            limit=self._candidate_limit(request.top_k, reranker),
            doc_type=request.doc_type.value if request.doc_type else None,
            category=request.category.value if request.category else None,
            year=request.year,
            quarter=request.quarter,
        )
        candidates = [
            Candidate(payload=payload, vector_score=score, vector_rank=rank)
            for rank, (payload, score) in enumerate(hits, start=1)
        ]

        selected, rerank_trace = self._apply_rerank(request, candidates, reranker)
        context_payloads = self._resolve_context(
            [candidate.payload for candidate in selected], request.expand_neighbors
        )
        retrieval_ms = self._elapsed_ms(retrieval_started)

        context_text = self._format_context(context_payloads)
        generation_started = perf_counter()
        reasoned = self._openai.answer_with_reasoning(
            RAG_SYSTEM_PROMPT,
            f"Question:\n{request.question}\n\nContext:\n{context_text}",
        )
        generation_ms = self._elapsed_ms(generation_started)

        sources = [
            SourceChunk(
                raw_text=candidate.payload.raw_text,
                metadata=candidate.payload.metadata,
                score=candidate.vector_score,
                rerank_score=candidate.rerank_score,
            )
            for candidate in selected
        ]
        trace = self._build_trace(
            request=request,
            selected=selected,
            context_payloads=context_payloads,
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            reasoning_summary=reasoned.reasoning_summary,
            rerank_trace=rerank_trace,
        )
        return QueryResponse(answer=reasoned.answer, sources=sources, trace=trace)

    def _candidate_limit(self, top_k: int, reranker: CohereReranker | None) -> int:
        if reranker is None:
            return top_k
        return top_k * self._settings.rerank_candidate_multiplier

    def _apply_rerank(
        self,
        request: QueryRequest,
        candidates: list[Candidate],
        reranker: CohereReranker | None,
    ) -> tuple[list[Candidate], RerankTrace | None]:
        if reranker is None:
            return candidates[: request.top_k], None

        rerank_started = perf_counter()
        outcome = reranker.rerank(
            query=request.question,
            documents=[candidate.payload.raw_text for candidate in candidates],
            top_n=request.top_k,
        )
        trace = RerankTrace(
            model=reranker.model,
            candidates=len(candidates),
            applied=outcome.applied,
            fallback_reason=outcome.fallback_reason,
            rerank_ms=self._elapsed_ms(rerank_started),
        )

        if not outcome.applied:
            return candidates[: request.top_k], trace

        reordered: list[Candidate] = []
        for document in outcome.documents:
            candidate = candidates[document.index]
            candidate.rerank_score = document.relevance_score
            reordered.append(candidate)
        return reordered[: request.top_k], trace

    def _build_trace(
        self,
        request: QueryRequest,
        selected: list[Candidate],
        context_payloads: list[QdrantPayload],
        retrieval_ms: int,
        generation_ms: int,
        reasoning_summary: str,
        rerank_trace: RerankTrace | None,
    ) -> RetrievalTrace:
        hit_ids = {candidate.payload.metadata.chunk_id for candidate in selected}
        filters: dict[str, str | int] = {"tenant_id": request.tenant_id}
        if request.doc_type:
            filters["doc_type"] = request.doc_type.value
        if request.category:
            filters["category"] = request.category.value
        if request.year is not None:
            filters["year"] = request.year
        if request.quarter is not None:
            filters["quarter"] = request.quarter

        return RetrievalTrace(
            embedding_model=self._settings.openai_embedding_model,
            answer_model=self._settings.openai_answer_model,
            filters=filters,
            top_k=request.top_k,
            hits=[
                RetrievedChunkTrace(
                    chunk_id=candidate.payload.metadata.chunk_id,
                    source_filename=candidate.payload.metadata.source_filename,
                    page_number=candidate.payload.metadata.page_number,
                    score=candidate.vector_score,
                    preview=self._preview(candidate.payload.raw_text),
                    vector_rank=candidate.vector_rank,
                    rerank_score=candidate.rerank_score,
                )
                for candidate in selected
            ],
            neighbor_chunk_ids=[
                payload.metadata.chunk_id
                for payload in context_payloads
                if payload.metadata.chunk_id not in hit_ids
            ],
            context_chunk_count=len(context_payloads),
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            reasoning_summary=reasoning_summary,
            rerank=rerank_trace,
        )

    @staticmethod
    def _elapsed_ms(started: float) -> int:
        return int((perf_counter() - started) * 1000)

    @staticmethod
    def _preview(raw_text: str) -> str:
        collapsed = " ".join(raw_text.split())
        if len(collapsed) <= PREVIEW_LENGTH:
            return collapsed
        return f"{collapsed[:PREVIEW_LENGTH]}..."

    def _resolve_context(
        self,
        hit_payloads: list[QdrantPayload],
        expand_neighbors: bool,
    ) -> list[QdrantPayload]:
        if not hit_payloads:
            return []

        if not expand_neighbors:
            return hit_payloads

        by_id: dict[str, QdrantPayload] = {
            payload.metadata.chunk_id: payload for payload in hit_payloads
        }
        neighbor_ids: list[str] = []
        for payload in hit_payloads:
            if payload.metadata.previous_chunk_id:
                neighbor_ids.append(payload.metadata.previous_chunk_id)
            if payload.metadata.next_chunk_id:
                neighbor_ids.append(payload.metadata.next_chunk_id)

        for neighbor in self._qdrant.get_by_chunk_ids(
            [chunk_id for chunk_id in neighbor_ids if chunk_id not in by_id]
        ):
            by_id[neighbor.metadata.chunk_id] = neighbor

        return self._order_by_chunk_chain(list(by_id.values()))

    @staticmethod
    def _order_by_chunk_chain(payloads: list[QdrantPayload]) -> list[QdrantPayload]:
        by_id = {payload.metadata.chunk_id: payload for payload in payloads}
        by_file: dict[str, list[QdrantPayload]] = {}
        for payload in payloads:
            by_file.setdefault(payload.metadata.source_filename, []).append(payload)

        ordered: list[QdrantPayload] = []
        seen: set[str] = set()

        for group in by_file.values():
            group_ids = {payload.metadata.chunk_id for payload in group}
            start = group[0]
            while (
                start.metadata.previous_chunk_id
                and start.metadata.previous_chunk_id in group_ids
            ):
                start = by_id[start.metadata.previous_chunk_id]

            current: QdrantPayload | None = start
            while current is not None:
                if current.metadata.chunk_id in group_ids and current.metadata.chunk_id not in seen:
                    ordered.append(current)
                    seen.add(current.metadata.chunk_id)
                next_id = current.metadata.next_chunk_id
                current = by_id.get(next_id) if next_id else None

        return ordered

    @staticmethod
    def _format_context(payloads: list[QdrantPayload]) -> str:
        sections: list[str] = []
        for payload in payloads:
            meta = payload.metadata
            sections.append(
                "\n".join(
                    [
                        (
                            f"[{meta.source_filename} p.{meta.page_number} "
                            f"chunk={meta.chunk_id} "
                            f"prev={meta.previous_chunk_id} next={meta.next_chunk_id}]"
                        ),
                        payload.raw_text,
                    ]
                )
            )
        return "\n\n---\n\n".join(sections)
