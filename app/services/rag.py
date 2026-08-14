"""RAG query path — see docs/rag_strategy.md for linked-chunk retrieval rules."""

import logging
from dataclasses import dataclass
from time import perf_counter

from app.config import Settings
from app.models.api import (
    ChatTurn,
    FigureReadingTrace,
    QueryRequest,
    QueryResponse,
    RerankTrace,
    RetrievalTrace,
    RetrievedChunkTrace,
    SourceChunk,
)
from app.models.llm import FigureReading
from app.models.metadata import QdrantPayload
from app.services.embedder import Embedder
from app.services.minio_client import MinioStorage
from app.services.openai_client import OpenAIService
from app.services.qdrant_service import QdrantService
from app.services.reranker import CohereReranker

logger = logging.getLogger(__name__)

PREVIEW_LENGTH = 160
MAX_FIGURE_READINGS = 4
MAX_CHAT_HISTORY_TURNS = 20


@dataclass
class Candidate:
    payload: QdrantPayload
    vector_score: float
    vector_rank: int
    rerank_score: float | None = None

RAG_SYSTEM_PROMPT = """You are an Financial analyst assistant.
Answer using only the provided context.
Each chunk includes metadata with chunk_id, previous_chunk_id, and next_chunk_id.
These form a double-linked list within each document — adjacent chunks may continue
tables, clauses, or org-chart details. Use neighboring context only when the retrieved
chunk appears incomplete or the question spans multiple sections.
OCR captions on figure chunks are retrieval hints and are often incomplete.
When the prompt includes "Authoritative figure readings", a vision model has already
looked at the stored figure crops. Those readings are primary evidence for plotted
values (years, totals, series). Prefer them over captions and over prose that does not
state the same number. Never claim that a year or value is missing if an authoritative
figure reading reports it. If a reading is approximate, keep that caveat in the answer.
If the context is still insufficient after those readings, say so.
Prior conversation turns may be included — resolve follow-ups against that dialogue and
the current retrieval context together.
Cite source filenames and page numbers when relevant."""



class RagService:
    def __init__(
        self,
        embedder: Embedder,
        qdrant: QdrantService,
        openai_service: OpenAIService,
        settings: Settings,
        reranker: CohereReranker | None = None,
        storage: MinioStorage | None = None,
    ) -> None:
        self._embedder = embedder
        self._qdrant = qdrant
        self._openai = openai_service
        self._settings = settings
        self._reranker = reranker
        self._storage = storage

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

        vision_started = perf_counter()
        figure_readings = self._read_figures(
            request.question,
            [candidate.payload for candidate in selected],
            context_payloads,
        )
        vision_ms = self._elapsed_ms(vision_started)

        context_text = self._format_context(context_payloads, figure_readings)
        generation_started = perf_counter()
        reasoned = self._openai.answer_with_reasoning(
            RAG_SYSTEM_PROMPT,
            self._build_user_prompt(
                request.question,
                context_text,
                figure_readings,
                request.chat_history,
            ),
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
            figure_readings=figure_readings,
            vision_ms=vision_ms,
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
        figure_readings: list[FigureReading],
        vision_ms: int,
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
            vision_ms=vision_ms,
            vision_model=self._settings.openai_vision_model if figure_readings else "",
            reasoning_summary=reasoning_summary,
            rerank=rerank_trace,
            figure_readings=[
                FigureReadingTrace(
                    chunk_id=reading.chunk_id,
                    source_filename=reading.source_filename,
                    page_number=reading.page_number,
                    applied=bool(reading.reading) and not reading.error,
                    reading=reading.reading,
                    preview=self._preview(reading.reading or reading.error),
                    error=reading.error,
                )
                for reading in figure_readings
            ],
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

    def _read_figures(
        self,
        question: str,
        selected: list[QdrantPayload],
        context_payloads: list[QdrantPayload],
    ) -> list[FigureReading]:
        if self._storage is None or not self._settings.openai_vision_model:
            return []

        readings: list[FigureReading] = []
        for payload in self._figure_payloads(selected, context_payloads):
            meta = payload.metadata
            key = meta.figure_object_key
            if key is None:
                continue
            source = f"{meta.source_filename} p.{meta.page_number}"
            try:
                image_bytes = self._storage.download(key)
                reading = self._openai.read_figure(
                    question=question,
                    caption=payload.raw_text,
                    image_bytes=image_bytes,
                    source=source,
                )
                logger.info(
                    "Vision model %s read %s (%d bytes): %s",
                    self._settings.openai_vision_model,
                    key,
                    len(image_bytes),
                    reading[:240],
                )
                readings.append(
                    FigureReading(
                        chunk_id=meta.chunk_id,
                        source_filename=meta.source_filename,
                        page_number=meta.page_number,
                        reading=reading,
                    )
                )
            except Exception as exc:
                logger.exception("Vision read failed for figure %s", key)
                readings.append(
                    FigureReading(
                        chunk_id=meta.chunk_id,
                        source_filename=meta.source_filename,
                        page_number=meta.page_number,
                        error=str(exc),
                    )
                )
        return readings

    @staticmethod
    def _figure_payloads(
        selected: list[QdrantPayload],
        context_payloads: list[QdrantPayload],
    ) -> list[QdrantPayload]:
        ordered: list[QdrantPayload] = []
        seen: set[str] = set()
        for payload in [*selected, *context_payloads]:
            chunk_id = payload.metadata.chunk_id
            if not payload.metadata.figure_object_key or chunk_id in seen:
                continue
            seen.add(chunk_id)
            ordered.append(payload)
            if len(ordered) >= MAX_FIGURE_READINGS:
                break
        return ordered

    @staticmethod
    def _build_user_prompt(
        question: str,
        context_text: str,
        figure_readings: list[FigureReading],
        chat_history: list[ChatTurn] | None = None,
    ) -> str:
        parts: list[str] = []
        history = list(chat_history or [])[-MAX_CHAT_HISTORY_TURNS:]
        if history:
            parts.append("Conversation so far:")
            for turn in history:
                label = "User" if turn.role == "user" else "Assistant"
                parts.append(f"{label}: {turn.content}")
        parts.append(f"Current question:\n{question}")
        applied = [reading for reading in figure_readings if reading.reading]
        if applied:
            parts.append(
                "Authoritative figure readings "
                "(vision model looked at the stored crops — use these for plotted values; "
                "do not claim a value is missing if it appears here):"
            )
            for index, reading in enumerate(applied, start=1):
                parts.append(
                    f"{index}. {reading.source_filename} p.{reading.page_number}:\n"
                    f"{reading.reading}"
                )
        parts.append(f"Context chunks:\n{context_text}")
        return "\n\n".join(parts)

    @staticmethod
    def _format_context(
        payloads: list[QdrantPayload],
        figure_readings: list[FigureReading] | None = None,
    ) -> str:
        readings_by_id = {
            reading.chunk_id: reading
            for reading in (figure_readings or [])
            if reading.reading
        }
        sections: list[str] = []
        for payload in payloads:
            meta = payload.metadata
            kind = "figure" if meta.figure_object_key else "text"
            lines = [
                (
                    f"[{meta.source_filename} p.{meta.page_number} {kind} "
                    f"chunk={meta.chunk_id} "
                    f"prev={meta.previous_chunk_id} next={meta.next_chunk_id}]"
                ),
            ]
            if meta.figure_object_key:
                lines.append("Caption:")
            lines.append(payload.raw_text)
            reading = readings_by_id.get(meta.chunk_id)
            if reading is not None:
                lines.extend(["", "Vision reading:", reading.reading])
            sections.append("\n".join(lines))
        return "\n\n---\n\n".join(sections)
