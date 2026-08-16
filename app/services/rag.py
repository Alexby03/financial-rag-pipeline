"""RAG query path — see docs/rag_strategy.md for linked-chunk retrieval rules."""

import logging
from dataclasses import dataclass
from time import perf_counter

from app.config import Settings
from app.models.api import (
    ChatTurn,
    FigureReadingTrace,
    FusionTrace,
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
from app.services.entity_scope import entity_rank_indices, filter_indices_to_scope
from app.services.lexical import bm25_rank_indices, content_phrases, tokenize
from app.services.minio_client import MinioStorage
from app.services.openai_client import OpenAIService
from app.services.qdrant_service import QdrantService
from app.services.reranker import CohereReranker
from app.services.rrf import reciprocal_rank_fusion

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
    lexical_rank: int | None = None
    rrf_score: float | None = None

RAG_SYSTEM_PROMPT = """You are an Financial analyst assistant.
Answer using only the provided context chunks in this turn.
Each chunk header includes source filename and page number — treat those as ground truth
for what you have. Never claim a page, table, or line item is missing if it appears
anywhere in Context chunks. Never ask the user to resend pages that are already in context.
Prior conversation turns may be wrong about missing data; prefer the current Context chunks
over earlier assistant denials.

Financial tables often carry period labels as
"Kolumner (vänster→höger efter radetikett): YYYY-MM-DD | YYYY-MM-DD" and/or as markdown
table header cells (Post | date1 | date2). Those dates are column headers left→right after
the row label (first amount column = first date, second = second date). When the user asks
for calendar year YYYY, use the column whose period ends in YYYY (e.g. 2024-06-30 for
"2024") unless they specify a different basis. Quote the amount with its period date.

Each chunk includes chunk_id, previous_chunk_id, and next_chunk_id (a double-linked list).
Use neighbors when a table or clause continues across chunks.
OCR figure captions are retrieval hints. When "Authoritative figure readings" are present,
prefer them for plotted numbers. If a reading is approximate, keep that caveat.
If — after reading every context chunk carefully — the answer is still not present, say so.
Cite source filenames and page numbers when relevant.
When the user names a company, year, or quarter, use only chunks for that entity. Never
blend figures or account lists from another company or period that happen to appear in
Context — say those other sources are out of scope instead of merging them."""



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
        hybrid = self._settings.hybrid_lexical

        query_vector = self._embedder.embed_query(request.question)
        hits = self._qdrant.search(
            query_vector=query_vector,
            tenant_id=request.tenant_id,
            limit=self._candidate_limit(request.top_k, reranker, hybrid),
            doc_type=request.doc_type.value if request.doc_type else None,
            category=request.category.value if request.category else None,
            year=request.year,
            quarter=request.quarter,
            company_name=request.company_name,
        )
        candidates = [
            Candidate(payload=payload, vector_score=score, vector_rank=rank)
            for rank, (payload, score) in enumerate(hits, start=1)
        ]

        selected, rerank_trace, fusion_trace = self._select_candidates(
            request, candidates, reranker, hybrid
        )
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
            fusion_trace=fusion_trace,
            figure_readings=figure_readings,
            vision_ms=vision_ms,
        )
        return QueryResponse(answer=reasoned.answer, sources=sources, trace=trace)

    def _candidate_limit(
        self,
        top_k: int,
        reranker: CohereReranker | None,
        hybrid: bool,
    ) -> int:
        if reranker is None and not hybrid:
            return top_k
        return top_k * self._settings.rerank_candidate_multiplier

    def _select_candidates(
        self,
        request: QueryRequest,
        candidates: list[Candidate],
        reranker: CohereReranker | None,
        hybrid: bool,
    ) -> tuple[list[Candidate], RerankTrace | None, FusionTrace | None]:
        if not candidates:
            return [], None, None

        rerank_trace: RerankTrace | None = None
        cohere_order: list[int] | None = None

        if reranker is not None:
            rerank_started = perf_counter()
            # Ask Cohere for the full candidate pool so RRF can use its full ranking.
            outcome = reranker.rerank(
                query=request.question,
                documents=[candidate.payload.raw_text for candidate in candidates],
                top_n=len(candidates),
            )
            rerank_trace = RerankTrace(
                model=reranker.model,
                candidates=len(candidates),
                applied=outcome.applied,
                fallback_reason=outcome.fallback_reason,
                rerank_ms=self._elapsed_ms(rerank_started),
            )
            if outcome.applied:
                cohere_order = []
                for document in outcome.documents:
                    candidates[document.index].rerank_score = document.relevance_score
                    cohere_order.append(document.index)

        if not hybrid and cohere_order is not None:
            selected = [candidates[index] for index in cohere_order[: request.top_k]]
            return self._apply_entity_scope(request, candidates, selected), rerank_trace, None

        if not hybrid:
            selected = candidates[: request.top_k]
            return self._apply_entity_scope(request, candidates, selected), rerank_trace, None

        vector_order = list(range(len(candidates)))
        documents = [candidate.payload.raw_text for candidate in candidates]
        metadatas = [candidate.payload.metadata for candidate in candidates]
        lexical_order = bm25_rank_indices(request.question, documents)
        for lexical_rank, index in enumerate(lexical_order, start=1):
            candidates[index].lexical_rank = lexical_rank

        # Dense+sparse RRF is the industry default. When Cohere ran, use its order
        # instead of raw vector order (cross-encoder already judged semantics).
        if cohere_order is not None:
            rankings = [cohere_order, lexical_order]
            tie_break = {index: rank for rank, index in enumerate(cohere_order, start=1)}
        else:
            rankings = [vector_order, lexical_order]
            tie_break = {index: rank for rank, index in enumerate(vector_order, start=1)}

        phrase_order = _exact_phrase_ranking(request.question, documents, lexical_order)
        weights = [1.0, 1.0]
        if phrase_order is not None:
            rankings.append(phrase_order)
            # Exact rare phrases must outweigh dense+BM25 agreement on cover pages.
            weights.append(2.0)
            tie_break = {index: rank for rank, index in enumerate(lexical_order, start=1)}

        scope, entity_order = entity_rank_indices(
            request.question,
            metadatas,
            explicit_company=request.company_name,
            explicit_year=request.year,
            explicit_quarter=request.quarter,
        )
        in_scope = filter_indices_to_scope(list(range(len(metadatas))), metadatas, scope)
        if entity_order is not None and 0 < len(in_scope) < len(metadatas):
            rankings.append(entity_order)
            # Stronger than phrase boost: wrong-company SIE ledgers must lose.
            weights.append(3.0)

        fused = reciprocal_rank_fusion(
            rankings,
            k=self._settings.rrf_k,
            weights=weights,
            tie_break_ranks=tie_break,
        )
        fused_indices = [index for index, _ in fused]
        fused_indices = filter_indices_to_scope(fused_indices, metadatas, scope)
        rrf_by_index = {index: score for index, score in fused}
        selected: list[Candidate] = []
        for index in fused_indices[: request.top_k]:
            candidate = candidates[index]
            candidate.rrf_score = rrf_by_index.get(index)
            selected.append(candidate)

        fusion_trace = FusionTrace(
            method="rrf",
            lexical_applied=True,
            cohere_applied=cohere_order is not None,
            candidates=len(candidates),
            rrf_k=self._settings.rrf_k,
        )
        return selected, rerank_trace, fusion_trace

    def _apply_entity_scope(
        self,
        request: QueryRequest,
        pool: list[Candidate],
        selected: list[Candidate],
    ) -> list[Candidate]:
        """Hard-drop off-entity chunks when the question names a company/period in the pool."""
        metadatas = [candidate.payload.metadata for candidate in pool]
        scope, entity_order = entity_rank_indices(
            request.question,
            metadatas,
            explicit_company=request.company_name,
            explicit_year=request.year,
            explicit_quarter=request.quarter,
        )
        if entity_order is None:
            return selected
        id_to_candidate = {
            candidate.payload.metadata.chunk_id: candidate for candidate in selected
        }
        pool_indices = {
            candidate.payload.metadata.chunk_id: index
            for index, candidate in enumerate(pool)
        }
        ordered = filter_indices_to_scope(
            [pool_indices[chunk_id] for chunk_id in id_to_candidate],
            metadatas,
            scope,
        )
        # If filtering emptied the selection, refill from in-scope pool order.
        if not ordered:
            ordered = filter_indices_to_scope(entity_order, metadatas, scope)[
                : request.top_k
            ]
            return [pool[index] for index in ordered]
        kept = [
            pool[index]
            for index in ordered
            if pool[index].payload.metadata.chunk_id in id_to_candidate
        ]
        if len(kept) >= min(request.top_k, len(selected)):
            return kept[: request.top_k]
        # Top up from in-scope pool so top_k stays filled with the right company.
        kept_ids = {candidate.payload.metadata.chunk_id for candidate in kept}
        for index in filter_indices_to_scope(entity_order, metadatas, scope):
            candidate = pool[index]
            if candidate.payload.metadata.chunk_id in kept_ids:
                continue
            kept.append(candidate)
            kept_ids.add(candidate.payload.metadata.chunk_id)
            if len(kept) >= request.top_k:
                break
        return kept

    def _build_trace(
        self,
        request: QueryRequest,
        selected: list[Candidate],
        context_payloads: list[QdrantPayload],
        retrieval_ms: int,
        generation_ms: int,
        reasoning_summary: str,
        rerank_trace: RerankTrace | None,
        fusion_trace: FusionTrace | None,
        figure_readings: list[FigureReading],
        vision_ms: int,
    ) -> RetrievalTrace:
        hit_ids = {candidate.payload.metadata.chunk_id for candidate in selected}
        filters: dict[str, str | int] = {"tenant_id": request.tenant_id}
        if request.company_name:
            filters["company_name"] = request.company_name
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
                    lexical_rank=candidate.lexical_rank,
                    rrf_score=candidate.rrf_score,
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
            fusion=fusion_trace,
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


def _exact_phrase_ranking(
    query: str,
    documents: list[str],
    lexical_order: list[int],
) -> list[int] | None:
    """Third RRF list: docs matching contiguous query phrases, then lexical order.

    Uses short content phrases (e.g. ``balanserat resultat``), not a join of every
    content word. Matches are weighted by rarity in the candidate pool so cover-page
    company-name n-grams do not outrank a unique account line.
    """
    phrases = content_phrases(query)
    if not phrases:
        return None

    joined_docs = [" ".join(tokenize(document)) for document in documents]
    phrase_df = {
        phrase: sum(1 for text in joined_docs if phrase in text) for phrase in phrases
    }
    # Discriminative phrases only: unique-in-pool first, else the rarest tier.
    # Cover pages share company-name n-grams; account lines usually appear once.
    positive_df = [n for n in phrase_df.values() if n > 0]
    if not positive_df:
        return None
    unique = [phrase for phrase, n_docs in phrase_df.items() if n_docs == 1]
    target = set(unique) if unique else {
        phrase for phrase, n_docs in phrase_df.items() if n_docs == min(positive_df)
    }

    scored: list[tuple[int, float]] = []
    for index, text in enumerate(joined_docs):
        score = 0.0
        for phrase in target:
            if phrase not in text:
                continue
            n_docs = phrase_df[phrase]
            score += len(phrase.split()) / n_docs
            if n_docs == 1:
                score += 3.0
        if score > 0:
            scored.append((index, score))
    if not scored:
        return None

    lexical_rank = {index: rank for rank, index in enumerate(lexical_order)}
    scored.sort(key=lambda item: (-item[1], lexical_rank.get(item[0], 10**9), item[0]))
    hits = [index for index, _ in scored]
    hit_set = set(hits)
    rest = [index for index in lexical_order if index not in hit_set]
    return hits + rest
