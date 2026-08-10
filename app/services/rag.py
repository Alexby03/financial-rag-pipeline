"""RAG query path — see docs/rag_strategy.md for linked-chunk retrieval rules."""

from app.models.api import QueryRequest, QueryResponse, SourceChunk
from app.models.metadata import QdrantPayload
from app.services.embedder import Embedder
from app.services.openai_client import OpenAIService
from app.services.qdrant_service import QdrantService

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
    ) -> None:
        self._embedder = embedder
        self._qdrant = qdrant
        self._openai = openai_service

    def query(self, request: QueryRequest) -> QueryResponse:
        query_vector = self._embedder.embed_query(request.question)
        hits = self._qdrant.search(
            query_vector=query_vector,
            tenant_id=request.tenant_id,
            limit=request.top_k,
            doc_type=request.doc_type.value if request.doc_type else None,
            category=request.category.value if request.category else None,
            year=request.year,
            quarter=request.quarter,
        )

        context_payloads = self._resolve_context(hits, request.expand_neighbors)
        context_text = self._format_context(context_payloads)
        answer = self._openai.chat_completion(
            RAG_SYSTEM_PROMPT,
            f"Question:\n{request.question}\n\nContext:\n{context_text}",
        )

        sources = [
            SourceChunk(raw_text=payload.raw_text, metadata=payload.metadata, score=score)
            for payload, score in hits
        ]
        return QueryResponse(answer=answer, sources=sources)

    def _resolve_context(
        self,
        hits: list[tuple[QdrantPayload, float]],
        expand_neighbors: bool,
    ) -> list[QdrantPayload]:
        if not hits:
            return []

        hit_payloads = [payload for payload, _ in hits]
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
