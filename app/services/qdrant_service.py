import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import Settings
from app.models.metadata import QdrantPayload


class QdrantService:
    def __init__(self, settings: Settings) -> None:
        self._collection = settings.qdrant_collection
        self._client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)

    @property
    def collection(self) -> str:
        return self._collection

    def ping(self) -> bool:
        try:
            self._client.get_collections()
            return True
        except Exception:
            return False

    def upsert_chunks(self, vectors: list[list[float]], payloads: list[QdrantPayload]) -> int:
        validated_payloads = [
            QdrantPayload.model_validate(payload.model_dump()) for payload in payloads
        ]
        points = [
            qmodels.PointStruct(
                id=payload.metadata.chunk_id,
                vector=vector,
                payload=payload.to_qdrant_dict(),
            )
            for vector, payload in zip(vectors, validated_payloads, strict=True)
        ]
        self._client.upsert(collection_name=self._collection, points=points)
        return len(points)

    def search(
        self,
        query_vector: list[float],
        tenant_id: str,
        limit: int,
        doc_type: str | None = None,
        category: str | None = None,
        year: int | None = None,
        quarter: int | None = None,
    ) -> list[tuple[QdrantPayload, float]]:
        must: list[qmodels.FieldCondition] = [
            qmodels.FieldCondition(
                key="metadata.tenant_id",
                match=qmodels.MatchValue(value=tenant_id),
            )
        ]
        if doc_type:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.doc_type",
                    match=qmodels.MatchValue(value=doc_type),
                )
            )
        if category:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.category",
                    match=qmodels.MatchValue(value=category),
                )
            )
        if year is not None:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.year",
                    match=qmodels.MatchValue(value=year),
                )
            )
        if quarter is not None:
            must.append(
                qmodels.FieldCondition(
                    key="metadata.quarter",
                    match=qmodels.MatchValue(value=quarter),
                )
            )

        response = self._client.query_points(
            collection_name=self._collection,
            query=query_vector,
            query_filter=qmodels.Filter(must=must),
            limit=limit,
            with_payload=True,
        )

        hits: list[tuple[QdrantPayload, float]] = []
        for hit in response.points:
            if hit.payload is None:
                continue
            payload = QdrantPayload.model_validate(hit.payload)
            hits.append((payload, hit.score or 0.0))
        return hits

    def get_by_chunk_ids(self, chunk_ids: list[str]) -> list[QdrantPayload]:
        if not chunk_ids:
            return []

        records = self._client.retrieve(
            collection_name=self._collection,
            ids=chunk_ids,
            with_payload=True,
        )
        payloads: list[QdrantPayload] = []
        for record in records:
            if record.payload is None:
                continue
            payloads.append(QdrantPayload.model_validate(record.payload))
        return payloads

    def delete_by_document(self, tenant_id: str, source_filename: str) -> None:
        self._client.delete(
            collection_name=self._collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[
                        qmodels.FieldCondition(
                            key="metadata.tenant_id",
                            match=qmodels.MatchValue(value=tenant_id),
                        ),
                        qmodels.FieldCondition(
                            key="metadata.source_filename",
                            match=qmodels.MatchValue(value=source_filename),
                        ),
                    ]
                )
            ),
        )

    @staticmethod
    def new_point_id() -> str:
        return str(uuid.uuid4())
