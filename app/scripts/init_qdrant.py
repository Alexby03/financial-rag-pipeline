"""Initialize Qdrant collection and payload indexes."""

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from app.config import get_settings


def init_collection() -> None:
    settings = get_settings()
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
    collection = settings.qdrant_collection

    if not client.collection_exists(collection):
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(
                size=settings.embedding_dimensions,
                distance=qmodels.Distance.COSINE,
            ),
        )

    index_fields: list[tuple[str, qmodels.PayloadSchemaType]] = [
        ("metadata.tenant_id", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.doc_type", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.category", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.year", qmodels.PayloadSchemaType.INTEGER),
        ("metadata.quarter", qmodels.PayloadSchemaType.INTEGER),
        ("metadata.company_name", qmodels.PayloadSchemaType.KEYWORD),
        ("metadata.source_filename", qmodels.PayloadSchemaType.KEYWORD),
    ]

    for field_name, field_schema in index_fields:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=field_name,
                field_schema=field_schema,
            )
        except Exception as exc:
            if "already exists" not in str(exc).lower():
                raise

    print(f"Collection '{collection}' is ready.")


if __name__ == "__main__":
    init_collection()
