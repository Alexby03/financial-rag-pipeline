import hashlib
import mimetypes

from app.models.api import IngestMetadata, IngestResponse
from app.services.chunker import Chunker
from app.services.embedder import Embedder
from app.services.minio_client import MinioStorage
from app.services.parser import DocumentParser
from app.services.qdrant_service import QdrantService


class IngestService:
    def __init__(
        self,
        storage: MinioStorage,
        parser: DocumentParser,
        chunker: Chunker,
        embedder: Embedder,
        qdrant: QdrantService,
    ) -> None:
        self._storage = storage
        self._parser = parser
        self._chunker = chunker
        self._embedder = embedder
        self._qdrant = qdrant

    def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: IngestMetadata,
    ) -> IngestResponse:
        content_hash = hashlib.sha256(file_bytes).hexdigest()

        already_indexed = self._qdrant.find_by_content_hash(metadata.tenant_id, content_hash)
        if already_indexed:
            # Parsing is the expensive step, so identical bytes are never parsed twice, whatever
            # the file happens to be called this time.
            object_key = already_indexed[0].metadata.object_key
            return IngestResponse(
                document_id=self._document_id_from_key(object_key),
                chunk_count=len(already_indexed),
                object_key=object_key,
                status="duplicate",
            )

        document_id = Chunker.new_document_id()
        object_key = f"{metadata.tenant_id}/{document_id}/{filename}"
        content_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"

        self._storage.upload(object_key, file_bytes, content_type)
        parsed = self._parser.parse(file_bytes, filename, metadata.doc_type)

        if parsed.content_format == "json":
            text_chunks = [
                chunk
                for page in parsed.pages
                for chunk in self._chunker.split_json(page.text, page.page_number)
            ]
        else:
            text_chunks = self._chunker.split_pages(parsed.pages)

        payloads = self._chunker.build_payloads(
            text_chunks, metadata, filename, object_key, content_hash
        )

        if payloads:
            self._qdrant.delete_by_document(metadata.tenant_id, filename, content_hash)
            vectors = self._embedder.embed_texts([payload.raw_text for payload in payloads])
            self._qdrant.upsert_chunks(vectors, payloads)

        return IngestResponse(
            document_id=document_id,
            chunk_count=len(payloads),
            object_key=object_key,
            status="completed",
        )

    @staticmethod
    def _document_id_from_key(object_key: str) -> str:
        parts = object_key.split("/")
        return parts[1] if len(parts) > 2 else ""
