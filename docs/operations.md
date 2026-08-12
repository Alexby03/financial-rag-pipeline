# Operations

What breaks, why, and what this MVP does not do yet. The pipelines themselves are described in
[ingestion_pipeline.md](ingestion_pipeline.md) and [rag_strategy.md](rag_strategy.md).

## Known failure modes

These have all been hit during development. The symptom is rarely where the cause is.

**`/query` returns 500 with a pydantic `Field required` error.**
A required field was added to `ChunkMetadata` while chunks stored earlier lack it. Payloads are
validated on read, so old chunks fail. Re-ingest the affected documents, or make the field
optional.

**`AttributeError: 'QdrantClient' object has no attribute 'search'`.**
`search` was removed in qdrant-client 1.16. Use `query_points` and read `response.points`.

**Vision parsing answers 400 on a PDF.**
The chat completions endpoint does not accept `application/pdf` as image input. PDFs are
rasterized to PNG per page before being sent; if this reappears, check that the PDF branch in
`DocumentParser` is still being taken.

**Rerank answers 404 `DeploymentNotFound`.**
`COHERE_RERANK_MODEL` does not match the Foundry deployment name. The URL and API key can be
entirely correct and you still get this. Copy the deployment name verbatim from the portal.

**Reranking appears to do nothing.**
It fails open by design. Check `trace.rerank` in the response, or the reasoning panel in the UI,
which shows the fallback reason. Reranking is also inactive whenever `COHERE_API_BASE` or
`COHERE_API_KEY` is blank, and that state is reported as "off" rather than as an error.

**A presigned URL returns 403 on `HEAD`.**
Presigned URLs are signed for a specific HTTP method. Use `GET`.

**A presigned URL is unreachable from the browser.**
The URL is signed against the host in `MINIO_PUBLIC_ENDPOINT`. Once the API runs inside Docker,
its internal endpoint (`minio:9000`) is not resolvable by the browser and the public endpoint
must be set separately.

**Config changes have no effect.**
`get_settings` and the service factories in `app/api/deps.py` are `lru_cache`d, so `.env` is read
once per process. Restart the API.

**Installing Streamlit changes Starlette's version.**
Streamlit and FastAPI share transitive dependencies. The combination in `requirements.txt` is
verified; after any dependency change, run the test suite before assuming the API still boots.

## Limitations and risks

Deliberate MVP trade-offs, recorded so they are chosen rather than discovered.

**There is no authentication.** Every endpoint is open, and `tenant_id` is taken from the request
body. Qdrant filtering and the object-key prefix check enforce isolation *given* a tenant id,
which protects against accidental cross-tenant reads but not against a caller who simply sends a
different id. This is the single largest gap before anything resembling production.

**Page numbers are not real.** Ingest never passes a page number, so every chunk is stored with
`page_number=1` and citations always read "p.1". Parsing produces one Markdown document for the
whole file, so the page boundary is lost before chunking. Fixing this means parsing per page and
carrying the index through to `TextChunk`.

**Presigned URLs are bearer capabilities.** Anyone holding the link can fetch the file until it
expires, with no further authorization. `PRESIGNED_URL_TTL_SECONDS` (default 900) is the only
control, and shortening it is the cheapest mitigation.

**Ingest is synchronous.** A large PDF blocks the request for as long as vision parsing takes,
one image per page. There is no job queue, retry, or progress reporting.

**Chunk text leaves the OpenAI boundary when reranking is on.** Candidate chunks are sent to the
Cohere deployment on Azure. For Financial material this is a data residency question worth answering
explicitly rather than by default.

**The MinIO credentials are development defaults.** `minioadmin`/`minioadmin` is hardcoded in
`docker-compose.yml` and defaulted in `Settings`.

**Retrieval quality is unmeasured.** There is no evaluation set, so changes to chunking,
embedding or reranking are judged by spot checks. The retrieval trace makes individual queries
inspectable, which is not the same as knowing whether a change helped on average.

## Local environment

Qdrant and MinIO come from `docker compose up -d qdrant minio minio-init`. The collection is
created by `python -m app.scripts.init_qdrant`, which is idempotent.

The test suite needs neither the Docker services nor a valid API key: external services are
replaced with fakes and all HTTP is mocked, including the Streamlit test that renders the app
end to end.
