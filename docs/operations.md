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
rasterized to PNG per page before being sent; if this reappears, check that
`DocumentParser._parse_pdf_pages` is still being taken and that no caller passes raw PDF bytes
to `chat_completion_with_image`, which now only handles a single image.

**Ingest job returns `status="duplicate"` and indexes nothing.**
Expected. The same tenant already has chunks with that content hash, so the worker skipped
parsing. To force a re-parse — after changing the parser, say — delete the document's chunks
first; the metadata alone cannot be corrected through `/ingest`.

**The same document appears twice in the sources.**
Chunks stored before content hashing carry `content_hash: null` and never match the duplicate
check, so a re-ingest under a different filename lands alongside them. Re-ingest once under each
name, or delete the hash-less chunks.

**A chunk contains "I'm sorry, I can't help with that."**
A vision refusal was stored as content. `is_refusal` in `DocumentParser` drops short apologetic
pages, so a refusal that gets through is either longer than `REFUSAL_MAX_LENGTH` or phrased
outside `REFUSAL_MARKERS`.

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

**Code changes seem to apply to some requests but not others.**
`uvicorn --reload` swaps modules between requests, so a long ingest started before a reload
finishes on the old code while the next request runs the new. When verifying a change, restart
deliberately and check that the port is not already held by an older server — a second uvicorn
exits with `[Errno 10048]` and every request silently keeps hitting the first one.

**Installing Streamlit changes Starlette's version.**
Streamlit and FastAPI share transitive dependencies. The combination in `requirements.txt` is
verified; after any dependency change, run the test suite before assuming the API still boots.

## Limitations and risks

Deliberate MVP trade-offs, recorded so they are chosen rather than discovered.

**There is no authentication.** Every endpoint is open, and `tenant_id` is taken from the request
body. Qdrant filtering and the object-key prefix check enforce isolation *given* a tenant id,
which protects against accidental cross-tenant reads but not against a caller who simply sends a
different id. This is the single largest gap before anything resembling production.

**Page numbers are only as good as the parse.** They are real now — pages are parsed separately
and the index is carried through to the chunk — but a page whose parse comes back empty is
skipped, so numbering follows the source document rather than the stored chunks. Non-paginated
inputs (images, plain text, org charts) are always page 1.

**Duplicate detection is byte-exact.** The content hash catches the same file uploaded twice
under different names, which is the common and expensive case. It does not catch the same
report re-exported from Word, scanned again, or saved with a different PDF producer — those are
different bytes and are indexed as separate documents.

**Presigned URLs are bearer capabilities.** Anyone holding the link can fetch the file until it
expires, with no further authorization. `PRESIGNED_URL_TTL_SECONDS` (default 900) is the only
control, and shortening it is the cheapest mitigation.

**Ingest jobs are in-memory only.** `POST /ingest` returns `202` immediately and a lifespan
worker runs parse/embed/upsert via `asyncio.to_thread`. A process restart drops queued and
running jobs; re-upload. There is no cancel, retry, or multi-worker queue yet — polling
`GET /ingest/jobs/{job_id}` is the only progress channel.

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
