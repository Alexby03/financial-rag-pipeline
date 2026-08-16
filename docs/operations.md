# Operations

What breaks, why, and what this MVP does not do yet. The pipelines themselves are described in
[ingestion_pipeline.md](ingestion_pipeline.md) and [rag_strategy.md](rag_strategy.md).
Parse-specific behaviour (SIE, XHTML, Excel, OCR) is in
[document_formats.md](document_formats.md).

## Known failure modes

These have all been hit during development. The symptom is rarely where the cause is.

**`/query` returns 500 with a pydantic `Field required` error.**
A required field was added to `ChunkMetadata` while chunks stored earlier lack it. Payloads are
validated on read, so old chunks fail. Re-ingest the affected documents, or make the field
optional.

**`AttributeError: 'QdrantClient' object has no attribute 'search'`.**
`search` was removed in qdrant-client 1.16. Use `query_points` and read `response.points`.

**Mistral OCR answers 404 or times out.**
Ingest needs `MISTRAL_OCR_ENDPOINT` and `MISTRAL_OCR_API_KEY`. A host ending in
`services.ai.azure.com` is turned into `/providers/mistral/azure/ocr`; a serverless
`models.ai.azure.com` host becomes `/v1/ocr`. A 404 is almost always the wrong host, or
`MISTRAL_OCR_MODEL` not matching the catalog id (`mistral-ocr-4-0`). Data Zone / SKU
labels such as `DataZoneStandard (Swedencentral)` are not the model name. A timeout means
the document exceeded `MISTRAL_OCR_TIMEOUT_SECONDS` (default 300). The worker marks the job
`failed` with the error string — check `/ingest/jobs/{id}`.

**OCR answers 400 `document_parser_too_many_pages`.**
Foundry caps a single OCR request at 30 pages. PDFs are split into 30-page slices before the
call (`MISTRAL_OCR_MAX_PAGES`). If this reappears, the file is not a PDF we could slice
(Office/PPTX), or slicing failed and the whole file was sent. A failed job deletes the MinIO
object it uploaded; leftovers from before that rollback have to be removed in the console.

**OCR answers 400 with a huge token count on a small `.xlsx`.**
Spreadsheets are not sent to Mistral — they are parsed locally with openpyxl. If you still
see this error, the API process is on an older build; restart after pulling. DOCX/PPTX still
use OCR and can hit the same limit on very large Office packages.

**Ingest job fails with `Could not determine company_name, ...`.**
OCR `document_annotation` did not fill the blanks and the upload form left them empty. Fill
them on upload, or re-try with a document that has a readable title page. Do not expect `.txt`
passthrough to infer metadata. SIE files auto-fill `doc_type=accounting_data`,
`category=financing`, plus `company_name` / `year` from `#FNAMN` / `#RAR` when present.
`accounting_data` is a document type, not a category — leave category on Auto for SIE.

**SIE ingest fails with `Failed to parse SIE file` or `Could not determine year`.**
The bytes are not valid SIE 4, or date tags are missing. Open the file and look for
`#RAR`, `#GEN`, `#OMFATTN`, or `#TAXAR` — not `#PROGRAM … 2011.2.1` (that is software
version). Tab-separated Norstedts exports are supported. SIE 5 XML is not. After changing
the SIE parser, delete existing chunks for that document before re-uploading or the
duplicate hash check will skip the re-parse.

**XHTML ingest fails with `No Chrome/Edge browser found`.**
Bolagsverket `.xhtml` is printed to PDF before OCR. Install Microsoft Edge or Google Chrome
on the API host, or set `CHROME_PATH` / `EDGE_PATH` to the browser binary. Without a
browser the job cannot create page images for Mistral.

**XHTML ingest is slow or times out.**
Each detected page is printed headlessly, then the merged PDF is OCR'd in 30-page slices.
Large iXBRL files embed fonts as base64 and take longer to print — raise
`MISTRAL_OCR_TIMEOUT_SECONDS` if the OCR step is what fails; a print timeout is fixed at
120s per page in `xhtml_pdf`.

**Ingest job returns `status="duplicate"` and indexes nothing.**
Expected. The same tenant already has chunks with that content hash, so the worker skipped
parsing. To force a re-parse — after changing the parser, say — delete the document's chunks
first; the metadata alone cannot be corrected through `/ingest`.

**The same document appears twice in the sources.**
Chunks stored before content hashing carry `content_hash: null` and never match the duplicate
check, so a re-ingest under a different filename lands alongside them. Re-ingest once under each
name, or delete the hash-less chunks.

**The answer ignores a chart that was cited as a source.**
OCR captions describe the figure; they often omit the plotted number. Query-time vision
downloads the crop from MinIO and injects a reading into the reasoning context. If the trace
shows no figure readings, `OPENAI_VISION_MODEL` is blank, the crop is missing, or the figure
chunk was never retrieved. A vision error is on `trace.figure_readings[].error` and the query
still answers from text.

**Figure crop is missing for vision.**
`figure_object_key` is set only when OCR returned `image_base64` for that figure. A detected
region without bytes is skipped rather than storing a broken key. The Open button still uses
the parent PDF.

**Rerank answers 404 `DeploymentNotFound`.**
`COHERE_RERANK_MODEL` does not match the Foundry deployment name. The URL and API key can be
entirely correct and you still get this. Copy the deployment name verbatim from the portal.

**Reranking appears to do nothing.**
It fails open by design. Check `trace.rerank` in the response, or the reasoning panel in the UI,
which shows the fallback reason. Reranking is also inactive whenever `COHERE_API_BASE` or
`COHERE_API_KEY` is blank, and that state is reported as "off" rather than as an error.
When it *is* applied, `trace.fusion.method` is `cohere` and hits are ordered by
`rerank_score`. A high Rerank column with INOD at the bottom of the table means you are
looking at an older build that still fused Cohere rank equally with BM25.

**A pitch-deck chunk shows `Kolumner … 1934 | 1933` that are not on the slide.**
That prefix is the chunker's table-period breadcrumb, not another file leaking in. The
years came from page 2 citing the Securities Exchange Act of 1934 and the Securities
Act of 1933; an older build treated any two 19xx tokens as fiscal columns and stamped
them onto every later page. Re-ingest after the chunker change: `Kolumner` is table-only,
years before 1990 are ignored. See [ingestion_pipeline.md](ingestion_pipeline.md) §3.

**A follow-up in the same chat retrieves the wrong company.**
Retrieval used to embed only the current question, so "berätta om de är väl positionerade…"
dropped INOD and ranked every growing-market deck by BM25/RRF. Prior user turns are now
concatenated onto the question (`trace.retrieval_question`) and entity scope inherits the
company from those turns. If the wrong issuer still wins, the first turn never named a
company that matches `metadata.company_name`, or `chat_history` was not sent (Clear chat /
tab reload). See [rag_strategy.md](rag_strategy.md) §1–3.

**`top_k=5` becomes a single source after entity filtering.**
Hard-drop removes other companies from the first mixed vector pool. An older build then
handed whatever was left to the reasoner — often one INOD chunk. Retrieval now re-queries
Qdrant with that `company_name` so the candidate window fills back up to `top_k`. If you
still see one hit, that company only has one chunk in the index.

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

**Page numbers are only as good as the parse.** They come from OCR page indexes and are carried
through to the chunk — but a page whose markdown comes back empty is skipped, so numbering
follows the source document rather than the stored chunks. Non-paginated inputs (images, plain
text) are always page 1.

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
