# REST API guide (frontend integration)

How to drive this RAG MVP from a browser or SPA. Controllers live in
`app/api/routes/`; request and response bodies are the Pydantic DTOs in
`app/models/api.py` (plus enums in `app/models/enums.py` and chunk metadata in
`app/models/metadata.py`).

The API is a pure HTTP surface. Streamlit was only ever a client; any new frontend
should call these endpoints the same way. Interactive OpenAPI is at
`http://<host>:8000/docs` when the API is running.

Related docs: [architecture.md](architecture.md),
[ingestion_pipeline.md](ingestion_pipeline.md),
[rag_strategy.md](rag_strategy.md), [answering.md](answering.md),
[operations.md](operations.md).

---

## Base URL and transport

| Item | Value |
|------|--------|
| Default local API | `http://localhost:8000` |
| Content types | JSON for `/query` and metadata patch; `multipart/form-data` for `POST /ingest` |
| Auth | None yet — see [Multi-tenancy](#multi-tenancy-and-security) |
| CORS | Not configured on the API today. A browser origin that is not the API host needs backend CORS (or a same-origin proxy) before `fetch` works |

Routers mounted from `app/main.py`:

| Prefix / path | Module | Purpose |
|---------------|--------|---------|
| `GET /health` | `health.py` | Liveness of Qdrant + MinIO |
| `/ingest` | `ingest.py` | Upload + job polling + metadata resume |
| `/query` | `query.py` | Ask a question, get answer + sources + trace |
| `/sources` | `sources.py` | Presigned MinIO URL for a cited file |

There is no session store on the server. Chat history, tenant selection, and upload
job lists belong in the client.

---

## Multi-tenancy and security

Every mutating or reading call that touches documents takes a `tenant_id`
(form field, JSON body, or query param). Chunks are filtered in Qdrant by
`metadata.tenant_id`; object keys are `{tenant_id}/{document_id}/{filename}` and
`GET /sources/url` rejects keys that do not start with `{tenant_id}/`.

**There is no authentication.** A caller who sends another tenant id can read that
tenant’s index and request its files. Treat `tenant_id` as a hard product input
you control (config, login claim later), not as something the end user invents in
a text box in production.

Presigned MinIO URLs are bearer capabilities until they expire
(`PRESIGNED_URL_TTL_SECONDS`, default 900). Anyone with the link can `GET` the
file; do not log or cache them in long-lived client storage.

---

## Shared enums and metadata DTOs

Use these string values exactly (they are `StrEnum`s). Invalid values fail
validation with FastAPI’s usual 422 body.

### `DocType` (`app/models/enums.py`)

`agreement` · `financial_report` · `minutes` · `org_chart` · `accounting_data` ·
`presentation` · `report_analysis` · `policy_regulation` · `transaction_doc` ·
`other`

`accounting_data` is for SIE ledgers (`.sie` / `.si` / `.se`), not a category.

### `Category`

`hr` · `financing` · `legal` · `operations` · `compliance_governance` ·
`m_and_a` · `strategy_market` · `other`

### `Language`

`sv` · `en` · `dk` · `no` · `fi` · `de` · `fr` · `other`

Danish is `dk`, not `da`.

### `ChunkMetadata` (on every source chunk)

Returned inside `QueryResponse.sources[].metadata`. Required fields after ingest:

| Field | Type | Notes |
|-------|------|--------|
| `tenant_id` | string | Isolation key |
| `company_name` | string | Exact match filter when set on query |
| `doc_type` | DocType | |
| `category` | Category | |
| `year` | int | |
| `quarter` | int \| null | 1–4 when set |
| `source_filename` | string | Display name |
| `object_key` | string | Pass to `GET /sources/url` |
| `content_hash` | string \| null | Optional on older chunks |
| `figure_object_key` | string \| null | Set only on figure chunks; **do not** open this for “View source” — open parent `object_key` |
| `page_number` | int | For PDF deep-links (`#page=N`) |
| `chunk_id` | string | UUID |
| `previous_chunk_id` / `next_chunk_id` | string \| null | Neighbour links |
| `language` | Language | |
| `processed_timestamp` | ISO datetime | |

---

## `GET /health`

**Response DTO:** `HealthResponse`

```json
{
  "status": "ok",
  "qdrant": "ok",
  "minio": "ok"
}
```

- `status` is `"ok"` only when both backends ping successfully; otherwise
  `"degraded"`.
- `qdrant` / `minio` are `"ok"` or `"unavailable"`.
- Use for a settings/status strip. A failed request means the API process is down,
  not only a backend.

---

## Ingest

Entry: `POST /ingest` → `202` + `job_id`. A lifespan worker drains Redis and runs
parse → chunk → embed → Qdrant off the event loop. Poll until a terminal status.
Details of what the worker does: [ingestion_pipeline.md](ingestion_pipeline.md).

### Status machine

```
queued → running → completed
                 → duplicate
                 → failed
                 → needs_metadata ──PATCH metadata──▶ queued → …
```

| Status | Client action |
|--------|----------------|
| `queued` | Keep polling; show waiting |
| `running` | Keep polling; use optional `stage` for progress copy |
| `needs_metadata` | Collect fields in `missing_fields`, then `PATCH …/metadata` |
| `completed` | Done; document is searchable |
| `duplicate` | Done; same bytes already indexed for this tenant (no new chunks) |
| `failed` | Show `error`; MinIO upload for that attempt is rolled back (except metadata pauses) |

**Stages** while `running` (`IngestJobStage`): `checking` · `uploading` ·
`parsing` · `embedding` · `indexing`.

**Terminal statuses:** `completed`, `failed`, `duplicate`. Treat
`needs_metadata` as paused, not finished.

Job payloads live in API process memory; Redis only holds job ids. An API restart
drops in-flight job state — re-upload. There is no cancel or automatic retry.

Suggested poll interval: ~1–2 s while status is `queued` or `running`. Use a long
HTTP timeout on the initial upload (large PDFs / OCR batches).

### `POST /ingest` — accept upload

**Content-Type:** `multipart/form-data`  
**Status:** `202 Accepted`  
**Response DTO:** `IngestJobAccepted`

| Form field | Required | Type | Notes |
|------------|----------|------|--------|
| `file` | yes | file | Original bytes |
| `tenant_id` | yes | string | |
| `company_name` | no | string | Explicit value wins over OCR |
| `doc_type` | no | DocType | Omit for Auto |
| `category` | no | Category | Omit for Auto (SIE usually stays Auto) |
| `year` | no | int | |
| `quarter` | no | int 1–4 | |
| `language` | no | Language | Omit for Auto (`sv` default for SIE) |

Omit empty optional fields rather than sending `""` / `"null"`.

```json
{ "job_id": "…", "status": "queued" }
```

Supported uploads include PDF, Office, images, `.txt` / `.md`, SIE
(`.sie` / `.si` / `.se`), Excel (`.xlsx` / `.xlsm`), and Bolagsverket `.xhtml`
(needs Chrome/Edge on the API host). Format behaviour:
[document_formats.md](document_formats.md).

### `GET /ingest/jobs/{job_id}` — poll

**Response DTO:** `IngestJobStatusResponse`  
**404** if the job id is unknown (including after process restart).

```json
{
  "job_id": "…",
  "status": "running",
  "stage": "parsing",
  "document_id": "",
  "chunk_count": 0,
  "object_key": "",
  "filename": "q3_report_2024.pdf",
  "missing_fields": [],
  "error": null
}
```

| Field | When useful |
|-------|-------------|
| `filename` | Always — label the upload in the UI |
| `stage` | While `running` |
| `document_id`, `object_key`, `chunk_count` | On `completed` / `duplicate` (and partially on `needs_metadata`) |
| `missing_fields` | On `needs_metadata` — e.g. `["company_name", "year"]` |
| `error` | On `failed` or human-readable text on `needs_metadata` |

### `PATCH /ingest/jobs/{job_id}/metadata` — resume after Auto gaps

**Body DTO:** `IngestMetadataPatch` (JSON)  
**Response:** `IngestJobStatusResponse` (typically `status: "queued"` again)

```json
{
  "company_name": "SME Corp AB",
  "doc_type": "financial_report",
  "category": "financing",
  "year": 2024,
  "quarter": 3,
  "language": "sv"
}
```

Only fields present in the JSON are merged (`exclude_unset`). Send at least the
fields listed in `missing_fields`.

| HTTP | Meaning |
|------|---------|
| 404 | Unknown job |
| 409 | Job is not in `needs_metadata` |

The MinIO object is already kept when the job paused; the client does **not**
re-upload the file.

### Ingest UX checklist

1. `POST /ingest` per file (batch = many posts + parallel polls).
2. Poll until terminal or `needs_metadata`.
3. On `needs_metadata`, show a form driven by `missing_fields`, then PATCH and
   keep polling.
4. On `duplicate`, tell the user the file was already indexed (same content hash
   for that tenant) — do not treat as failure.
5. Do not expect metadata-only correction of an already indexed document through
   `/ingest`; force re-parse requires deleting existing chunks server-side first.

---

## Query

### `POST /query`

**Content-Type:** `application/json`  
**Request DTO:** `QueryRequest`  
**Response DTO:** `QueryResponse`

Synchronous and can be slow (embed → retrieve → optional Cohere → optional vision →
reasoning model). Use a generous client timeout (on the order of minutes for hard
questions with figures).

#### Request body

```json
{
  "question": "What was Q3 revenue?",
  "tenant_id": "default_tenant_test_corp_123",
  "company_name": null,
  "doc_type": "financial_report",
  "category": "financing",
  "year": 2024,
  "quarter": 3,
  "top_k": 5,
  "expand_neighbors": false,
  "rerank": true,
  "chat_history": [
    { "role": "user", "content": "Tell me about INOD Q4" },
    { "role": "assistant", "content": "…" }
  ]
}
```

| Field | Default | Role |
|-------|---------|------|
| `question` | required | Current user turn |
| `tenant_id` | required | Hard Qdrant filter |
| `company_name` | null | Optional **exact** metadata filter; leave null to let retrieval infer from text / history |
| `doc_type`, `category`, `year`, `quarter` | null | Optional hard filters (`must` clauses) |
| `top_k` | 5 | 1–20 final chunks for the answer |
| `expand_neighbors` | false | Pull prev/next chunks into context after ranking |
| `rerank` | true | Ask for Cohere when configured; fails open if off/broken |
| `chat_history` | `[]` | Prior turns; **not** stored server-side |

#### `ChatTurn`

```json
{ "role": "user" | "assistant", "content": "…" }
```

Client responsibilities:

- Persist history in the browser/session only.
- For assistant turns, send **answer text only** (`QueryResponse.answer`), not the
  full response object or the retrieval trace.
- On “clear chat” or tab reload, send `chat_history: []` (or omit it).
- Follow-ups like “are they well positioned?” need prior **user** turns so
  company scope and `trace.retrieval_question` still see names like INOD. See
  [rag_strategy.md](rag_strategy.md) and [ranking.md](ranking.md).

#### Response body

```json
{
  "answer": "Markdown-ish answer with citations…",
  "sources": [
    {
      "raw_text": "…chunk markdown…",
      "metadata": { "…ChunkMetadata…" },
      "score": 0.81,
      "rerank_score": 0.96
    }
  ],
  "trace": { "…RetrievalTrace…" }
}
```

| Part | Use in UI |
|------|-----------|
| `answer` | Primary assistant message |
| `sources` | Citation list: filename, page, chunk preview; open file via `/sources/url` |
| `trace` | Optional “agent reasoning” / debug panel |

`score` is vector similarity; `rerank_score` is Cohere when applied. They are
different scales — do not threshold them against each other
([rag_strategy.md](rag_strategy.md)).

#### `RetrievalTrace` (transparency panel)

Useful fields for a developer or power-user panel:

| Field | Meaning |
|-------|---------|
| `embedding_model` / `answer_model` / `vision_model` | Models used |
| `filters` | Effective filters applied |
| `top_k` | Requested top_k |
| `retrieval_question` | Text actually embedded / BM25’d (history + current) |
| `hits[]` | Candidate table: ranks, scores, preview |
| `fusion.method` | `"cohere"` or `"rrf"` — final order rule |
| `rerank` | Whether Cohere ran; `fallback_reason` if not |
| `neighbor_chunk_ids` | Extra context when expand is on |
| `context_chunk_count` | Chunks sent to the reasoner |
| `retrieval_ms` / `generation_ms` / `vision_ms` | Timings |
| `reasoning_summary` | May be empty for trivial lookups — not an error |
| `figure_readings[]` | Vision pass results / errors |

Hits-table score columns (`vector_rank`, `vector_score` via `score`,
`lexical_rank`, `rrf_score`, `rerank_score`) are documented in
[rag_strategy.md](rag_strategy.md). When `fusion.method === "cohere"`, row order
follows Cohere; BM25/RRF are informational.

---

## Sources (open cited files)

File bytes never return through the API. The frontend asks for a short-lived URL;
the browser fetches MinIO directly.

### `GET /sources/url`

**Query params**

| Param | Required | Notes |
|-------|----------|--------|
| `object_key` | yes | From `sources[].metadata.object_key` |
| `tenant_id` | yes | Must be a prefix of `object_key` |

**Response DTO:** `SourceUrlResponse`

```json
{
  "url": "https://…presigned…",
  "filename": "q3_report_2024.pdf",
  "disposition": "inline",
  "expires_in_seconds": 900
}
```

| HTTP | Meaning |
|------|---------|
| 403 | `object_key` does not start with `{tenant_id}/` |
| 404 | Object missing in MinIO |

`disposition`:

- `inline` — PDF, plain text, markdown, HTML, images (browser can render).
- `attachment` — everything else (xlsx, docx, SIE, …): offer download.

**Client rules that matter:**

1. Use **`GET`** on the presigned URL (not `HEAD` — signature is method-specific).
2. For PDFs, append `#page={page_number}` **in the browser** after the signed URL.
   The fragment is not part of the signature, so deep-linking does not invalidate it.
3. For figure chunks, still open the parent `object_key` (the original PDF), not
   `figure_object_key`. Crops are for server-side vision only.
4. When the API runs in Docker, MinIO must expose a browser-reachable
   `MINIO_PUBLIC_ENDPOINT`; otherwise the signed host is unreachable from the client.
5. Refresh the URL before `expires_in_seconds` if the user keeps the tab open.

---

## End-to-end flows

### Upload one or many documents

```
Client                         API                         MinIO / Qdrant
  │  POST /ingest (multipart)   │
  │◀──── 202 { job_id } ───────│  enqueue Redis
  │  GET /ingest/jobs/{id}  (poll)
  │◀──── running / stage ──────│──────────▶ upload, parse, embed, upsert
  │  … needs_metadata?          │
  │  PATCH …/metadata           │  re-queue
  │  GET … (poll)               │
  │◀──── completed | duplicate | failed
```

### Ask a question with citations

```
Client                         API                         MinIO
  │  POST /query { question, tenant_id, chat_history, filters… }
  │◀──── { answer, sources, trace }
  │  GET /sources/url?object_key&tenant_id   (per citation the user opens)
  │◀──── { url, disposition, … }
  │  GET url (+ #page=N for PDF) ─────────────────────────▶ file bytes
```

Keep `chat_history` in client state between turns; send it on every `/query`.

---

## Error handling (HTTP)

| Code | Typical cause |
|------|----------------|
| 202 | Ingest accepted (not an error) |
| 403 | Source object key / tenant mismatch |
| 404 | Unknown ingest job or missing MinIO object |
| 409 | Metadata PATCH while job is not `needs_metadata` |
| 422 | Validation (bad enum, `quarter` out of 1–4, missing `tenant_id`, …) |
| 500 | Server/pipeline failure — for ingest, prefer job `failed` + `error` string after 202 |

Ingest failures after accept usually appear as job status `failed` with `error`,
not as a failed `POST /ingest`. Query failures surface as HTTP errors on
`POST /query`.

Operational failure modes (OCR, embeddings, stale chunk schema, etc.):
[operations.md](operations.md).

---

## DTO quick reference (`app/models/api.py`)

| DTO | Direction | Endpoint |
|-----|-----------|----------|
| `HealthResponse` | ← | `GET /health` |
| `IngestJobAccepted` | ← | `POST /ingest` |
| `IngestJobStatusResponse` | ← | `GET/PATCH …/jobs/…` |
| `IngestMetadata` | internal (built from form) | ingest enqueue |
| `IngestMetadataPatch` | → | `PATCH /ingest/jobs/{id}/metadata` |
| `QueryRequest` | → | `POST /query` |
| `ChatTurn` | → (nested) | `POST /query` |
| `QueryResponse` | ← | `POST /query` |
| `SourceChunk` | ← (nested) | `POST /query` |
| `RetrievalTrace` (+ nested traces) | ← (nested) | `POST /query` |
| `SourceUrlResponse` | ← | `GET /sources/url` |
| `ChunkMetadata` | ← (nested on sources) | `app/models/metadata.py` |
| `DocType` / `Category` / `Language` | both | `app/models/enums.py` |

`IngestResponse` exists for the sync ingest result shape inside the worker; the
HTTP surface exposes the same fields via `IngestJobStatusResponse` after
completion.

---

## Practical frontend notes

- **Type generation:** Point OpenAPI codegen at `/openapi.json` if you want
  TypeScript types that track the Pydantic models.
- **Null filters:** Prefer omitting null filter fields or sending JSON `null`; do
  not send the string `"All"`.
- **Company filter vs Auto:** Empty `company_name` lets ranking infer entity scope
  from the question and prior user turns. A filled value is a hard Qdrant filter.
- **Rerank toggle:** `rerank: false` skips Cohere even when configured. When
  Cohere is not configured, the API still answers; inspect `trace.rerank`.
- **No WebSocket / SSE:** Progress is poll-only for ingest; query is one request /
  one response.
- **CORS:** Plan for an API change or a reverse proxy before shipping a separate
  frontend origin.
- **Timeouts:** Upload and query are the long calls; health and job poll and
  source URL can stay short (~10–30 s).
