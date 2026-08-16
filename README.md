# Financial RAG Pipeline

Multimodal RAG MVP for Financial documents. Stack: FastAPI, Qdrant, MinIO, Mistral OCR 4, OpenAI embeddings.

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key
- Azure AI Foundry deployment of Mistral OCR 4 (for PDF/Office/image ingest)

## Quick start

```bash
# 1. Start infrastructure
docker compose up -d qdrant minio minio-init

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY plus MISTRAL_OCR_ENDPOINT / MISTRAL_OCR_API_KEY

# 4. Initialize Qdrant collection
python -m app.scripts.init_qdrant

# 5. Run API
uvicorn app.main:app --reload

# 6. Run the chat frontend (separate terminal)
streamlit run streamlit_app.py
```

API docs: http://localhost:8000/docs
Chat UI: http://localhost:8501

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check for Qdrant and MinIO |
| POST | `/ingest` | Accept a document upload (`202` + `job_id`); work continues in the API worker |
| GET | `/ingest/jobs/{job_id}` | Poll ingest job status (`queued` / `running` / `completed` / `duplicate` / `failed`) |
| POST | `/query` | RAG query with tenant and optional filters |
| GET | `/sources/url` | Presigned MinIO URL for a chunk's source file |

## Reranking (Cohere Rerank on Azure AI Foundry)

Vector search alone ranks by embedding similarity, which buries chunks whose wording differs
from the question. When configured, the query path retrieves
`top_k * RERANK_CANDIDATE_MULTIPLIER` candidates from Qdrant, sends them to Cohere Rerank, and
keeps the `top_k` best by Cohere order (BM25/RRF scores stay on the trace but do not override
the reranker).

Reranking is off until both `COHERE_API_BASE` and `COHERE_API_KEY` are set, so the app runs
unchanged without a deployment. When the endpoint is unreachable, unauthorized, or returns an
unexpected payload, the query falls back to vector+BM25 fusion and the reason is reported in the
retrieval trace instead of failing the request.

`COHERE_API_BASE` accepts any of the shapes Foundry hands out, with or without the trailing
`/rerank`:

| Deployment type | Base URL |
|-----------------|----------|
| Serverless endpoint | `https://<name>.<region>.models.ai.azure.com/v1` |
| Foundry project endpoint | `https://<project>.services.ai.azure.com/providers/cohere` |

A base ending in `/providers/cohere` resolves to `/v2/rerank`; anything else gets `/rerank`
appended unless it already ends there.

For project endpoints `COHERE_RERANK_MODEL` must match the deployment name exactly
(e.g. `Cohere-rerank-v4.0-pro`), otherwise the endpoint returns a 500.

## Frontend

`streamlit_app.py` is a Streamlit chat UI that talks to the API over HTTP.

- **Agent transparency**: every answer carries a retrieval trace showing the Qdrant filters,
  each hit with its similarity score and rerank score, how many neighbouring chunks were pulled
  in through the double-linked list, any figure vision readings, and the reasoning summary from
  the answer model.
- **Session chat**: turns stay in the Streamlit tab (`st.session_state`) and are sent as
  `chat_history` on the next `/query`. Retrieval concatenates recent user turns onto the
  current question so a follow-up like "are they well positioned?" still finds INOD
  rather than every growing-market deck in the tenant. Refreshing the tab or clicking
  Clear chat starts over; nothing is persisted on the server.
- **Clickable sources**: each source resolves through `/sources/url` to a presigned MinIO URL
  for the original file. PDFs open inline on the cited page, spreadsheets and other formats
  download. Figure chunks also open the parent PDF, not the stored crop. The bytes go straight
  from MinIO to the browser rather than being proxied through the API.
- **Upload tab**: accepts a document, then polls the ingest job while a non-dismissible toast
  shows queue/stage progress until indexing finishes.

Point the frontend at another API with `RAG_API_URL`. When the API runs inside Docker, set
`MINIO_PUBLIC_ENDPOINT` to the host the browser can reach, since presigned URLs are signed
against that hostname.

## Ingest example

```bash
# Accept the upload (returns immediately)
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@q3_report_2024.pdf" \
  -F "tenant_id=default_tenant_test_corp_123" \
  -F "company_name=SME Corp AB" \
  -F "doc_type=financial_report" \
  -F "category=financing" \
  -F "year=2024" \
  -F "quarter=3" \
  -F "language=sv"
# → {"job_id":"...","status":"queued"}

# Poll until completed / duplicate / failed
curl "http://localhost:8000/ingest/jobs/<job_id>"
```

Documents are identified by a sha256 of their bytes, not by filename. The worker reports
`status="duplicate"` with the existing chunk count when the tenant already has that content,
without a second OCR pass. Jobs live only in the API process memory.

Company name, document type, category, year, and quarter may be omitted on upload; OCR (or
SIE `#FNAMN` / `#RAR` for `.sie`/`.si`/`.se`) fills them from the document when it can. If
they are still blank after parse, the job fails.

## Query example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Q3 revenue?",
    "tenant_id": "default_tenant_test_corp_123",
    "doc_type": "financial_report",
    "category": "financing",
    "year": 2024,
    "quarter": 3
  }'
```

## Development

```bash
pytest
ruff check .
```

The suite needs neither the Docker services nor a valid API key — external services are faked
and all HTTP is mocked, including the Streamlit test that renders the app end to end.

## Architecture

See [docs/architecture.md](docs/architecture.md) for the stack,
[docs/ingestion_pipeline.md](docs/ingestion_pipeline.md) for how documents become searchable,
[docs/document_formats.md](docs/document_formats.md) for how each file type becomes Markdown,
[docs/rag_strategy.md](docs/rag_strategy.md) for how they are retrieved,
[docs/ranking.md](docs/ranking.md) for scoring,
[docs/answering.md](docs/answering.md) for generation, and
[docs/operations.md](docs/operations.md) for known failure modes and limitations.
