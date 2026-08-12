# Financial RAG Pipeline

Multimodal RAG MVP for Financial documents. Stack: FastAPI, Qdrant, MinIO, OpenAI (gpt-4o + embeddings).

## Prerequisites

- Python 3.11+
- Docker and Docker Compose
- OpenAI API key

## Quick start

```bash
# 1. Start infrastructure
docker compose up -d qdrant minio minio-init

# 2. Install dependencies
pip install -e ".[dev]"

# 3. Configure environment
cp .env.example .env
# Edit .env and set OPENAI_API_KEY

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
| POST | `/ingest` | Upload document with metadata |
| POST | `/query` | RAG query with tenant and optional filters |
| GET | `/sources/url` | Presigned MinIO URL for a chunk's source file |

## Reranking (Cohere Rerank on Azure AI Foundry)

Vector search alone ranks by embedding similarity, which buries chunks whose wording differs
from the question. When configured, the query path retrieves
`top_k * RERANK_CANDIDATE_MULTIPLIER` candidates from Qdrant, sends them to Cohere Rerank, and
keeps the `top_k` best.

Reranking is off until both `COHERE_API_BASE` and `COHERE_API_KEY` are set, so the app runs
unchanged without a deployment. When the endpoint is unreachable, unauthorized, or returns an
unexpected payload, the query falls back to the vector order and the reason is reported in the
retrieval trace instead of failing the request.

`COHERE_API_BASE` must include the version segment, since the reranker posts to
`{COHERE_API_BASE}/rerank`:

| Deployment type | Base URL |
|-----------------|----------|
| Serverless endpoint | `https://<name>.<region>.models.ai.azure.com/v1` |
| Foundry project endpoint | `https://<project>.services.ai.azure.com/providers/cohere/v2` |

For project endpoints `COHERE_RERANK_MODEL` must match the deployment name exactly
(e.g. `Cohere-rerank-v4.0-pro`), otherwise the endpoint returns a 500.

## Frontend

`streamlit_app.py` is a Streamlit chat UI that talks to the API over HTTP.

- **Agent transparency**: every answer carries a retrieval trace showing the Qdrant filters,
  each hit with its similarity score and rerank score, how many neighbouring chunks were pulled
  in through the double-linked list, and the reasoning summary from the answer model.
- **Clickable sources**: each source resolves through `/sources/url` to a presigned MinIO URL.
  PDFs and images open inline in a new tab, spreadsheets and other formats download. The bytes
  go straight from MinIO to the browser rather than being proxied through the API.
- **Upload tab**: ingests documents without leaving the UI.

Point the frontend at another API with `RAG_API_URL`. When the API runs inside Docker, set
`MINIO_PUBLIC_ENDPOINT` to the host the browser can reach, since presigned URLs are signed
against that hostname.

## Ingest example

```bash
curl -X POST "http://localhost:8000/ingest" \
  -F "file=@q3_report_2024.pdf" \
  -F "tenant_id=default_tenant_test_corp_123" \
  -F "company_name=SME Corp AB" \
  -F "doc_type=financial_report" \
  -F "category=Financing" \
  -F "year=2024" \
  -F "quarter=3" \
  -F "language=sv"
```

## Query example

```bash
curl -X POST "http://localhost:8000/query" \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What was Q3 revenue?",
    "tenant_id": "default_tenant_test_corp_123",
    "doc_type": "financial_report",
    "category": "Financing",
    "year": 2024,
    "quarter": 3
  }'
```

## Development

```bash
pytest
ruff check .
```

## Architecture

See [docs/architecture.md](docs/architecture.md) and [docs/rag_strategy.md](docs/rag_strategy.md).
