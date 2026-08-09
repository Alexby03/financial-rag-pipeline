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
```

API docs: http://localhost:8000/docs

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Health check for Qdrant and MinIO |
| POST | `/ingest` | Upload document with metadata |
| POST | `/query` | RAG query with tenant and optional filters |

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
