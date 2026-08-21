# Architecture

An MVP for a multimodal RAG system in the Financial industry.

## Stack

| Concern | Choice |
|---------|--------|
| API | FastAPI |
| Vector database | Qdrant |
| Blob storage | MinIO |
| Document parsing | Mistral OCR 4 on Azure AI Foundry |
| Embeddings | OpenAI `text-embedding-3-small`, 1536 dimensions |
| Reranking | Cohere Rerank via Azure AI Foundry (optional) |
| Answer generation | OpenAI reasoning model through the Responses API |
| Figure reading | OpenAI vision model on stored MinIO crops, text fed back to the reasoner |
| Frontend | Any HTTP client (see [api_guide.md](api_guide.md)); Streamlit was the MVP UI |

Qdrant and MinIO run through Docker Compose, along with Redis for the ingest job
queue. The API process hosts an ingest worker alongside HTTP. Frontends are pure
HTTP clients and hold no MinIO/Qdrant credentials of their own.

## Shape

```
                 ┌───────────┐
   upload  ──────▶  FastAPI  ├──Redis──▶ lifespan worker
                 │  (202)    │              │
                 │           ├──────────────┼──▶ MinIO        (original files + figure crops)
   job status ───▶           │              ├──▶ Qdrant       (chunks + metadata)
   question ─────▶           ├──────────────┼──▶ OpenAI       (embeddings, vision, answers)
                 └─────┬─────┘──────────────┼──▶ Mistral OCR  (ingest parse)
                       │                    └──▶ Cohere       (reranking)
                       │
                 presigned URL
                       │
   browser ◀───────────┴───────────────────────  MinIO        (file bytes, direct)
```

Two properties are worth noting in that picture. File bytes never travel back through the API:
the API hands the browser a short-lived presigned URL and MinIO serves the file directly.
And every external dependency except Qdrant, OpenAI, and Mistral OCR is optional — the system
runs with reranking switched off. Plain `.txt` / `.md` ingest does not need OCR. Swedish SIE
accounting files (`.sie` / `.si` / `.se`) and Excel workbooks (`.xlsx` / `.xlsm`) are parsed
locally into Markdown; the raw file still lands in MinIO. XHTML annual reports need a local
Chrome/Edge binary for print-to-PDF before OCR.

## Multi-tenancy

`tenant_id` is stored on every chunk and enforced as a Qdrant filter on every query, and object
keys are prefixed with it so source lookups can be checked before a presigned URL is issued.

The isolation is only as strong as the tenant id it is given, and the API has no authentication
yet — the id arrives in the request. See [operations.md](operations.md) for that and the other
known gaps.

## Documentation

- [api_guide.md](api_guide.md) — REST endpoints and DTOs for frontend clients
- [ingestion_pipeline.md](ingestion_pipeline.md) — how documents become searchable
- [document_formats.md](document_formats.md) — how each file type becomes Markdown
- [rag_strategy.md](rag_strategy.md) — how a question becomes an answer (map)
- [ranking.md](ranking.md) — recall, BM25, RRF, Cohere, entity scope
- [answering.md](answering.md) — neighbours, vision, generation, sources
- [operations.md](operations.md) — failure modes, limitations and risks
