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
| Frontend | Streamlit |

Qdrant and MinIO run through Docker Compose. The API process hosts an in-memory ingest worker
alongside HTTP; the Streamlit frontend is a pure HTTP client and holds no credentials of its
own.

## Shape

```
                 ┌───────────┐
   upload  ──────▶  FastAPI  ├──queue──▶ lifespan worker
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
runs with reranking switched off. Plain `.txt` / `.md` ingest does not need OCR.

## Multi-tenancy

`tenant_id` is stored on every chunk and enforced as a Qdrant filter on every query, and object
keys are prefixed with it so source lookups can be checked before a presigned URL is issued.

The isolation is only as strong as the tenant id it is given, and the API has no authentication
yet — the id arrives in the request. See [operations.md](operations.md) for that and the other
known gaps.

## Documentation

- [ingestion_pipeline.md](ingestion_pipeline.md) — how documents become searchable
- [rag_strategy.md](rag_strategy.md) — how they are retrieved and answered
- [operations.md](operations.md) — failure modes, limitations and risks
