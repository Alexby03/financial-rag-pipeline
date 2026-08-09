from fastapi import FastAPI

from app.api.routes import health, ingest, query

app = FastAPI(
    title="Financial RAG Pipeline",
    description="Multimodal RAG for Financial documents",
    version="0.1.0",
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
