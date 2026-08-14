import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.deps import get_ingest_job_manager, get_ingest_service, get_app_settings
from app.api.routes import health, ingest, query, sources


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    settings = get_app_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(levelname)s:%(name)s:%(message)s",
        force=True,
    )
    jobs = get_ingest_job_manager()
    worker = asyncio.create_task(jobs.run_worker(get_ingest_service()))
    try:
        yield
    finally:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass


app = FastAPI(
    title="Financial RAG Pipeline",
    description="Multimodal RAG for Financial documents",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health.router)
app.include_router(ingest.router)
app.include_router(query.router)
app.include_router(sources.router)
