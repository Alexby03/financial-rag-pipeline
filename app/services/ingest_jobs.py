"""In-memory ingest queue drained by a FastAPI lifespan worker.

Jobs are not durable: a process restart drops them. That is an accepted MVP trade-off —
re-upload after a crash. The worker runs blocking ingest off the event loop via
``asyncio.to_thread`` so HTTP stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from uuid import uuid4

from app.models.api import (
    IngestJobStage,
    IngestJobStatus,
    IngestJobStatusResponse,
    IngestMetadata,
)
from app.services.ingest import IngestService

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[IngestJobStage], None]


@dataclass
class IngestJob:
    job_id: str
    file_bytes: bytes
    filename: str
    metadata: IngestMetadata
    status: IngestJobStatus = "queued"
    stage: IngestJobStage | None = None
    document_id: str = ""
    chunk_count: int = 0
    object_key: str = ""
    error: str | None = None


@dataclass
class IngestJobManager:
    """Queue + status store shared by the accept route and the lifespan worker."""

    _jobs: dict[str, IngestJob] = field(default_factory=dict)
    _queue: asyncio.Queue[str] = field(default_factory=asyncio.Queue)

    async def enqueue(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: IngestMetadata,
    ) -> str:
        job_id = str(uuid4())
        self._jobs[job_id] = IngestJob(
            job_id=job_id,
            file_bytes=file_bytes,
            filename=filename,
            metadata=metadata,
        )
        await self._queue.put(job_id)
        return job_id

    def get_status(self, job_id: str) -> IngestJobStatusResponse | None:
        job = self._jobs.get(job_id)
        if job is None:
            return None
        return IngestJobStatusResponse(
            job_id=job.job_id,
            status=job.status,
            stage=job.stage,
            document_id=job.document_id,
            chunk_count=job.chunk_count,
            object_key=job.object_key,
            error=job.error,
        )

    async def run_worker(self, ingest_service: IngestService) -> None:
        while True:
            job_id = await self._queue.get()
            try:
                await self._process_job(job_id, ingest_service)
            finally:
                self._queue.task_done()

    async def _process_job(self, job_id: str, ingest_service: IngestService) -> None:
        job = self._jobs[job_id]
        job.status = "running"

        def on_progress(stage: IngestJobStage) -> None:
            job.stage = stage

        try:
            result = await asyncio.to_thread(
                ingest_service.ingest,
                job.file_bytes,
                job.filename,
                job.metadata,
                on_progress,
            )
            job.status = "duplicate" if result.status == "duplicate" else "completed"
            job.document_id = result.document_id
            job.chunk_count = result.chunk_count
            job.object_key = result.object_key
        except Exception as exc:
            logger.exception("Ingest job %s failed", job_id)
            job.status = "failed"
            job.error = str(exc)
        finally:
            job.stage = None
            # Drop bytes once work is done so long-running demos do not accumulate uploads.
            job.file_bytes = b""
