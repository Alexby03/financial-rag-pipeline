"""Redis-backed ingest queue drained by a FastAPI lifespan worker.

Job payloads stay in-process; Redis only holds job ids. A process restart drops pending
jobs — re-upload. Completed documents already live in MinIO and Qdrant. The worker runs
blocking ingest off the event loop via ``asyncio.to_thread`` so HTTP stays responsive.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Protocol
from uuid import uuid4

from app.models.api import (
    IngestJobStage,
    IngestJobStatus,
    IngestJobStatusResponse,
    IngestMetadata,
    IngestMetadataPatch,
)
from app.services.ingest import IngestMetadataError, IngestService

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[IngestJobStage], None]


class JobQueue(Protocol):
    async def push(self, job_id: str) -> None: ...

    async def pop(self) -> str: ...

    async def close(self) -> None: ...


class InMemoryJobQueue:
    """Test / offline stand-in that matches the Redis list interface."""

    def __init__(self) -> None:
        self._queue: asyncio.Queue[str] = asyncio.Queue()

    async def push(self, job_id: str) -> None:
        await self._queue.put(job_id)

    async def pop(self) -> str:
        return await self._queue.get()

    async def close(self) -> None:
        return None


class RedisJobQueue:
    def __init__(self, redis_url: str, key: str = "ingest:jobs") -> None:
        import redis.asyncio as redis_async

        self._redis = redis_async.from_url(redis_url, decode_responses=True)
        self._key = key

    async def push(self, job_id: str) -> None:
        await self._redis.rpush(self._key, job_id)

    async def pop(self) -> str:
        # timeout=0 blocks until an item is available.
        result = await self._redis.blpop(self._key, timeout=0)
        if result is None:
            raise RuntimeError("Redis BLPOP returned unexpectedly")
        _key, job_id = result
        return job_id

    async def close(self) -> None:
        await self._redis.aclose()


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
    missing_fields: list[str] = field(default_factory=list)
    error: str | None = None


@dataclass
class IngestJobManager:
    """Queue + status store shared by the accept route and the lifespan worker."""

    _queue: JobQueue = field(default_factory=InMemoryJobQueue)
    _jobs: dict[str, IngestJob] = field(default_factory=dict)

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
        await self._queue.push(job_id)
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
            filename=job.filename,
            missing_fields=list(job.missing_fields),
            error=job.error,
        )

    async def submit_metadata(self, job_id: str, patch: IngestMetadataPatch) -> IngestJobStatusResponse:
        job = self._jobs.get(job_id)
        if job is None:
            raise KeyError(job_id)
        if job.status != "needs_metadata":
            raise ValueError(f"Job {job_id} is {job.status}, not needs_metadata")

        updates = patch.model_dump(exclude_unset=True)
        merged = job.metadata.model_copy(update=updates)
        job.metadata = merged
        job.status = "queued"
        job.missing_fields = []
        job.error = None
        await self._queue.push(job_id)
        status = self.get_status(job_id)
        assert status is not None
        return status

    async def run_worker(self, ingest_service: IngestService) -> None:
        try:
            while True:
                job_id = await self._queue.pop()
                await self._process_job(job_id, ingest_service)
        finally:
            await self._queue.close()

    async def _process_job(self, job_id: str, ingest_service: IngestService) -> None:
        job = self._jobs[job_id]
        job.status = "running"
        job.error = None
        job.missing_fields = []

        def on_progress(stage: IngestJobStage) -> None:
            job.stage = stage

        resume_object_key = job.object_key or None
        resume_document_id = job.document_id or None

        try:
            result = await asyncio.to_thread(
                ingest_service.ingest,
                job.file_bytes,
                job.filename,
                job.metadata,
                on_progress,
                resume_object_key=resume_object_key,
                resume_document_id=resume_document_id,
            )
            job.status = "duplicate" if result.status == "duplicate" else "completed"
            job.document_id = result.document_id
            job.chunk_count = result.chunk_count
            job.object_key = result.object_key
            job.file_bytes = b""
        except IngestMetadataError as exc:
            logger.info("Ingest job %s needs metadata: %s", job_id, exc.missing)
            job.status = "needs_metadata"
            job.missing_fields = list(exc.missing)
            job.object_key = exc.object_key or job.object_key
            job.document_id = exc.document_id or job.document_id
            job.error = str(exc)
            # Keep file_bytes so resume can re-parse without re-upload from the client.
        except Exception as exc:
            logger.exception("Ingest job %s failed", job_id)
            job.status = "failed"
            job.error = str(exc)
            job.file_bytes = b""
        finally:
            job.stage = None
