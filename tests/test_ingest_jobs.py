import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_ingest_job_manager
from app.api.routes import ingest as ingest_routes
from app.models.api import IngestMetadata, IngestMetadataPatch, IngestResponse
from app.models.enums import Category, DocType, Language
from app.services.ingest import IngestMetadataError
from app.services.ingest_jobs import IngestJobManager, InMemoryJobQueue, RedisJobQueue


class FakeIngestService:
    def __init__(
        self,
        result: IngestResponse | None = None,
        error: Exception | None = None,
        *,
        metadata_missing: list[str] | None = None,
    ) -> None:
        self.result = result or IngestResponse(
            document_id="doc-1",
            chunk_count=3,
            object_key="tenant_1/doc-1/report.pdf",
            status="completed",
        )
        self.error = error
        self.metadata_missing = metadata_missing
        self.calls: list[tuple[bytes, str, IngestMetadata, str | None]] = []

    def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: IngestMetadata,
        on_progress: Any = None,
        *,
        resume_object_key: str | None = None,
        resume_document_id: str | None = None,
    ) -> IngestResponse:
        self.calls.append((file_bytes, filename, metadata, resume_object_key))
        if on_progress is not None:
            on_progress("checking")
            on_progress("parsing")
        if self.metadata_missing is not None and not _metadata_complete(metadata):
            raise IngestMetadataError(
                self.metadata_missing,
                object_key=resume_object_key or f"tenant_1/doc-pause/{filename}",
                document_id=resume_document_id or "doc-pause",
            )
        if self.error is not None:
            raise self.error
        return self.result


def _metadata_complete(metadata: IngestMetadata) -> bool:
    return all(
        [
            metadata.company_name,
            metadata.doc_type,
            metadata.category,
            metadata.year is not None,
            metadata.language,
        ]
    )


@pytest.fixture
def job_manager() -> IngestJobManager:
    return IngestJobManager(_queue=InMemoryJobQueue())


def test_redis_job_queue_falls_back_to_in_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "redis" or name.startswith("redis."):
            raise ModuleNotFoundError("No module named 'redis'")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    async def run() -> None:
        queue = RedisJobQueue("redis://localhost:6379/0")
        await queue.push("job-123")
        assert await queue.pop() == "job-123"
        await queue.close()

    asyncio.run(run())


def test_worker_reports_completed_status(job_manager: IngestJobManager) -> None:
    service = FakeIngestService()
    metadata = IngestMetadata(
        tenant_id="tenant_1",
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        quarter=3,
        language=Language.SV,
    )

    async def run() -> None:
        job_id = await job_manager.enqueue(b"pdf-bytes", "report.pdf", metadata)
        worker = asyncio.create_task(job_manager.run_worker(service))  # type: ignore[arg-type]
        for _ in range(50):
            status = job_manager.get_status(job_id)
            assert status is not None
            if status.status == "completed":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not complete")
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

        final = job_manager.get_status(job_id)
        assert final is not None
        assert final.status == "completed"
        assert final.chunk_count == 3
        assert final.object_key == "tenant_1/doc-1/report.pdf"
        assert final.filename == "report.pdf"
        assert final.stage is None

    asyncio.run(run())


def test_worker_reports_duplicate_from_ingest_result(job_manager: IngestJobManager) -> None:
    service = FakeIngestService(
        result=IngestResponse(
            document_id="doc-9",
            chunk_count=1,
            object_key="tenant_1/doc-9/report.pdf",
            status="duplicate",
        )
    )
    metadata = IngestMetadata(
        tenant_id="tenant_1",
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        language=Language.SV,
    )

    async def run() -> None:
        job_id = await job_manager.enqueue(b"pdf-bytes", "report.pdf", metadata)
        worker = asyncio.create_task(job_manager.run_worker(service))  # type: ignore[arg-type]
        for _ in range(50):
            status = job_manager.get_status(job_id)
            assert status is not None
            if status.status == "duplicate":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("job did not report duplicate")
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    asyncio.run(run())


def test_worker_pauses_on_metadata_then_resumes(job_manager: IngestJobManager) -> None:
    service = FakeIngestService(metadata_missing=["company_name", "year"])
    metadata = IngestMetadata(tenant_id="tenant_1")

    async def run() -> None:
        first = await job_manager.enqueue(b"pdf-bytes", "notes.txt", metadata)
        second = await job_manager.enqueue(
            b"other",
            "ok.pdf",
            IngestMetadata(
                tenant_id="tenant_1",
                company_name="SME Corp AB",
                doc_type=DocType.FINANCIAL_REPORT,
                category=Category.FINANCING,
                year=2024,
                language=Language.SV,
            ),
        )
        worker = asyncio.create_task(job_manager.run_worker(service))  # type: ignore[arg-type]

        for _ in range(100):
            paused = job_manager.get_status(first)
            other = job_manager.get_status(second)
            assert paused is not None and other is not None
            if paused.status == "needs_metadata" and other.status == "completed":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("expected needs_metadata while the other job completed")

        paused = job_manager.get_status(first)
        assert paused is not None
        assert paused.missing_fields == ["company_name", "year"]
        assert paused.object_key.endswith("notes.txt")
        assert job_manager._jobs[first].file_bytes == b"pdf-bytes"

        await job_manager.submit_metadata(
            first,
            IngestMetadataPatch(
                company_name="SME Corp AB",
                doc_type=DocType.OTHER,
                category=Category.OTHER,
                year=2024,
                language=Language.SV,
            ),
        )

        for _ in range(100):
            status = job_manager.get_status(first)
            assert status is not None
            if status.status == "completed":
                break
            await asyncio.sleep(0.01)
        else:
            pytest.fail("resumed job did not complete")

        assert service.calls[-1][3] is not None
        worker.cancel()
        with pytest.raises(asyncio.CancelledError):
            await worker

    asyncio.run(run())


def test_ingest_endpoint_accepts_and_exposes_job_status() -> None:
    manager = IngestJobManager(_queue=InMemoryJobQueue())
    service = FakeIngestService()

    app = FastAPI()
    app.include_router(ingest_routes.router)
    app.dependency_overrides[get_ingest_job_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("report.pdf", b"%PDF-1.7", "application/pdf")},
            data={
                "tenant_id": "tenant_1",
                "company_name": "SME Corp AB",
                "doc_type": "financial_report",
                "category": "financing",
                "year": "2024",
                "quarter": "3",
                "language": "sv",
            },
        )
        assert response.status_code == 202
        payload = response.json()
        assert payload["status"] == "queued"
        job_id = payload["job_id"]

        queued = client.get(f"/ingest/jobs/{job_id}")
        assert queued.status_code == 200
        assert queued.json()["status"] == "queued"
        assert queued.json()["filename"] == "report.pdf"

        asyncio.run(manager._process_job(job_id, service))  # type: ignore[arg-type]

        done = client.get(f"/ingest/jobs/{job_id}")
        assert done.status_code == 200
        body = done.json()
        assert body["status"] == "completed"
        assert body["chunk_count"] == 3
        assert body["object_key"] == "tenant_1/doc-1/report.pdf"


def test_ingest_endpoint_accepts_omitted_metadata_fields() -> None:
    manager = IngestJobManager(_queue=InMemoryJobQueue())
    app = FastAPI()
    app.include_router(ingest_routes.router)
    app.dependency_overrides[get_ingest_job_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("report.pdf", b"%PDF-1.7", "application/pdf")},
            data={"tenant_id": "tenant_1"},
        )
        assert response.status_code == 202
        job_id = response.json()["job_id"]

    job = manager.get_status(job_id)
    assert job is not None
    stored = manager._jobs[job_id]
    assert stored.metadata.company_name is None
    assert stored.metadata.doc_type is None
    assert stored.metadata.year is None
    assert stored.metadata.language is None


def test_patch_metadata_requeues_needs_metadata_job() -> None:
    manager = IngestJobManager(_queue=InMemoryJobQueue())
    service = FakeIngestService(metadata_missing=["company_name"])
    app = FastAPI()
    app.include_router(ingest_routes.router)
    app.dependency_overrides[get_ingest_job_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"tenant_id": "tenant_1"},
        )
        job_id = response.json()["job_id"]
        asyncio.run(manager._process_job(job_id, service))  # type: ignore[arg-type]

        paused = client.get(f"/ingest/jobs/{job_id}")
        assert paused.json()["status"] == "needs_metadata"

        patched = client.patch(
            f"/ingest/jobs/{job_id}/metadata",
            json={
                "company_name": "SME Corp AB",
                "doc_type": "other",
                "category": "other",
                "year": 2024,
                "language": "sv",
            },
        )
        assert patched.status_code == 200
        assert patched.json()["status"] == "queued"

        asyncio.run(manager._process_job(job_id, service))  # type: ignore[arg-type]
        done = client.get(f"/ingest/jobs/{job_id}")
        assert done.json()["status"] == "completed"


def test_patch_metadata_rejects_non_paused_job() -> None:
    manager = IngestJobManager(_queue=InMemoryJobQueue())
    app = FastAPI()
    app.include_router(ingest_routes.router)
    app.dependency_overrides[get_ingest_job_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.post(
            "/ingest",
            files={"file": ("notes.txt", b"hello", "text/plain")},
            data={"tenant_id": "tenant_1"},
        )
        job_id = response.json()["job_id"]
        patched = client.patch(
            f"/ingest/jobs/{job_id}/metadata",
            json={"company_name": "SME Corp AB"},
        )
        assert patched.status_code == 409


def test_missing_ingest_job_returns_404() -> None:
    manager = IngestJobManager(_queue=InMemoryJobQueue())
    app = FastAPI()
    app.include_router(ingest_routes.router)
    app.dependency_overrides[get_ingest_job_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.get("/ingest/jobs/does-not-exist")
        assert response.status_code == 404
