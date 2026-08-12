import asyncio
from typing import Any

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.api.deps import get_ingest_job_manager
from app.api.routes import ingest as ingest_routes
from app.models.api import IngestMetadata, IngestResponse
from app.models.enums import Category, DocType
from app.services.ingest_jobs import IngestJobManager


class FakeIngestService:
    def __init__(
        self,
        result: IngestResponse | None = None,
        error: Exception | None = None,
    ) -> None:
        self.result = result or IngestResponse(
            document_id="doc-1",
            chunk_count=3,
            object_key="tenant_1/doc-1/report.pdf",
            status="completed",
        )
        self.error = error
        self.calls: list[tuple[bytes, str, IngestMetadata]] = []

    def ingest(
        self,
        file_bytes: bytes,
        filename: str,
        metadata: IngestMetadata,
        on_progress: Any = None,
    ) -> IngestResponse:
        self.calls.append((file_bytes, filename, metadata))
        if on_progress is not None:
            on_progress("checking")
            on_progress("parsing")
        if self.error is not None:
            raise self.error
        return self.result


@pytest.fixture
def job_manager() -> IngestJobManager:
    return IngestJobManager()


def test_worker_reports_completed_status(job_manager: IngestJobManager) -> None:
    service = FakeIngestService()
    metadata = IngestMetadata(
        tenant_id="tenant_1",
        company_name="SME Corp AB",
        doc_type=DocType.FINANCIAL_REPORT,
        category=Category.FINANCING,
        year=2024,
        quarter=3,
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


def test_ingest_endpoint_accepts_and_exposes_job_status() -> None:
    manager = IngestJobManager()
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
                "category": "Financing",
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

        asyncio.run(manager._process_job(job_id, service))  # type: ignore[arg-type]

        done = client.get(f"/ingest/jobs/{job_id}")
        assert done.status_code == 200
        body = done.json()
        assert body["status"] == "completed"
        assert body["chunk_count"] == 3
        assert body["object_key"] == "tenant_1/doc-1/report.pdf"


def test_missing_ingest_job_returns_404() -> None:
    manager = IngestJobManager()
    app = FastAPI()
    app.include_router(ingest_routes.router)
    app.dependency_overrides[get_ingest_job_manager] = lambda: manager

    with TestClient(app) as client:
        response = client.get("/ingest/jobs/does-not-exist")
        assert response.status_code == 404
