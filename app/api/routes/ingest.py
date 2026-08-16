from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile

from app.api.deps import get_ingest_job_manager
from app.models.api import IngestJobAccepted, IngestJobStatusResponse, IngestMetadata
from app.models.enums import Category, DocType, Language
from app.services.ingest_jobs import IngestJobManager

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestJobAccepted, status_code=202)
async def ingest_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    company_name: str | None = Form(default=None),
    doc_type: DocType | None = Form(default=None),
    category: Category | None = Form(default=None),
    year: int | None = Form(default=None),
    quarter: int | None = Form(default=None),
    language: Language | None = Form(default=None),
    jobs: IngestJobManager = Depends(get_ingest_job_manager),
) -> IngestJobAccepted:
    file_bytes = await file.read()
    filename = file.filename or "upload.bin"
    metadata = IngestMetadata(
        tenant_id=tenant_id,
        company_name=company_name or None,
        doc_type=doc_type,
        category=category,
        year=year,
        quarter=quarter,
        language=language,
    )
    job_id = await jobs.enqueue(file_bytes, filename, metadata)
    return IngestJobAccepted(job_id=job_id)


@router.get("/jobs/{job_id}", response_model=IngestJobStatusResponse)
async def get_ingest_job(
    job_id: str,
    jobs: IngestJobManager = Depends(get_ingest_job_manager),
) -> IngestJobStatusResponse:
    status = jobs.get_status(job_id)
    if status is None:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return status
