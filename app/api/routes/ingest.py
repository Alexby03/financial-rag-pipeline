from fastapi import APIRouter, Depends, File, Form, UploadFile

from app.api.deps import get_ingest_service
from app.models.api import IngestMetadata, IngestResponse
from app.models.enums import Category, DocType
from app.services.ingest import IngestService

router = APIRouter(prefix="/ingest", tags=["ingest"])


@router.post("", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    tenant_id: str = Form(...),
    company_name: str = Form(...),
    doc_type: DocType = Form(...),
    category: Category = Form(...),
    year: int = Form(...),
    quarter: int | None = Form(default=None),
    language: str = Form(default="sv"),
    ingest_service: IngestService = Depends(get_ingest_service),
) -> IngestResponse:
    file_bytes = await file.read()
    filename = file.filename or "upload.bin"
    metadata = IngestMetadata(
        tenant_id=tenant_id,
        company_name=company_name,
        doc_type=doc_type,
        category=category,
        year=year,
        quarter=quarter,
        language=language,
    )
    return ingest_service.ingest(file_bytes, filename, metadata)
