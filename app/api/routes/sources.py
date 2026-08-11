from pathlib import PurePosixPath

from fastapi import APIRouter, Depends, HTTPException, Query

from app.api.deps import get_minio_storage
from app.models.api import SourceUrlResponse
from app.services.minio_client import MinioStorage, is_viewable_in_browser

router = APIRouter(prefix="/sources", tags=["sources"])


@router.get("/url", response_model=SourceUrlResponse)
def get_source_url(
    object_key: str = Query(..., description="Object key from chunk metadata"),
    tenant_id: str = Query(...),
    storage: MinioStorage = Depends(get_minio_storage),
) -> SourceUrlResponse:
    if not object_key.startswith(f"{tenant_id}/"):
        raise HTTPException(status_code=403, detail="Object does not belong to this tenant")
    if not storage.object_exists(object_key):
        raise HTTPException(status_code=404, detail="Source file not found in storage")

    filename = PurePosixPath(object_key).name
    inline = is_viewable_in_browser(filename)
    return SourceUrlResponse(
        url=storage.presigned_url(object_key, filename, inline),
        filename=filename,
        disposition="inline" if inline else "attachment",
        expires_in_seconds=storage.url_ttl_seconds,
    )
