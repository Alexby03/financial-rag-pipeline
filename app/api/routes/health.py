from fastapi import APIRouter, Depends

from app.api.deps import get_minio_storage, get_qdrant_service
from app.models.api import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check(
    qdrant=Depends(get_qdrant_service),
    minio=Depends(get_minio_storage),
) -> HealthResponse:
    qdrant_ok = qdrant.ping()
    minio_ok = minio.ping()
    status = "ok" if qdrant_ok and minio_ok else "degraded"
    return HealthResponse(
        status=status,
        qdrant="ok" if qdrant_ok else "unavailable",
        minio="ok" if minio_ok else "unavailable",
    )
