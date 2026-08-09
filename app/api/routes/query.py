from fastapi import APIRouter, Depends

from app.api.deps import get_rag_service
from app.models.api import QueryRequest, QueryResponse
from app.services.rag import RagService

router = APIRouter(prefix="/query", tags=["query"])


@router.post("", response_model=QueryResponse)
def query_documents(
    request: QueryRequest,
    rag_service: RagService = Depends(get_rag_service),
) -> QueryResponse:
    return rag_service.query(request)
