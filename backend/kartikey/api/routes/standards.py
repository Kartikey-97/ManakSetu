"""
kartikey/api/routes/standards.py

GET /api/v1/standards/search  — search standards by query
GET /api/v1/standards/{id}    — get a specific standard by ID

These endpoints allow the frontend to provide a manual standard search/lookup
feature for the procurement officer, independent of the automated analysis.

Delegates entirely to Kshiraj's knowledge retrieval service.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from shared.models import Standard
from shared.utils import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/standards", tags=["standards"])


@router.get("", response_model=list[Standard])
async def list_standards(
    offset: int = Query(0, ge=0),
    limit: int = Query(24, ge=1, le=100),
) -> list[Standard]:
    """Return a stable page of the canonical local BIS catalog."""
    from kartikey.orchestration.knowledge_registry import get_registry

    standards = sorted(
        get_registry().standards_store.list_all(),
        key=lambda standard: (standard.is_number, standard.year or 0),
    )
    return standards[offset : offset + limit]


@router.get("/search", response_model=list[Standard])
async def search_standards(
    q: str = Query(..., min_length=2, description="Search query — product description, IS number, etc."),
    limit: int = Query(10, ge=1, le=50),
) -> list[Standard]:
    """
    Search for relevant Indian Standards by natural language query or exact IS number.
    
    This provides a manual search capability for the frontend.
    Delegates to kshiraj/knowledge/retrieval_service.py.
    """
    logger.info("Manual standards search: q='%s' limit=%d", q, limit)
    
    from kartikey.orchestration.knowledge_registry import get_registry
    from kshiraj.knowledge.retrieval_service import RetrievalQuery
    
    registry = get_registry()
    query = RetrievalQuery(query_text=q, top_k=limit, include_evidence=False)
    result = registry.retrieval_service.search_standards(query)
    
    # Return the Standard objects from the CandidateStandards
    return [c.standard for c in result.candidates]





@router.get("/bis-sync-status")
async def bis_sync_status(analysis_id: str | None = Query(None)) -> dict:
    """
    Returns BIS live sync status from memory.
    Pass ?analysis_id=<id> to get status scoped to a specific analysis.
    """
    from shared.sync_state import get_sync_status
    return get_sync_status(analysis_id=analysis_id)

@router.get("/{standard_id}", response_model=Standard)

async def get_standard(standard_id: str) -> Standard:
    """
    Get the full details of a specific standard by its internal ID.
    """
    from kartikey.orchestration.knowledge_registry import get_registry
    registry = get_registry()
    
    standard = registry.standards_store.get_by_id(standard_id)
    if not standard:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "STANDARD_NOT_FOUND",
                "message": f"Cannot find ID '{standard_id}'.",
            },
        )
    return standard

