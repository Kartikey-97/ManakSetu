"""
kartikey/api/routes/analyses.py

POST /api/v1/analyses      — create a new analysis job
GET  /api/v1/analyses/{id} — poll status + retrieve results

Design:
  - POST returns 202 Accepted immediately with analysis_id.
    The analysis runs as a FastAPI BackgroundTask.
  - GET returns the current state at any lifecycle stage.
    Frontend polls this every few seconds until status is "completed" or "failed".
  - SQLite (AnalysisRepository) is the single source of truth. Reads go to it
    directly; the pipeline checkpoints to it after every step, which is what makes
    polling work — a GET midway through an analysis sees the partial state the
    pipeline has written so far.

    This replaced a process-global ``_analyses`` dict that was the real store,
    with SQLite only shadowing it. That arrangement had three problems: the dict
    grew for the lifetime of the process and never evicted, it made the API
    correct only under a single uvicorn worker (two workers meant two disjoint
    sets of analyses, and a poll could 404 against the worker that did not run
    the job), and having two sources of truth meant a save failure was invisible
    because reads never touched the database.

Status lifecycle:
  queued → extracting → retrieving → analyzing → enriching → completed
                                                            ↘ partially_completed
                                                            ↘ failed
"""

from __future__ import annotations

from fastapi import APIRouter, BackgroundTasks, HTTPException

from kartikey.analysis.query_quality import assess_query
from kartikey.orchestration.pipeline import run_analysis_pipeline
from kartikey.persistence.analysis_repository import AnalysisRepository
from shared.contracts import AnalysisResponse, CreateAnalysisRequest
from shared.models import Analysis, AnalysisStatus, InputType
from shared.config import settings
from shared.utils import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/analyses", tags=["analyses"])

repository = AnalysisRepository(settings.analysis_database_path)


async def initialize_persistence() -> None:
    """
    Create the schema if this is a first run.

    Nothing is loaded into memory: the routes read from the repository on each
    request, so there is no cache to warm and history survives a restart because
    it was never held in the process to begin with.
    """
    await repository.initialize()


# ===========================================================================
# POST /api/v1/analyses
# ===========================================================================

@router.post("", status_code=202)
async def create_analysis(
    body: CreateAnalysisRequest,
    background_tasks: BackgroundTasks,
) -> dict:
    """
    Create a new procurement analysis.

    Accepts either:
      - { "input_type": "text",     "text": "<tender description>" }
      - { "input_type": "document", "document_id": "<id from /documents/upload>" }

    Returns 202 Accepted with analysis_id.
    Poll GET /api/v1/analyses/{id} for progress and results.

    Errors:
      400 — missing required field for the given input_type
      422 — the text carries no identifiable procurement requirement
    """
    # ----------------------------------------------------------------
    # Validate required fields per input type
    # ----------------------------------------------------------------
    if body.input_type == InputType.TEXT:
        if not body.text or not body.text.strip():
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "MISSING_TEXT",
                    "message": "'text' is required and must not be empty when input_type is 'text'.",
                },
            )

        # Quality gate, before an analysis job exists. Rejecting here rather than
        # inside the pipeline means noise never reaches the LLM extractor,
        # retrieval, the applicability model or Gemini, and the user gets a
        # synchronous answer instead of a job that fails a few seconds later.
        quality = assess_query(body.text)
        if not quality.is_meaningful:
            logger.info(
                "Rejected text input as not analysable (%s): signals=%s",
                quality.signals.get("reason"), quality.signals,
            )
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "NOT_A_PROCUREMENT_REQUIREMENT",
                    "message": quality.message,
                    "signals": quality.signals,
                },
            )

    elif body.input_type == InputType.DOCUMENT:
        if not body.document_id:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "MISSING_DOCUMENT_ID",
                    "message": "'document_id' is required when input_type is 'document'. "
                               "Upload a document first via POST /api/v1/documents/upload.",
                },
            )

    # ----------------------------------------------------------------
    # Create analysis object
    # ----------------------------------------------------------------
    analysis = Analysis(
        input_type=body.input_type,
        raw_text=body.text,
        document_id=body.document_id,
        tender_id=body.tender_id,
        tender_title=body.tender_title,
        status=AnalysisStatus.QUEUED,
        metadata=body.metadata,
    )
    await repository.save(analysis)

    logger.info(
        "Analysis created: id=%s input_type=%s",
        analysis.id,
        analysis.input_type,
    )

    # ----------------------------------------------------------------
    # Dispatch pipeline as a background task
    # BackgroundTasks runs after the response is sent.
    #
    # The pipeline still takes a dict because that is its working handle on the
    # object it mutates through the run — but it is a fresh one holding only this
    # job, not a shared global. repository.save is the checkpoint, so every step
    # transition lands in SQLite where the GET route will read it.
    # ----------------------------------------------------------------
    background_tasks.add_task(
        run_analysis_pipeline, analysis.id, {analysis.id: analysis}, repository.save,
    )

    return {
        "analysis_id": analysis.id,
        "status": analysis.status,
        "message": (
            "Analysis queued. Poll GET /api/v1/analyses/{id} for status and results."
        ),
    }


# ===========================================================================
# GET /api/v1/analyses/{analysis_id}
# ===========================================================================

@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str) -> AnalysisResponse:
    """
    Poll an analysis for status and results.

    Returns the current state at any stage of the pipeline.
    Fields are populated progressively:
      - requirements: available after 'extracting' step (Step 5)
      - standards, findings: available after 'analyzing' step (Step 6)
      - full evidence: available after 'enriching' step (Step 7)

    Errors:
      404 — analysis_id not found
    """
    analysis = await repository.get(analysis_id)
    if not analysis:
        raise HTTPException(
            status_code=404,
            detail={
                "error": "ANALYSIS_NOT_FOUND",
                "message": f"No analysis found with id '{analysis_id}'.",
            },
        )

    # Collect the unique standards this analysis touched, applicable ones first.
    #
    # `cited_standards` is included deliberately. The two lists mean different
    # things: `applicable_standards` are the standards that genuinely govern the
    # requirement, while `cited_standards` holds a standard the tender named that
    # the analysis then rejected — an XLPE cable specified against IS 1554, which
    # covers PVC. Collecting only the first list dropped exactly those from the
    # response, so the single most valuable finding the system produces arrived
    # with no standard attached and the Standards tab rendered empty on precisely
    # the documents worth reviewing. The finding still carries the verdict; this
    # only makes sure the standard it is about is in the payload.
    from kartikey.orchestration.knowledge_registry import get_registry
    store = get_registry().standards_store

    seen_standard_ids: set[str] = set()
    all_standards = []
    
    from kshiraj.bis_live_ingestion.normalizer import normalize_designation
    
    for group in ("applicable_standards", "cited_standards"):
        for finding in analysis.findings:
            stds = getattr(finding, group, None) or []
            
            # Use a while loop or iterate carefully to allow appending
            new_stds_to_append = []
            
            for i, standard in enumerate(stds):
                # Always return the freshest state from in-memory store in case it was synced
                fresh_std = store.get_by_id(standard.id)
                if fresh_std:
                    # Preserve transient ML scores attached to the document finding
                    fresh_std.relevance_score = getattr(standard, 'relevance_score', None)
                    fresh_std.semantic_score = getattr(standard, 'semantic_score', None)
                    fresh_std.text_excerpt = getattr(standard, 'text_excerpt', None)
                    
                    stds[i] = fresh_std
                    standard = fresh_std
                if standard.id not in seen_standard_ids:
                    all_standards.append(standard)
                    seen_standard_ids.add(standard.id)
                    
                # Look for superseding standard in the store
                if standard.superseded_by:
                    norm_target = normalize_designation(standard.superseded_by)
                    superseding = next((s for s in store.list_all() if (s.is_number and s.is_number.replace(' ', '').lower() == standard.superseded_by.replace(' ', '').lower()) or (hasattr(s, 'designation') and s.designation.replace(' ', '').lower() == standard.superseded_by.replace(' ', '').lower())), None)
                    if superseding and superseding.id not in seen_standard_ids:
                        # Inherit the applicability score from the standard it supersedes
                        if hasattr(standard, 'relevance_score') and standard.relevance_score is not None:
                            superseding.relevance_score = standard.relevance_score
                        if hasattr(standard, 'semantic_score') and standard.semantic_score is not None:
                            superseding.semantic_score = standard.semantic_score
                        all_standards.append(superseding)
                        seen_standard_ids.add(superseding.id)
                        # We also attach it to the finding so adapter.ts sees it in findings
                        new_stds_to_append.append(superseding)
            
            if new_stds_to_append:
                # Avoid appending if already present (mutating in-memory cache)
                stds.extend([s for s in new_stds_to_append if s.id not in [existing.id for existing in stds]])

    return AnalysisResponse(
        id=analysis.id,
        status=analysis.status,
        input_type=analysis.input_type,
        tender_id=analysis.tender_id,
        tender_title=analysis.tender_title,
        document_id=analysis.document_id,
        created_at=analysis.created_at.isoformat(),
        updated_at=analysis.updated_at.isoformat(),
        requirements=analysis.requirements,
        total_requirements=analysis.total_requirements,
        standards=all_standards,
        findings=analysis.findings,
        qco_findings=getattr(analysis, "qco_findings", None) or [],
        product_profile=getattr(analysis, "product_profile", None),
        issues_found=analysis.issues_found,
        summary=analysis.summary,
        error_message=analysis.error_message,
        metadata=analysis.metadata,
        analysis_mode=analysis.metadata.get("analysis_mode", "fallback"),
        degraded_reason=analysis.metadata.get("degraded_reason"),
    )


# ===========================================================================
# GET /api/v1/analyses  (list — useful for frontend dashboard)
# ===========================================================================

@router.get("", response_model=list[dict])
async def list_analyses() -> list[dict]:
    """
    List all analyses with their current status.
    Useful for a frontend dashboard showing recent analysis jobs.
    """
    analyses = await repository.list()
    return [
        {
            "analysis_id": a.id,
            "status": a.status,
            "input_type": a.input_type,
            "tender_id": a.tender_id,
            "tender_title": a.tender_title,
            "created_at": a.created_at.isoformat(),
            "updated_at": a.updated_at.isoformat(),
            "total_requirements": a.total_requirements,
            "issues_found": a.issues_found,
            "metadata": a.metadata,
            "summary": a.summary,
            "analysis_mode": a.metadata.get("analysis_mode", "fallback"),
            "degraded_reason": a.metadata.get("degraded_reason"),
        }
        for a in sorted(analyses, key=lambda x: x.created_at, reverse=True)
    ]

@router.post("/{analysis_id}/sync-bis")
async def trigger_manual_bis_sync(analysis_id: str) -> dict:
    """Manually trigger BIS sync for an analysis."""
    import asyncio
    from kartikey.orchestration.knowledge_registry import get_registry
    from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient, BISClientConfig
    from kshiraj.bis_live_ingestion.sync import BISSyncService
    from shared.sync_state import record_sync_result
    from datetime import datetime, timezone

    try:
        analysis_obj = await repository.get(analysis_id)
        if not analysis_obj:
            return {"status": "error", "message": "Analysis not found"}

        # Collect unique base IS numbers from findings (no year, no Amd suffix)
        # Using std.is_number (e.g. "IS 16107") not std.designation ("IS 16107:2023 Amd.1")
        # so that normalize_designation() produces a clean "IS 16107" that matches BIS portal results.
        is_numbers: list[str] = []
        seen: set[str] = set()
        for finding in (analysis_obj.findings or []):
            for std in (finding.applicable_standards or []):
                base = getattr(std, 'base_designation', None) or getattr(std, 'is_number', None)
                if base and base not in seen:
                    seen.add(base)
                    is_numbers.append(base)


        if not is_numbers:
            return {"status": "skipped", "message": "No standards to sync for this analysis"}

        store = get_registry().standards_store

        def do_sync() -> list:
            """Blocking BIS HTTP calls — run off event loop via asyncio.to_thread."""
            new_superseded = []
            config = BISClientConfig(timeout_seconds=15.0, max_retries=1)
            with BISClient(config=config) as client:
                svc = BISSyncService(client=client, standards_store=store)
                for is_num in is_numbers:
                    res = svc.sync_designation(is_num)
                    if res.supersedes:
                        new_superseded.append(res.supersedes)
                    record_sync_result(
                        is_number=is_num,
                        synced_at=datetime.now(timezone.utc),
                        changed=res.changed,
                        errors=res.errors,
                        analysis_id=analysis_id,
                    )
            return new_superseded

        # Bug fix: must await asyncio.to_thread so the coroutine is actually scheduled
        new_superseded_ids = await asyncio.to_thread(do_sync)
        
        if new_superseded_ids:
            from kartikey.orchestration.pipeline import rescore_and_merge
            from kshiraj.bis_live_ingestion.normalizer import normalize_designation
            
            for sup_is in new_superseded_ids:
                # Find it in the store EXACTLY matching the returned supersedes string (ignoring whitespace/case)
                new_std = next((s for s in store.list_all() if (s.is_number and s.is_number.replace(' ', '').lower() == sup_is.replace(' ', '').lower()) or (hasattr(s, 'designation') and s.designation.replace(' ', '').lower() == sup_is.replace(' ', '').lower())), None)
                if new_std:
                    asyncio.create_task(rescore_and_merge(analysis_id, new_std))

        return {"status": "sync_complete", "standards_synced": is_numbers}

    except Exception as e:
        import logging
        logging.getLogger(__name__).warning("Manual BIS sync failed: %s", e)
        return {"status": "error", "message": str(e)}
