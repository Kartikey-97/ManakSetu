from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from typing import Literal
from datetime import datetime
import json

from kartikey.persistence.analysis_repository import AnalysisRepository

router = APIRouter(prefix="/analyses", tags=["decisions"])

class FindingDecisionUpdate(BaseModel):
    decision: Literal["accepted", "rejected", "reviewed", "deleted"]

class StandardDecisionUpdate(BaseModel):
    decision: Literal["accepted", "rejected", "reviewed"]

def get_repository():
    # Factory for dependency injection
    from shared.config import settings
    return AnalysisRepository(settings.analysis_database_path)

@router.patch("/{analysis_id}/findings/{finding_id}/decision")
async def update_finding_decision(
    analysis_id: str,
    finding_id: str,
    payload: FindingDecisionUpdate,
    repository: AnalysisRepository = Depends(get_repository)
):
    analysis = await repository.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    finding_found = False
    for finding in analysis.findings:
        if finding.id == finding_id or finding.requirement_id == finding_id:
            finding.officer_decision = payload.decision
            finding.officer_decision_at = datetime.utcnow().isoformat()
            finding.officer_decision_by = "officer"
            finding_found = True
            break
            
    if not finding_found:
        raise HTTPException(status_code=404, detail="Finding not found")
        
    await repository.save(analysis)
    return {"status": "ok"}

@router.patch("/{analysis_id}/standards/{standard_id}/decision")
async def update_standard_decision(
    analysis_id: str,
    standard_id: str,
    payload: StandardDecisionUpdate,
    repository: AnalysisRepository = Depends(get_repository)
):
    analysis = await repository.get(analysis_id)
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
        
    if not getattr(analysis, "standard_decisions", None):
        analysis.standard_decisions = {}
        
    analysis.standard_decisions[standard_id] = {
        "decision": payload.decision,
        "at": datetime.utcnow().isoformat(),
        "by": "officer"
    }
    
    await repository.save(analysis)
    return {"status": "ok"}
