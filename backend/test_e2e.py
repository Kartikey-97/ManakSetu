import asyncio
import json
import logging
import pytest
from dotenv import load_dotenv
load_dotenv()

from shared.models import Analysis, InputType, Standard, StandardStatus
from kartikey.orchestration.pipeline import run_analysis_pipeline
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry

logging.basicConfig(level=logging.INFO)

@pytest.mark.asyncio
async def test_e2e():
    print("==================================================")
    print("STEP 10: END-TO-END PIPELINE TEST")
    print("==================================================")
    
    # 1. Initialize Registry
    print("\n[+] Initializing Knowledge Registry and Loading Seed Data...")
    registry = initialize_knowledge_registry()
    
    # Add a few manual standards for the test
    registry.standards_store.add(Standard(is_number="IS 732", title="Code of Practice for Electrical Wiring Installations", year=2019, status=StandardStatus.ACTIVE))
    registry.standards_store.add(Standard(is_number="IS 15885", title="Safety of Lamp Controlgear", year=2012, status=StandardStatus.ACTIVE))
    registry.standards_store.add(Standard(is_number="IS 13450", title="Medical Electrical Equipment - Basic Safety", year=2024, status=StandardStatus.ACTIVE))
            
    print(f"    Loaded {len(registry.standards_store.list_all())} standards.")
    
    # 2. Create Fake Tender
    tender_text = (
        "1. All electrical installations and wiring must strictly follow IS 732. "
        "2. The control gear must be safe as per IS 15885. "
        "3. The medical equipment used should follow the basic safety standards of IS 13450 Part 1."
    )
    
    analysis = Analysis(
        is_number="e2e-test-123",
        input_type=InputType.TEXT,
        raw_text=tender_text,
        status="queued"
    )
    # A local store, not the API's. This exercises the pipeline directly, so it
    # has no business writing into the route module's state — and since the routes
    # now read from SQLite, borrowing their dict would not have worked anyway.
    store = {analysis.id: analysis}

    print("\n[+] Starting AI Pipeline (Extract -> Retrieve -> Analyze -> Enrich)")
    print(f"    Input Text: '{tender_text}'")

    # 3. Run Pipeline
    await run_analysis_pipeline(analysis.id, store)

    # 4. Show Results
    completed_analysis = store[analysis.id]
    print(f"\n[+] Pipeline Finished! Status: {completed_analysis.status.value}")
    
    print("\n[+] EXTRACTED REQUIREMENTS:")
    for req in completed_analysis.requirements:
        print(f"    - [{req.category.value}] {req.text}")
        
    print("\n[+] COMPLIANCE FINDINGS:")
    for f in completed_analysis.findings:
        print(f"    -> Requirement: {f.requirement_id[:8]}... | Verdict: {f.verdict.value.upper()}")
        print(f"       Reason: {f.reason}")
        print(f"       Standards Involved: {f.applicable_standards}")
            
    print("\n==================================================")
    print("E2E TEST COMPLETE")

if __name__ == "__main__":
    asyncio.run(test_e2e())
