import asyncio
from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
from kshiraj.bis_live_ingestion.sync import BISSyncService
from kartikey.orchestration.knowledge_registry import initialize_knowledge_registry, get_registry
import sys

async def run():
    initialize_knowledge_registry()
    
    is_numbers = ["IS 16107", "IS 10322 (Part 5/Sec 3)", "IS 10322 (Part 1)"]
    print(f"Syncing: {is_numbers}")
    
    store = get_registry().standards_store
    with BISClient() as client:
        svc = BISSyncService(client=client, standards_store=store)
        for is_num in is_numbers:
            print(f"Syncing {is_num}...")
            res = await asyncio.to_thread(svc.sync_designation, is_num)
            print(f"  errors: {res.errors}")
            
asyncio.run(run())
