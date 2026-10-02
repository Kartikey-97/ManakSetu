from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
from kshiraj.bis_live_ingestion.sync import BISSyncService
import asyncio

class DummyStore:
    def list_all(self): return []
    def upsert(self, s): pass

async def test():
    with BISClient() as c:
        service = BISSyncService(standards_store=DummyStore(), client=c)
        for desig in ["IS 16107", "IS 10322 (Part 5/Sec 3)", "IS 10322 (Part 1)", "IS 16106", "IS 1944"]:
            res = await asyncio.to_thread(service.sync_designation, desig)
            print(f"{desig}: errors={res.errors}")

asyncio.run(test())
