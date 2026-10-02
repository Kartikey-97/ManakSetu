from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
import json
with BISClient() as c:
    for item in c.search_standards("IS 16107:2023"):
        print(item.get("standardNumber"))
