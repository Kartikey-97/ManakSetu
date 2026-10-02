from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
with BISClient() as c:
    for item in c.search_standards("IS 13450"):
        print(item.get("standardNumber"))
