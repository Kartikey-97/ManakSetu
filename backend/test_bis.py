from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
with BISClient() as c:
    print("search:", c.search_standards("IS 10322 (PART 5/SEC 3)"))
