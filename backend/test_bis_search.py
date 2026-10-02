from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
import json
with BISClient() as c:
    print(c.search_standards("IS 10322 (PART 5) (SEC 3)"))
    print(c.search_standards("IS 10322 (PART 5/SEC 3)"))
