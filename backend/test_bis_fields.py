from kshiraj.bis_live_ingestion.adapters.bis_client import BISClient
import json
with BISClient() as c:
    results = c.search_standards("IS 1944")
    if results:
        print("Search fields:")
        print(json.dumps(results[0], indent=2))
        enc_id = results[0].get("standardEncId")
        if enc_id:
            detail = c.get_standard_details(str(enc_id))
            print("\nDetail fields:")
            print(json.dumps(detail, indent=2))
