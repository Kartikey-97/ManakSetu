import urllib.request
import json
req = urllib.request.Request("http://127.0.0.1:8000/api/v1/analyses", headers={"X-API-Key": "sk_standiq_dev_26108"})
with urllib.request.urlopen(req) as response:
    analyses = json.loads(response.read())
    for a in analyses[:5]:
        aid = a["analysis_id"]
        req2 = urllib.request.Request(f"http://127.0.0.1:8000/api/v1/standards/bis-sync-status?analysis_id={aid}", headers={"X-API-Key": "sk_standiq_dev_26108"})
        data = json.loads(urllib.request.urlopen(req2).read())
        if data["total_synced"] > 0:
            print(f"Analysis {aid}:")
            print(json.dumps(data, indent=2))
