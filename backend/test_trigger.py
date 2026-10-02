import urllib.request
import json
req = urllib.request.Request("http://127.0.0.1:8000/api/v1/analyses", headers={"X-API-Key": "sk_standiq_dev_26108"})
try:
    with urllib.request.urlopen(req) as response:
        analyses = json.loads(response.read())
        aid = analyses[0]["analysis_id"]
        print(f"Triggering sync for {aid}...")
        req2 = urllib.request.Request(f"http://127.0.0.1:8000/api/v1/analyses/{aid}/sync-bis", method="POST", headers={"X-API-Key": "sk_standiq_dev_26108", "Content-Type": "application/json"}, data=b"{}")
        with urllib.request.urlopen(req2) as r2:
            print(r2.read().decode())
        import time
        time.sleep(5)
        req3 = urllib.request.Request(f"http://127.0.0.1:8000/api/v1/standards/bis-sync-status?analysis_id={aid}", headers={"X-API-Key": "sk_standiq_dev_26108"})
        with urllib.request.urlopen(req3) as r3:
            print(r3.read().decode())
except Exception as e:
    print(e)
