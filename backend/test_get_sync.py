from shared.sync_state import get_sync_status
import json

def run():
    print(json.dumps(get_sync_status(), indent=2))

run()
