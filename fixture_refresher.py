# fixtures_refresher.py

import time
from fixtures_cache import refresh_fixtures_and_results

while True:
    refresh_fixtures_and_results()
    print("[fixtures_refresher] Cache updated. Sleeping for 15 minutes...")
    time.sleep(15 * 60)
