# results_cache.py

import json
import os
from datetime import datetime
from league_api import get_current_season_id, get_fixtures_for_season, parse_fixture

CACHE_FILE = "results_cache.json"

def load_results_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return []

def save_results_cache(data):
    with open(CACHE_FILE, "w") as f:
        json.dump(data, f, default=str)  # default=str handles date/datetime

def fetch_new_results(last_date, last_time):
    """Fetch results newer than last_date/last_time"""
    season_id = get_current_season_id()
    fixtures = get_fixtures_for_season(season_id)
    parsed = [parse_fixture(f) for f in fixtures]
    results = [f for f in parsed if f["played"] and f["date"] is not None and f["time"] is not None]
    # Only results strictly newer
    newer = [
        f for f in results
        if (str(f["date"]), str(f["time"])) > (str(last_date), str(last_time))
    ]
    return newer

def update_results_cache():
    cache = load_results_cache()
    if cache:
        last = max(cache, key=lambda f: (f["date"], f["time"]))
        last_date, last_time = last["date"], last["time"]
    else:
        last_date, last_time = "2000-01-01", "00:00"

    new_results = fetch_new_results(last_date, last_time)
    if new_results:
        print(f"[ResultsCache] Appending {len(new_results)} new results.")
        cache.extend(new_results)
        # Sort and remove duplicates just in case
        seen = set()
        unique = []
        for f in sorted(cache, key=lambda f: (f["date"], f["time"])):
            k = (f["date"], f["time"], f["home_player"], f["away_player"])
            if k not in seen:
                unique.append(f)
                seen.add(k)
        save_results_cache(unique)
        return unique
    else:
        print("[ResultsCache] No new results found.")
        return cache
