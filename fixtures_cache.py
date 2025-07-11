# fixtures_cache.py

import pickle
import os
from datetime import datetime, timedelta
from league_api import get_current_season_id, get_fixtures_for_season, parse_fixture

CACHE_FILE = "fixtures_cache.pkl"
CACHE_TTL_MINUTES = 15

def _now():
    return datetime.now()

def load_cache():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "rb") as f:
            cache = pickle.load(f)
        if isinstance(cache, dict) and "data" in cache and "cached_at" in cache:
            cached_at = cache["cached_at"]
            if isinstance(cached_at, str):
                cached_at = datetime.fromisoformat(cached_at)
            if (_now() - cached_at) < timedelta(minutes=CACHE_TTL_MINUTES):
                return cache["data"]
    return None

def save_cache(data):
    with open(CACHE_FILE, "wb") as f:
        pickle.dump({
            "data": data,
            "cached_at": _now().isoformat()
        }, f)

def refresh_fixtures_and_results():
    print("[CACHE] Refreshing fixtures/results from API...")
    season_id = get_current_season_id()
    all_raw_fixtures = get_fixtures_for_season(season_id)
    fixtures = [
        parse_fixture(f) for f in all_raw_fixtures
        if parse_fixture(f)['date'] is not None
    ]
    fixtures.sort(key=lambda x: (x['date'], x['time']))
    save_cache(fixtures)
    return fixtures

def get_fixtures_and_results():
    cached = load_cache()
    if cached is not None:
        return cached
    return refresh_fixtures_and_results()
