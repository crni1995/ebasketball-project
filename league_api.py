import requests
from datetime import datetime
import re

LEAGUE_ID = 969936953
BASE_URL = "https://api.leaguerepublic.com/json"

def get_seasons():
    """Fetch all seasons for the league."""
    url = f"{BASE_URL}/getSeasonsForLeague/{LEAGUE_ID}.json"
    return requests.get(url).json()

def get_current_season_id():
    """Get the seasonID for the current season."""
    seasons = get_seasons()
    for s in seasons:
        if s.get("currentSeason"):
            return s["seasonID"]
    return seasons[0]["seasonID"] if seasons else None

def get_fixtures_for_season(season_id):
    """Fetch ALL fixtures for the season (past and future)."""
    url = f"{BASE_URL}/getFixturesForSeason/{season_id}.json"
    return requests.get(url).json()

def split_club_player(team_str):
    """
    Split a string like 'Los Angeles Lakers (Tokyo)' into ('Los Angeles Lakers', 'Tokyo')
    """
    m = re.match(r"^(.*) \((.*)\)$", team_str or "")
    if m:
        return m.group(1).strip(), m.group(2).strip()
    else:
        return team_str, None

def parse_fixture(fixture):
    # Parse date string: e.g. '20240620 18:30'
    dt = None
    try:
        dt = datetime.strptime(fixture.get("fixtureDate", ""), "%Y%m%d %H:%M")
    except Exception:
        pass

    home_club, home_player = split_club_player(fixture.get("homeTeamName", ""))
    away_club, away_player = split_club_player(fixture.get("roadTeamName", ""))

    return {
        "date": dt.date() if dt else None,
        "time": dt.strftime("%H:%M") if dt else "",
        "group": fixture.get("fixtureGroupDesc", ""),
        "home": fixture.get("homeTeamName", ""),
        "home_club": home_club,
        "home_player": home_player,
        "away": fixture.get("roadTeamName", ""),
        "away_club": away_club,
        "away_player": away_player,
        "home_score": fixture.get("homeScore"),
        "away_score": fixture.get("roadScore"),
        "played": bool(fixture.get("result")),
        "fixture_status": fixture.get("fixtureStatusDesc", ""),
    }
