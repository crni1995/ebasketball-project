from league_api import (
    get_current_season_id,
    get_fixtures_for_season,
    parse_fixture
)
from datetime import datetime

def print_all_fixtures(fixtures):
    print(f"\n{'Date':<12}{'Time':<6}{'Home Club':<22}{'Home Player':<15}{'Score':<7}{'Away Club':<22}{'Away Player':<15}{'Group':<20}{'Status':<15}")
    print("-" * 145)
    for m in fixtures:
        score = (
            f"{m['home_score']} : {m['away_score']}"
            if m['home_score'] is not None and m['away_score'] is not None
            else "vs"
        )
        print(
            f"{str(m['date']):<12}{m['time']:<6}{m['home_club']:<22}{(m['home_player'] or ''):<15}{score:<7}{m['away_club']:<22}{(m['away_player'] or ''):<15}{m['group']:<20}{m['fixture_status']:<15}"
        )

def main():
    print("Fetching all fixtures for current season (all groups)...")
    season_id = get_current_season_id()
    all_raw_fixtures = get_fixtures_for_season(season_id)
    print(f"Total fixtures found: {len(all_raw_fixtures)}")

    # List ALL fixtures with a valid date (no group filtering)
    all_parsed = [
        parse_fixture(f) for f in all_raw_fixtures
        if parse_fixture(f)['date'] is not None
    ]
    all_parsed.sort(key=lambda x: (x['date'], x['time']))

    print_all_fixtures(all_parsed)

if __name__ == "__main__":
    main()
