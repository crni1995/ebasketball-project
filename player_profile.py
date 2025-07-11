def safe_int(val):
    try:
        return int(val)
    except (TypeError, ValueError):
        return 0

def get_player_profile_data(player_name, matches, recent_n=10):
    recent_matches = []
    total_points = 0
    wins = 0
    games = 0
    team_stats = {}  # NBA team -> {games, wins, points}

    for m in matches:
        ht_team = m.get('home_club', '')
        ht_player = m.get('home_player', '')
        at_team = m.get('away_club', '')
        at_player = m.get('away_player', '')
        hs = safe_int(m.get('home_score', 0))
        as_ = safe_int(m.get('away_score', 0))
        date = m.get('date', '')

        player_side = None
        opponent = None
        player_team = None
        player_points = None
        opponent_points = None
        result = None

        if ht_player == player_name:
            player_side = 'home'
            opponent = f"{at_team} ({at_player})"
            player_team = ht_team
            player_points = hs
            opponent_points = as_
            result = 'W' if hs > as_ else 'L'
        elif at_player == player_name:
            player_side = 'away'
            opponent = f"{ht_team} ({ht_player})"
            player_team = at_team
            player_points = as_
            opponent_points = hs
            result = 'W' if as_ > hs else 'L'

        if player_side:
            games += 1
            total_points += player_points
            if result == 'W':
                wins += 1
            # Per-team breakdown
            if player_team not in team_stats:
                team_stats[player_team] = {'games': 0, 'wins': 0, 'points': 0}
            team_stats[player_team]['games'] += 1
            if result == 'W':
                team_stats[player_team]['wins'] += 1
            team_stats[player_team]['points'] += player_points

            # Recent matches
            recent_matches.append({
                'opponent': opponent,
                'player_team': player_team,
                'points': player_points,
                'opponent_points': opponent_points,
                'result': result,
                'date': date
            })

    # Only last N matches, sorted by date descending
    recent_matches = sorted(recent_matches, key=lambda x: x.get('date', ''), reverse=True)[:recent_n]

    profile = {
        'player_name': player_name,
        'recent_matches': recent_matches,
        'winrate': f"{wins}/{games}" if games else "0/0",
        'winrate_pct': f"{(wins/games*100):.1f}%" if games else "0%",
        'total_points': total_points,
        'team_stats': [
            {
                'team': team,
                'games': s['games'],
                'wins': s['wins'],
                'winrate_pct': f"{(s['wins']/s['games']*100):.1f}%",
                'points': s['points'],
            } for team, s in team_stats.items()
        ]
    }
    return profile
