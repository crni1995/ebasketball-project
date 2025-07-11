from flask import Flask, render_template, request, url_for
from fixtures_cache import get_fixtures_and_results, refresh_fixtures_and_results
from datetime import datetime, timedelta
from ml_odds import predict_ml_odds_for_upcoming
from player_profile import get_player_profile_data
from collections import defaultdict

from flask_httpauth import HTTPBasicAuth
app = Flask(__name__)
auth = HTTPBasicAuth()

# ========== AUTH SECTION ==========
users = {
    "admin": "admin"  # Change this!
}

@auth.get_password
def get_pw(username):
    if username in users:
        return users.get(username)
    return None

# ==== Blacklists ====
PLAYER_NAME_BLACKLIST = [
    "Cyber Sharks", "Cyber esport", "L&A esport", "eAdriatic esport"
]
CLUB_NAME_BLACKLIST = [
    "Berlin", "Dublin", "Mumbai", "Bangkok", "London", "Doha", "Dakar", "Tokyo",
    "Salvador", "Manila", "Amsterdam", "Ottawa", "Stockholm", "Cairo", "Rio",
    "Brooklyn", "Helsinki", "Sevilla", "Krakow", "Dubai"
]

# ==== Fixtures/Results Helpers (cache logic removed, now using fixtures_cache) ====

def get_all_fixtures():
    return get_fixtures_and_results()

def filter_by_day(fixtures, mode, played=True):
    today = datetime.now().date()
    if mode == "today":
        target = today
        return [f for f in fixtures if f['date'] == target and f['played'] == played]
    elif mode == "yesterday":
        target = today - timedelta(days=1)
        return [f for f in fixtures if f['date'] == target and f['played'] == played]
    elif mode == "this_week":
        start = today - timedelta(days=today.weekday())
        end = start + timedelta(days=6)
        return [f for f in fixtures if start <= f['date'] <= end and f['played'] == played]
    else:
        return [f for f in fixtures if f['played'] == played]

def compute_standings(fixtures, key="player"):
    stats = defaultdict(lambda: {
        "games": 0, "wins": 0, "losses": 0, "draws": 0,
        "points_for": 0, "points_against": 0,
    })
    for f in fixtures:
        if not f["played"]:
            continue
        home = f["home_player"] if key == "player" else f["home_club"]
        away = f["away_player"] if key == "player" else f["away_club"]

        try:
            hs = int(f["home_score"])
            as_ = int(f["away_score"])
        except (TypeError, ValueError):
            continue

        if home:
            stats[home]["games"] += 1
            stats[home]["points_for"] += hs
            stats[home]["points_against"] += as_
            if hs > as_:
                stats[home]["wins"] += 1
            elif hs < as_:
                stats[home]["losses"] += 1
            else:
                stats[home]["draws"] += 1
        if away:
            stats[away]["games"] += 1
            stats[away]["points_for"] += as_
            stats[away]["points_against"] += hs
            if as_ > hs:
                stats[away]["wins"] += 1
            elif as_ < hs:
                stats[away]["losses"] += 1
            else:
                stats[away]["draws"] += 1

    standings = []
    for k, v in stats.items():
        winrate = v["wins"] / v["games"] if v["games"] else 0
        avg_for = v["points_for"] / v["games"] if v["games"] else 0
        avg_against = v["points_against"] / v["games"] if v["games"] else 0
        standings.append({
            "name": k,
            **v,
            "winrate": winrate,
            "avg_for": avg_for,
            "avg_against": avg_against
        })
    standings.sort(key=lambda x: (x["wins"], x["winrate"], x["avg_for"]), reverse=True)
    return standings

# ==== ROUTES ==== 

@app.route("/")
@auth.login_required
def index():
    fixtures = get_all_fixtures()
    results_day = request.args.get('results_day', 'all')
    fixtures_day = request.args.get('fixtures_day', 'all')
    results = filter_by_day(fixtures, results_day, played=True)
    upcoming = filter_by_day(fixtures, fixtures_day, played=False)
    return render_template(
        "index.html",
        results=sorted(results, key=lambda x: (x['date'], x['time']), reverse=True)[:50],
        upcoming=sorted(upcoming, key=lambda x: (x['date'], x['time']))[:50],
        results_day=results_day,
        fixtures_day=fixtures_day
    )

@app.route("/standings")
@auth.login_required
def standings():
    fixtures = get_all_fixtures()
    player_standings = compute_standings(fixtures, key="player")
    club_standings = compute_standings(fixtures, key="club")

    player_blacklist_lower = [name.lower() for name in PLAYER_NAME_BLACKLIST]
    club_blacklist_lower = [name.lower() for name in CLUB_NAME_BLACKLIST]

    player_standings = [
        p for p in player_standings
        if p["name"] and p["name"].lower() not in player_blacklist_lower 
    ]
    club_standings = [
        c for c in club_standings
        if c["name"].lower() not in club_blacklist_lower
    ]

    return render_template(
        "standings.html",
        player_standings=player_standings,
        club_standings=club_standings
    )

@app.route("/odds")
@auth.login_required
def odds_dashboard():
    fixtures = get_all_fixtures()
    today = datetime.now().date()

    print("\n[DEBUG] Total fixtures loaded:", len(fixtures))
    upcoming_fixtures = [f for f in fixtures if not f['played']]
    print(f"[DEBUG] Not played (upcoming): {len(upcoming_fixtures)}")
    for f in upcoming_fixtures[:10]:
        print(f"[DEBUG] UPCOMING: {f['date']} {f['time']} | {f['home_club']} ({f['home_player']}) vs {f['away_club']} ({f['away_player']})")

    upcoming = [
        f for f in fixtures if not f['played'] and f['date'] >= today
    ]
    print(f"[DEBUG] Upcoming after date filter: {len(upcoming)}")
    for f in upcoming[:10]:
        print(f"[DEBUG] FILTERED: {f['date']} {f['time']} | {f['home_club']} ({f['home_player']}) vs {f['away_club']} ({f['away_player']})")

    upcoming = predict_ml_odds_for_upcoming(fixtures, upcoming)
    return render_template(
        "odds.html",
        upcoming=upcoming
    )

@app.route("/backtest")
@auth.login_required
def backtest():
    fixtures = get_all_fixtures()
    played = [f for f in fixtures if f["played"] and f.get("home_score") is not None and f.get("away_score") is not None]
    played_sorted = sorted(played, key=lambda x: (x["date"], x["time"]))
    last_100 = played_sorted[-100:]

    preds = predict_ml_odds_for_upcoming(
        fixtures,
        last_100,
        n_recent=2000,
        recency_decay=0.97,
        old_decay=0.85
    )

    from sklearn.metrics import mean_absolute_error
    actual_totals = [int(m['home_score']) + int(m['away_score']) for m in last_100]
    pred_totals = [p['total_points_pred'] for p in preds]
    actual_margins = [int(m['home_score']) - int(m['away_score']) for m in last_100]
    pred_margins = [float(p['handicap']) for p in preds]
    winner_acc = sum((a > 0) == (p > 0) for a, p in zip(actual_margins, pred_margins)) / len(actual_margins)

    over_count = sum(1 for a, p in zip(actual_totals, pred_totals) if a > p)
    under_count = sum(1 for a, p in zip(actual_totals, pred_totals) if a < p)
    push_count = sum(1 for a, p in zip(actual_totals, pred_totals) if abs(a - p) < 0.01)
    total = len(actual_totals)
    over_pct = 100 * over_count / total
    under_pct = 100 * under_count / total
    push_pct = 100 * push_count / total

    stats = {
        "mae_total": round(mean_absolute_error(actual_totals, pred_totals), 2),
        "mae_margin": round(mean_absolute_error(actual_margins, pred_margins), 2),
        "winner_acc": f"{winner_acc*100:.1f}%",
        "over_pct": f"{over_pct:.1f}%",
        "under_pct": f"{under_pct:.1f}%",
        "push_pct": f"{push_pct:.1f}%"
    }

    matchups = []
    for m, p in zip(last_100, preds):
        actual_total = int(m["home_score"]) + int(m["away_score"])
        pred_total = p.get("total_points_pred", 0)
        if actual_total > pred_total:
            ou_result = "Over"
        elif actual_total < pred_total:
            ou_result = "Under"
        else:
            ou_result = "Push"
        matchup = {
            "date": m["date"],
            "time": m["time"],
            "home_player": m.get("home_player", ""),
            "away_player": m.get("away_player", ""),
            "home_score": m["home_score"],
            "away_score": m["away_score"],
            "winner_odds": p.get("winner_odds", "-"),
            "total_points_pred": p.get("total_points_pred", "-"),
            "handicap": p.get("handicap", "-"),
            "reasoning": p.get("reasoning", "-"),
            "actual_total": actual_total,
            "ou_result": ou_result
        }
        matchups.append(matchup)

    matchups = matchups[::-1]

    return render_template("backtest.html", matchups=matchups, stats=stats)

@app.route("/player/<player_name>")
@auth.login_required
def player_profile(player_name):
    fixtures = get_all_fixtures()
    profile = get_player_profile_data(player_name, fixtures)
    return render_template("player_profile.html", profile=profile)

@app.route("/refresh")
def refresh():
    refresh_fixtures_and_results()
    return "<h2>Cache refreshed! Go back to <a href='/'>Home</a></h2>"

if __name__ == "__main__":
    app.run(debug=True)
