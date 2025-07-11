import pandas as pd
import numpy as np
import pickle
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import mean_squared_error
import json

def is_valid_score(val):
    try:
        int(val)
        return True
    except (TypeError, ValueError):
        return False

def compute_player_winrate_and_rank(fixtures):
    from collections import defaultdict
    stats = defaultdict(lambda: {"games": 0, "wins": 0})
    for f in fixtures:
        if not f["played"]:
            continue
        for key, score_key, opp_score_key in [
            ("home_player", "home_score", "away_score"),
            ("away_player", "away_score", "home_score"),
        ]:
            player = f[key]
            if not player: continue
            try:
                pf, pa = int(f[score_key]), int(f[opp_score_key])
            except:
                continue
            stats[player]["games"] += 1
            if pf > pa:
                stats[player]["wins"] += 1
    winrates = {p: (v["wins"] / v["games"] if v["games"] else 0.5) for p, v in stats.items()}
    sorted_players = sorted(winrates.items(), key=lambda x: x[1], reverse=True)
    ranks = {p: i+1 for i, (p, _) in enumerate(sorted_players)}
    return winrates, ranks

def rolling_elo(fixtures, player, k=28, base=1500, cutoff=None, cutoff_time=None):
    history = [
        f for f in fixtures
        if f["played"] and (f["home_player"] == player or f["away_player"] == player)
    ]
    if cutoff:
        history = [
            f for f in history
            if (f["date"], f["time"]) < (cutoff, cutoff_time)
        ]
    history = sorted(history, key=lambda x: (x["date"], x["time"]))
    elo = base
    for match in history:
        hs, as_ = match["home_score"], match["away_score"]
        if not (is_valid_score(hs) and is_valid_score(as_)):
            continue
        opp = match["away_player"] if match["home_player"] == player else match["home_player"]
        opp_elo = base  # Fast/simple; you can cache better for "perfect" elo
        pf, pa = (int(hs), int(as_)) if match["home_player"] == player else (int(as_), int(hs))
        score = 1 if pf > pa else 0.5 if pf == pa else 0
        exp_score = 1 / (1 + 10 ** ((opp_elo - elo) / 400))
        elo += k * (score - exp_score)
    return elo

def get_player_recent_stats_weighted(fixtures, player, n_recent=2000, recency_decay=0.97, old_decay=0.85, cutoff=None, cutoff_time=None):
    history = [
        f for f in fixtures
        if f["played"] and (f["home_player"] == player or f["away_player"] == player)
    ]
    if cutoff:
        history = [
            f for f in history
            if (f["date"], f["time"]) < (cutoff, cutoff_time)
        ]
    history = sorted(history, key=lambda x: (x["date"], x["time"]), reverse=True)
    weighted_wins = weighted_games = weighted_margin = weighted_points = 0.0
    weighted_margin2 = weighted_points2 = 0.0
    big_win = big_loss = 0
    opp_wr_sum = opp_count = 0
    for i, match in enumerate(history[:n_recent*2]):  # double window for weighting
        hs, as_ = match["home_score"], match["away_score"]
        if not (is_valid_score(hs) and is_valid_score(as_)):
            continue
        pf, pa = (int(hs), int(as_)) if match["home_player"] == player else (int(as_), int(hs))
        win = 1 if pf > pa else 0
        margin = pf - pa
        total = pf + pa
        opp = match["away_player"] if match["home_player"] == player else match["home_player"]
        opp_wr, _, _ = get_player_recent_stats_simple(fixtures, opp, n=200, cutoff=match["date"], cutoff_time=match["time"])
        opp_wr_sum += opp_wr
        opp_count += 1
        if i < n_recent:
            weight = recency_decay ** i
        else:
            weight = (recency_decay ** n_recent) * (old_decay ** (i - n_recent))
        weighted_wins += win * weight
        weighted_margin += margin * weight
        weighted_margin2 += (margin ** 2) * weight
        weighted_points += total * weight
        weighted_points2 += (total ** 2) * weight
        weighted_games += weight
        if margin >= 20: big_win += 1
        if margin <= -15: big_loss += 1
    if weighted_games == 0:
        return 0.5, 0, 120, 0, 0, 0, 0, 0
    return (
        weighted_wins / weighted_games,
        weighted_margin / weighted_games,
        weighted_points / weighted_games,
        np.sqrt(weighted_margin2/weighted_games - (weighted_margin/weighted_games)**2),
        np.sqrt(weighted_points2/weighted_games - (weighted_points/weighted_games)**2),
        big_win,
        big_loss,
        opp_wr_sum/opp_count if opp_count else 0.5
    )

def get_player_recent_stats_simple(fixtures, player, n=200, cutoff=None, cutoff_time=None):
    history = [
        f for f in fixtures
        if f["played"] and (f["home_player"] == player or f["away_player"] == player)
    ]
    if cutoff:
        history = [
            f for f in history
            if (f["date"], f["time"]) < (cutoff, cutoff_time)
        ]
    history = sorted(history, key=lambda x: (x["date"], x["time"]), reverse=True)[:n]
    wins = games = 0
    for match in history:
        hs, as_ = match["home_score"], match["away_score"]
        if not (is_valid_score(hs) and is_valid_score(as_)):
            continue
        pf, pa = (int(hs), int(as_)) if match["home_player"] == player else (int(as_), int(hs))
        if pf > pa: wins += 1
        games += 1
    return (wins/games if games else 0.5), 0, 0

def get_h2h_stats_recent(fixtures, player1, player2, n=15, decay=0.93, cutoff=None, cutoff_time=None):
    h2h = [
        f for f in fixtures
        if f["played"] and (
            (f["home_player"] == player1 and f["away_player"] == player2) or
            (f["home_player"] == player2 and f["away_player"] == player1)
        )
    ]
    if cutoff:
        h2h = [
            f for f in h2h
            if (f["date"], f["time"]) < (cutoff, cutoff_time)
        ]
    h2h = [f for f in h2h if is_valid_score(f["home_score"]) and is_valid_score(f["away_score"])]
    h2h = sorted(h2h, key=lambda x: (x["date"], x["time"]), reverse=True)[:n]
    weighted_wins = weighted_games = weighted_margin = weighted_points = 0.0
    weight = 1.0
    for match in h2h:
        hs, as_ = int(match["home_score"]), int(match["away_score"])
        pf, pa = (hs, as_) if match["home_player"] == player1 else (as_, hs)
        win = 1 if pf > pa else 0
        margin = pf - pa
        total = pf + pa
        weighted_wins += win * weight
        weighted_margin += margin * weight
        weighted_points += total * weight
        weighted_games += weight
        weight *= decay
    if weighted_games == 0:
        return 0.5, 0, 120, 0
    return (
        weighted_wins / weighted_games,
        weighted_margin / weighted_games,
        weighted_points / weighted_games,
        len(h2h)
    )

def extract_features(
    fixtures,
    target_matches,
    n_recent=2000,
    recency_decay=0.97,
    old_decay=0.85,
    h2h_weight=0.37,        # Slightly reduced H2H
    standings_weight=0.16,  # Slightly reduced standings
):
    winrates, ranks = compute_player_winrate_and_rank(fixtures)
    max_rank = max(ranks.values()) if ranks else 1
    features = []
    for match in target_matches:
        home, away = match["home_player"], match["away_player"]

        (h_wr, h_margin, h_pts, h_margin_sd, h_pts_sd, h_bigwin, h_bigloss, h_oppwr) = get_player_recent_stats_weighted(
            fixtures, home, n_recent=n_recent, recency_decay=recency_decay, old_decay=old_decay, cutoff=match["date"], cutoff_time=match["time"])
        (a_wr, a_margin, a_pts, a_margin_sd, a_pts_sd, a_bigwin, a_bigloss, a_oppwr) = get_player_recent_stats_weighted(
            fixtures, away, n_recent=n_recent, recency_decay=recency_decay, old_decay=old_decay, cutoff=match["date"], cutoff_time=match["time"])
        
        h2h_wr, h2h_margin, h2h_pts, h2h_count = get_h2h_stats_recent(fixtures, home, away, n=15, decay=0.93, cutoff=match["date"], cutoff_time=match["time"])
        a_h2h_wr, a_h2h_margin, a_h2h_pts, _ = get_h2h_stats_recent(fixtures, away, home, n=15, decay=0.93, cutoff=match["date"], cutoff_time=match["time"])
        
        home_elo = rolling_elo(fixtures, home, cutoff=match["date"], cutoff_time=match["time"])
        away_elo = rolling_elo(fixtures, away, cutoff=match["date"], cutoff_time=match["time"])

        # --- Reduced H2H effect ---
        h2h_wr *= h2h_weight
        a_h2h_wr *= h2h_weight
        h2h_margin *= h2h_weight
        a_h2h_margin *= h2h_weight
        h2h_pts *= h2h_weight
        a_h2h_pts *= h2h_weight

        # --- Add league standings/winrate as weak features ---
        home_season_winrate = winrates.get(home, 0.5) * standings_weight
        away_season_winrate = winrates.get(away, 0.5) * standings_weight
        home_league_rank = (1 - (ranks.get(home, max_rank)/max_rank)) * standings_weight  # higher = better
        away_league_rank = (1 - (ranks.get(away, max_rank)/max_rank)) * standings_weight

        features.append({
            "home_wr": h_wr, "away_wr": a_wr,
            "home_margin": h_margin, "away_margin": a_margin,
            "home_pts": h_pts, "away_pts": a_pts,
            "home_margin_sd": h_margin_sd, "away_margin_sd": a_margin_sd,
            "home_pts_sd": h_pts_sd, "away_pts_sd": a_pts_sd,
            "home_bigwin": h_bigwin, "away_bigwin": a_bigwin,
            "home_bigloss": h_bigloss, "away_bigloss": a_bigloss,
            "home_oppwr": h_oppwr, "away_oppwr": a_oppwr,
            "h2h_wr": h2h_wr, "a_h2h_wr": a_h2h_wr,
            "h2h_margin": h2h_margin, "a_h2h_margin": a_h2h_margin,
            "h2h_pts": h2h_pts, "a_h2h_pts": a_h2h_pts,
            "h2h_count": h2h_count,
            "home_elo": home_elo, "away_elo": away_elo,
            "home_season_winrate": home_season_winrate,
            "away_season_winrate": away_season_winrate,
            "home_league_rank": home_league_rank,
            "away_league_rank": away_league_rank,
        })
    df = pd.DataFrame(features)
    cols = sorted(list(df.columns))
    df = df.reindex(columns=cols)
    return df

def train_models(fixtures, n_recent=2000, recency_decay=0.97, old_decay=0.85, h2h_weight=0.37, standings_weight=0.16):
    matches = [
        f for f in fixtures
        if (
            f["played"]
            and f["home_player"] and f["away_player"]
            and is_valid_score(f["home_score"]) and is_valid_score(f["away_score"])
        )
    ]
    X = extract_features(fixtures, matches, n_recent=n_recent, recency_decay=recency_decay, old_decay=old_decay,
                        h2h_weight=h2h_weight, standings_weight=standings_weight)
    y_home = [int(f["home_score"]) for f in matches]
    y_away = [int(f["away_score"]) for f in matches]

    cols = sorted(list(X.columns))
    X = X.reindex(columns=cols)
    with open("feature_columns.json", "w") as f:
        json.dump(cols, f)

    X_train, X_test, y_home_train, y_home_test, y_away_train, y_away_test = train_test_split(
        X, y_home, y_away, test_size=0.2, random_state=42)

    xgb_params = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 4, 6, 8],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.7, 1.0],
    }
    home_xgb = XGBRegressor()
    search_home = RandomizedSearchCV(home_xgb, xgb_params, n_iter=8, scoring='neg_mean_squared_error', cv=3, verbose=1)
    search_home.fit(X_train, y_home_train)
    home_reg = search_home.best_estimator_

    away_xgb = XGBRegressor()
    search_away = RandomizedSearchCV(away_xgb, xgb_params, n_iter=8, scoring='neg_mean_squared_error', cv=3, verbose=1)
    search_away.fit(X_train, y_away_train)
    away_reg = search_away.best_estimator_

    y_home_pred = home_reg.predict(X_test)
    y_away_pred = away_reg.predict(X_test)
    print("HOME SCORE MODEL - MSE:", mean_squared_error(y_home_test, y_home_pred))
    print("AWAY SCORE MODEL - MSE:", mean_squared_error(y_away_test, y_away_pred))

    importances = home_reg.feature_importances_
    feat_names = X_train.columns
    sorted_idx = np.argsort(importances)[::-1]
    print("Top features for home prediction:")
    for idx in sorted_idx[:15]:
        print(f"{feat_names[idx]}: {importances[idx]:.4f}")

    wrong = [(y_home_test[i], y_home_pred[i]) for i in range(len(y_home_pred)) if abs(y_home_test[i] - y_home_pred[i]) > 10]
    print("Examples of large home errors (truth, pred):", wrong[:5])

    pickle.dump(home_reg, open("home_score_xgb.pkl", "wb"))
    pickle.dump(away_reg, open("away_score_xgb.pkl", "wb"))
    pickle.dump(y_home_train, open("y_home_train.pkl", "wb"))
    pickle.dump(y_away_train, open("y_away_train.pkl", "wb"))

    return home_reg, away_reg, y_home_train, y_away_train

def predict_ml_odds_for_upcoming(fixtures, upcoming_matches, n_recent=2000, recency_decay=0.97, old_decay=0.85, h2h_weight=0.37, standings_weight=0.16):
    from scipy.stats import norm
    home_reg = pickle.load(open("home_score_xgb.pkl", "rb"))
    away_reg = pickle.load(open("away_score_xgb.pkl", "rb"))

    X = extract_features(fixtures, upcoming_matches, n_recent=n_recent, recency_decay=recency_decay, old_decay=old_decay,
                        h2h_weight=h2h_weight, standings_weight=standings_weight)
    with open("feature_columns.json") as f:
        feat_order = json.load(f)
    X = X.reindex(columns=feat_order)

    home_preds = home_reg.predict(X)
    away_preds = away_reg.predict(X)

    try:
        y_home_train = pickle.load(open("y_home_train.pkl", "rb"))
        y_away_train = pickle.load(open("y_away_train.pkl", "rb"))
    except:
        y_home_train = home_preds
        y_away_train = away_preds

    std_home = np.std(y_home_train)
    std_away = np.std(y_away_train)
    std_margin = np.sqrt(std_home**2 + std_away**2)

    # New: Map ELO diff to a margin in realistic range (about 40 pts = 1 point margin)
    def elo_margin(home_elo, away_elo, scale=40):
        return (home_elo - away_elo) / scale

    for i, m in enumerate(upcoming_matches):
        home_score_pred = home_preds[i]
        away_score_pred = away_preds[i]
        margin_model = home_score_pred - away_score_pred

        # ELO-based margin
        elo_diff = X.iloc[i]["home_elo"] - X.iloc[i]["away_elo"]
        margin_elo = elo_margin(X.iloc[i]["home_elo"], X.iloc[i]["away_elo"], scale=34)

        # Final margin is 65% model, 35% ELO (tweak as you like)
        margin_mu = 0.65 * margin_model + 0.35 * margin_elo

        # Probability Home wins (margin > 0)
        p_home_win = float(norm.cdf(margin_mu / std_margin))
        p_away_win = 1 - p_home_win

        target_odd = 1.85
        target_overround = (1 / target_odd) * 2  # ≈ 1.081
        probs = [p_home_win, p_away_win]
        prob_sum = sum(probs)
        scaling = target_overround / prob_sum
        adj_probs = [p * scaling for p in probs]
        home_odds = round(1 / adj_probs[0], 2)
        away_odds = round(1 / adj_probs[1], 2)

        m["winner_odds"] = f"{home_odds} / {away_odds}"
        m["total_points_pred"] = round(home_score_pred + away_score_pred, 1)
        m["handicap"] = f"{margin_mu:+.1f}"

        # ==========================
        # REASONING GENERATION HERE
        # ==========================
        X_row = X.iloc[i]
        reasons = []
        if abs(X_row["home_elo"] - X_row["away_elo"]) > 20:
            if X_row["home_elo"] > X_row["away_elo"]:
                reasons.append(f"Home ELO advantage ({int(X_row['home_elo'])} vs {int(X_row['away_elo'])})")
            else:
                reasons.append(f"Away ELO advantage ({int(X_row['away_elo'])} vs {int(X_row['home_elo'])})")
        if abs(X_row["h2h_margin"]) > 3 and X_row["h2h_count"] > 5:
            side = "Home" if X_row["h2h_margin"] > 0 else "Away"
            reasons.append(f"{side} has H2H avg margin {X_row['h2h_margin']:+.1f}")
        if abs(X_row["home_margin"] - X_row["away_margin"]) > 3:
            side = "Home" if X_row["home_margin"] > X_row["away_margin"] else "Away"
            reasons.append(f"{side} recent avg win margin {max(X_row['home_margin'], X_row['away_margin']):+.1f}")
        if abs(X_row["home_wr"] - X_row["away_wr"]) > 0.12:
            side = "Home" if X_row["home_wr"] > X_row["away_wr"] else "Away"
            reasons.append(f"{side} recent win rate {max(X_row['home_wr'], X_row['away_wr'])*100:.0f}%")
        if abs(X_row["home_season_winrate"] - X_row["away_season_winrate"]) > 0.06:
            side = "Home" if X_row["home_season_winrate"] > X_row["away_season_winrate"] else "Away"
            reasons.append(f"{side} better season winrate")
        if abs(X_row["home_league_rank"] - X_row["away_league_rank"]) > 0.04:
            side = "Home" if X_row["home_league_rank"] > X_row["away_league_rank"] else "Away"
            reasons.append(f"{side} higher league rank")
        if not reasons:
            reasons.append("Teams are closely matched on stats")
        m["reasoning"] = "; ".join(reasons)

    return upcoming_matches
