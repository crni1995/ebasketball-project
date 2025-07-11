from app import get_all_fixtures
from ml_odds import predict_ml_odds_for_upcoming

fixtures = get_all_fixtures()
played = [f for f in fixtures if f["played"] and f.get("home_score") is not None and f.get("away_score") is not None]
played_sorted = sorted(played, key=lambda x: (x["date"], x["time"]))
last_100 = played_sorted[-100:]

predictions = predict_ml_odds_for_upcoming(
    fixtures,
    last_100,
    n_recent=2000,
    recency_decay=0.97,
    old_decay=0.85
)

for i, (m, pred) in enumerate(zip(last_100, predictions), 1):
    print(f"Match {i}/100")
    print(f"Date: {m['date']} Time: {m['time']}")
    print(f"{m['home_player']} vs {m['away_player']}")
    print(f"  Actual:    {m['home_score']} - {m['away_score']}")
    print(f"  Predicted Total: {pred['total_points_pred']}, Handicap: {pred['handicap']}, Odds (home/away): {pred['winner_odds']}")
    print(f"  Pred Margin: {float(pred['handicap']):+.1f}")
    print(f"  Reasoning: {pred.get('reasoning','')}")
    print("-" * 50)
    input("Press Enter for next match...")

# After all, print accuracy:
from sklearn.metrics import mean_absolute_error

actual_totals = [int(m['home_score']) + int(m['away_score']) for m in last_100]
pred_totals = [pred['total_points_pred'] for pred in predictions]
actual_margins = [int(m['home_score']) - int(m['away_score']) for m in last_100]
pred_margins = [float(pred['handicap']) for pred in predictions]
winner_acc = sum((a > 0) == (p > 0) for a, p in zip(actual_margins, pred_margins)) / len(actual_margins)

print()
print("Mean Absolute Error (Total Points):", mean_absolute_error(actual_totals, pred_totals))
print("Mean Absolute Error (Margin):", mean_absolute_error(actual_margins, pred_margins))
print("Winner Accuracy:", f"{winner_acc*100:.1f}%")
