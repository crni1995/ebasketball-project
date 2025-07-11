# train_ml.py

from results_cache import update_results_cache
from ml_odds import train_models
import json
import os

LAST_TRAIN_FILE = "last_model_fixture.json"

def get_last_trained_fixture():
    if os.path.exists(LAST_TRAIN_FILE):
        with open(LAST_TRAIN_FILE, "r") as f:
            return json.load(f)
    return None

def set_last_trained_fixture(fix):
    fix_to_save = fix.copy()
    if hasattr(fix_to_save["date"], "isoformat"):
        fix_to_save["date"] = fix_to_save["date"].isoformat()
    if hasattr(fix_to_save["time"], "isoformat"):
        fix_to_save["time"] = fix_to_save["time"].isoformat()
    with open(LAST_TRAIN_FILE, "w") as f:
        json.dump(fix_to_save, f)

def auto_train():
    fixtures = update_results_cache()  # Now this loads and updates results!
    played = [f for f in fixtures if f["played"] and f.get("home_score") is not None and f.get("away_score") is not None]
    if not played:
        print("[AutoTrain] No played fixtures found.")
        return

    latest_played = max(played, key=lambda x: (x["date"], x["time"]))
    last_trained = get_last_trained_fixture()
    lp_date = latest_played["date"]
    lp_time = latest_played["time"]
    if last_trained and (last_trained["date"], last_trained["time"]) == (lp_date, lp_time):
        print("[AutoTrain] No new results. Skipping retrain.")
        return

    print("Training ML models for eBasketball odds...")
    train_models(
        fixtures,
        n_recent=2000,
        recency_decay=0.97,
        old_decay=0.85,
        h2h_weight=0.37,
        standings_weight=0.16
    )
    set_last_trained_fixture({"date": lp_date, "time": lp_time})
    print("Training complete! Models and feature order saved.")

if __name__ == "__main__":
    auto_train()
