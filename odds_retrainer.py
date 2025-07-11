# odds_retrainer.py
import time
import subprocess

while True:
    print("\n[OddsRetrainer] Running auto_train (model retraining) ...")
    subprocess.run(["python3", "train_ml.py"])
    print("[OddsRetrainer] Sleeping 35 minutes...\n")
    time.sleep(35 * 60)
# This script is designed to run indefinitely, retraining the ML models every 35 minutes.