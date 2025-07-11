from apscheduler.schedulers.blocking import BlockingScheduler
import train_ml

if __name__ == "__main__":
    scheduler = BlockingScheduler()
    scheduler.add_job(train_ml.auto_train, "interval", minutes=35)
    print("Auto ML model retraining scheduler started! (every 35 minutes)")
    scheduler.start()
    print("Scheduler stopped. Exiting...")