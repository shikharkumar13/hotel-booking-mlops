import os
import sys

# Add project root to Python path so src/ imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prefect import flow, task


@task(name="Preprocess Data", retries=2, retry_delay_seconds=30, log_prints=True)
def preprocess_task():
    from src.data.preprocess import run
    run()


@task(name="Train and Register Model", retries=1, retry_delay_seconds=60, log_prints=True)
def train_task():
    from src.models.train import run
    run()


@flow(name="Hotel Booking Training Pipeline", log_prints=True)
def training_pipeline():
    preprocess_task()
    train_task()


if __name__ == "__main__":
    training_pipeline()
