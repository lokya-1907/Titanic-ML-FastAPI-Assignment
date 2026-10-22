"""Streaming inference simulation: one record at a time, one-second delay."""
from pathlib import Path
import time
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    model_path = ROOT / "model.pkl"
    records_path = ROOT / "test_records.csv"
    if not model_path.exists() or not records_path.exists():
        raise SystemExit("Run `python train_model.py` first.")
    model = joblib.load(model_path)
    records = pd.read_csv(records_path).head(10)
    if len(records) < 10:
        raise SystemExit(f"Need at least 10 held-out records; found {len(records)}.")
    print("STREAMING INFERENCE (one record every second)")
    for index, (_, row) in enumerate(records.iterrows(), start=1):
        incoming_record = pd.DataFrame([row])
        prediction = int(model.predict(incoming_record)[0])
        label = "Survived" if prediction == 1 else "Did not survive"
        print(f"Incoming record {index:02d}: prediction={prediction} ({label})")
        if index < len(records):
            time.sleep(1)


if __name__ == "__main__":
    main()
