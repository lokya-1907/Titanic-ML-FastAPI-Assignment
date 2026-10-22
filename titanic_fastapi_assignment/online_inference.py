"""Online inference: predict one unseen held-out passenger record."""
from pathlib import Path
import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    model_path = ROOT / "model.pkl"
    records_path = ROOT / "test_records.csv"
    if not model_path.exists() or not records_path.exists():
        raise SystemExit("Run `python train_model.py` first.")
    model = joblib.load(model_path)
    records = pd.read_csv(records_path)
    if records.empty:
        raise SystemExit("No held-out records found.")
    passenger = records.iloc[[0]]
    prediction = int(model.predict(passenger)[0])
    label = "Survived" if prediction == 1 else "Did not survive"
    print("ONLINE INFERENCE")
    print("Input passenger details:")
    print(passenger.to_string(index=False))
    print(f"Prediction: {prediction} ({label})")


if __name__ == "__main__":
    main()
