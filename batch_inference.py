"""Batch inference: predict 20 held-out passenger records together."""
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
    records = pd.read_csv(records_path).head(20).copy()
    if len(records) < 20:
        raise SystemExit(f"Need at least 20 held-out records; found {len(records)}.")
    predictions = model.predict(records)
    output = records.copy()
    output.insert(0, "Record_ID", range(1, len(output) + 1))
    output["Predicted_Survived"] = predictions.astype(int)
    output["Prediction_Label"] = output["Predicted_Survived"].map(
        {1: "Survived", 0: "Did not survive"}
    )
    output.to_csv(ROOT / "batch_predictions.csv", index=False)
    print("BATCH INFERENCE")
    print(f"Records processed together: {len(output)}")
    print(output[["Record_ID", "Predicted_Survived", "Prediction_Label"]].to_string(index=False))
    print(f"Saved results to: {ROOT / 'batch_predictions.csv'}")


if __name__ == "__main__":
    main()
