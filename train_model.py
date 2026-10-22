"""Train and save one reusable Titanic preprocessing + classification pipeline."""
from pathlib import Path
import json
import sys

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "train.csv"
MODEL_PATH = ROOT / "model.pkl"

FEATURES = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]


def main():
    if not DATA_PATH.exists():
        print("ERROR: train.csv not found.")
        print("Download it from https://www.kaggle.com/competitions/titanic/data")
        print(f"and place it here: {DATA_PATH}")
        sys.exit(1)

    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + ["Survived"])
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"train.csv is missing required columns: {sorted(missing)}")

    X = data[FEATURES].copy()
    y = data["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    preprocess = ColumnTransformer(transformers=[
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])
    pipeline = Pipeline(steps=[
        ("preprocess", preprocess),
        ("classifier", LogisticRegression(max_iter=2000, random_state=42)),
    ])

    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    accuracy = float(accuracy_score(y_test, predictions))
    joblib.dump(pipeline, MODEL_PATH)

    # Save held-out records without target labels; inference scripts only use these features.
    X_test.reset_index(drop=True).to_csv(ROOT / "test_records.csv", index=False)
    y_test.reset_index(drop=True).rename("Survived").to_csv(ROOT / "test_labels.csv", index=False)

    metrics = {
        "model": "LogisticRegression in a scikit-learn Pipeline",
        "dataset": "Kaggle Titanic train.csv",
        "random_state": 42,
        "training_rows": int(len(X_train)),
        "testing_rows": int(len(X_test)),
        "test_accuracy": round(accuracy, 4),
        "test_accuracy_percent": round(accuracy * 100, 2),
        "features": FEATURES,
        "note": "Inference examples use held-out test records not used for fitting."
    }
    (ROOT / "training_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print("TRAINING COMPLETE")
    print(f"Training records: {len(X_train)}")
    print(f"Held-out testing records: {len(X_test)}")
    print(f"Test accuracy: {accuracy:.4f} ({accuracy * 100:.2f}%)")
    print(f"Saved model: {MODEL_PATH}")
    print("Saved test_records.csv and test_labels.csv")


if __name__ == "__main__":
    main()
