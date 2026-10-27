"""
Train baseline models and save artifacts for the Streamlit dashboard.
The split is chronological to reduce time-series leakage.
"""
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV

from generate_data import generate_dataset

DATA_PATH = Path("data/sensor_data.csv")
MODEL_DIR = Path("models")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BASE_COLUMNS = [
    "temperature", "humidity", "soil_moisture",
    "temperature_peer", "soil_moisture_peer"
]


def make_features(df):
    """Create temporal and cross-sensor features without using labels."""
    out = df.copy()
    out["temperature_change"] = out["temperature"].diff().fillna(0)
    out["humidity_change"] = out["humidity"].diff().fillna(0)
    out["soil_moisture_change"] = out["soil_moisture"].diff().fillna(0)

    out["temperature_rolling_mean"] = out["temperature"].rolling(10, min_periods=1).mean()
    out["humidity_rolling_mean"] = out["humidity"].rolling(10, min_periods=1).mean()
    out["soil_rolling_mean"] = out["soil_moisture"].rolling(10, min_periods=1).mean()

    out["temperature_rolling_std"] = out["temperature"].rolling(10, min_periods=2).std().fillna(0)
    out["humidity_rolling_std"] = out["humidity"].rolling(10, min_periods=2).std().fillna(0)
    out["soil_rolling_std"] = out["soil_moisture"].rolling(10, min_periods=2).std().fillna(0)

    out["temperature_peer_diff"] = (
        out["temperature"] - out["temperature_peer"]
    ).abs()
    out["soil_peer_diff"] = (
        out["soil_moisture"] - out["soil_moisture_peer"]
    ).abs()

    # Approximate frozen-sensor indicator: near-zero change over a sequence.
    out["soil_frozen"] = (
        out["soil_moisture"].diff().abs().rolling(5, min_periods=3).sum().fillna(1e-6) < 1e-6
    ).astype(int)

    return out


FEATURE_COLUMNS = [
    *BASE_COLUMNS,
    "temperature_change", "humidity_change", "soil_moisture_change",
    "temperature_rolling_mean", "humidity_rolling_mean", "soil_rolling_mean",
    "temperature_rolling_std", "humidity_rolling_std", "soil_rolling_std",
    "temperature_peer_diff", "soil_peer_diff", "soil_frozen",
]


def load_or_create_data():
    if DATA_PATH.exists():
        df = pd.read_csv(DATA_PATH, parse_dates=["timestamp"])
    else:
        df = generate_dataset()
        DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(DATA_PATH, index=False)
    return df


def main():
    df = load_or_create_data()
    df = make_features(df)
    X = df[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    y = df["label"].astype(int)

    split = int(len(df) * 0.75)
    X_train, X_test = X.iloc[:split], X.iloc[split:]
    y_train, y_test = y.iloc[:split], y.iloc[split:]

    # Calibrated Random Forest: calibration uses internal CV on training data.
    base_model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=250,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])
    # Keep calibration folds chronological only approximately; this is a teaching
    # prototype. For a formal study, use a time-aware calibration procedure.
    model = CalibratedClassifierCV(base_model, method="sigmoid", cv=3)
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    print("\n=== Calibrated Random Forest ===")
    print(classification_report(y_test, pred, target_names=["normal", "manipulated"], zero_division=0))
    print("Confusion matrix [normal, manipulated]:")
    print(confusion_matrix(y_test, pred))
    if len(np.unique(y_test)) == 2:
        print("ROC-AUC:", round(roc_auc_score(y_test, probs), 4))

    # Isolation Forest is trained only on rows labeled normal in training data.
    normal_train = X_train[y_train == 0]
    isolation = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("detector", IsolationForest(
            n_estimators=200,
            contamination="auto",
            random_state=42,
        )),
    ])
    isolation.fit(normal_train)

    joblib.dump({
        "model": model,
        "isolation_forest": isolation,
        "feature_columns": FEATURE_COLUMNS,
        "base_columns": BASE_COLUMNS,
        "decision_threshold": 0.65,
        "review_lower_threshold": 0.35,
    }, MODEL_DIR / "sensor_models.joblib")
    print(f"\nSaved model artifacts to {MODEL_DIR / 'sensor_models.joblib'}")


if __name__ == "__main__":
    main()
