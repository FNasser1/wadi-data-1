
"""
failure_predictor.py
--------------------
Random Forest classifier for failure-mode prediction:
Classes: ["normal", "bearing_wear", "imbalance", "misalignment"]

Features: rolling stats + vibration features from data_processor.py
Train/test split is done by groups (machine_id) to avoid leakage.

CLI:
    python failure_predictor.py --train_csv processed_features.csv --model_out rf_failure_model.joblib
    # If --train_csv is not provided, a synthetic dataset is generated automatically.

API:
    from failure_predictor import train_model, load_model, predict_proba_df
"""

from __future__ import annotations
import argparse
from typing import Tuple, Dict
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.model_selection import GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder
import joblib

from data_generator import generate_dataset
from data_processor import clean_sensor_data, add_rolling_stats, add_vibration_features

RANDOM_STATE = 1337
CLASSES = ["normal", "bearing_wear", "imbalance", "misalignment"]


def _make_labels(df: pd.DataFrame, deg_threshold: float = 0.45) -> pd.Series:
    fm = df["failure_mode"].fillna("healthy")
    deg = df["degradation_level"].fillna(0.0)
    labels = np.where((fm != "healthy") & (deg >= deg_threshold), fm, "normal")
    return pd.Series(labels, index=df.index).astype(str)


def _prepare_features(raw_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series, pd.Series, list]:
    df = clean_sensor_data(raw_df, resample_freq="5min")  # downsample
    df = add_rolling_stats(df, window="60min")
    df = add_vibration_features(df, window="60min")

    y = _make_labels(df, deg_threshold=0.45)

    base_cols = ["rpm","pressure","temperature","vib_x","vib_y","vib_z","vib_overall"]
    roll_cols = [c for c in df.columns if any(c.startswith(k) for k in ["rpm_mean","rpm_std","rpm_max",
                                                                       "pressure_mean","pressure_std","pressure_max",
                                                                       "temperature_mean","temperature_std","temperature_max",
                                                                       "vib_x_mean","vib_x_std","vib_x_max",
                                                                       "vib_y_mean","vib_y_std","vib_y_max",
                                                                       "vib_z_mean","vib_z_std","vib_z_max"])]
    vib_feat_cols = [c for c in df.columns if any(substr in c for substr in ["_rms_","_kurtosis_","_crest_"])]
    feat_cols = base_cols + roll_cols + vib_feat_cols

    X = df[feat_cols].copy()
    valid = X.notna().all(axis=1)
    X = X[valid]
    y = y.loc[X.index]
    groups = df.loc[X.index, "machine_id"]

    return X, y, groups, feat_cols


def _generate_or_load(csv_path: str | None, days: int = 30, n_machines: int = 10) -> pd.DataFrame:
    if csv_path:
        return pd.read_csv(csv_path)
    df, _ = generate_dataset(days=days, n_machines=n_machines, freq="1min", seed=RANDOM_STATE)
    return df


def train_model(train_csv: str | None = None, model_out: str = "rf_failure_model.joblib") -> Dict:
    raw = _generate_or_load(train_csv)
    X, y, groups, feat_cols = _prepare_features(raw)

    le = LabelEncoder()
    le.fit(CLASSES)
    y_enc = le.transform(y)

    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=RANDOM_STATE)
    train_idx, test_idx = next(gss.split(X, y_enc, groups=groups))
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y_enc[train_idx], y_enc[test_idx]

    rf = RandomForestClassifier(
        n_estimators=400,
        max_depth=16,
        min_samples_split=4,
        min_samples_leaf=2,
        n_jobs=-1,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE
    )
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=le.classes_, digits=3)
    cm = confusion_matrix(y_test, y_pred)

    payload = {"model": rf, "label_encoder": le, "feature_columns": feat_cols, "classes": list(le.classes_)}
    joblib.dump(payload, model_out)

    return {
        "accuracy": float(acc),
        "classification_report": report,
        "confusion_matrix": cm.tolist(),
        "model_path": model_out,
        "n_train": int(len(train_idx)),
        "n_test": int(len(test_idx)),
    }


def load_model(model_path: str = "rf_failure_model.joblib"):
    return joblib.load(model_path)


def predict_proba_df(model_bundle, df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = model_bundle["feature_columns"]
    needs_proc = any(c not in df.columns for c in feat_cols)
    if needs_proc:
        df = clean_sensor_data(df, resample_freq="5min")
        df = add_rolling_stats(df, window="60min")
        df = add_vibration_features(df, window="60min")

    X = df[feat_cols].dropna()
    probs = model_bundle["model"].predict_proba(X)
    return pd.DataFrame(probs, columns=model_bundle["classes"], index=X.index)


def main():
    parser = argparse.ArgumentParser(description="Train RandomForest failure predictor")
    parser.add_argument("--train_csv", type=str, default=None, help="Path to processed CSV (optional)")
    parser.add_argument("--model_out", type=str, default="rf_failure_model.joblib")
    args = parser.parse_args()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = train_model(args.train_csv, args.model_out)

    print("=== Failure Predictor (RandomForest) ===")
    print(f"Saved model: {result['model_path']}")
    print(f"Train size: {result['n_train']}, Test size: {result['n_test']}")
    print(f"Accuracy: {result['accuracy']:.3f}")
    print("Classification report:\n", result["classification_report"])
    print("Confusion matrix:", result["confusion_matrix"])


if __name__ == "__main__":
    main()
