"""Train VenturePulse startup traction model and package for deployment."""

from __future__ import annotations

import shutil
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split


DATA_PATH = "../data/raw/venturepulse_dataset.csv"
DEPLOYMENT_MODEL_PATH = Path("models/model")

CATEGORICAL_COLS = [
    "sector",
    "location",
    "funding_stage",
    "founder_background",
    "team_diversity",
]


def precision_at_k(y_true, y_pred_proba, k=100):
    actual_k = min(k, len(y_true))
    top_k_idx = np.argsort(y_pred_proba)[::-1][:actual_k]
    return float(y_true.iloc[top_k_idx].sum() / actual_k)


def load_data(filepath=DATA_PATH):
    print("Loading data...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df):,} rows | Success rate: {df['high_traction'].mean():.1%}")
    return df


def prepare_features(df_train, df_val, df_test):
    feature_cols = [c for c in df_train.columns if c != "high_traction"]
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    y_train = df_train["high_traction"].copy()
    y_val = df_val["high_traction"].copy()
    y_test = df_test["high_traction"].copy()

    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    impute_values = X_train[numeric_cols].median()
    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(impute_values[col])
        X_val[col] = X_val[col].fillna(impute_values[col])
        X_test[col] = X_test[col].fillna(impute_values[col])

    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=False)
    X_val = pd.get_dummies(X_val, columns=CATEGORICAL_COLS, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=False)
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"Features: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_log(X_train, X_val, y_train, y_val):
    print("Training model...")
    mlflow.set_experiment("venturepulse-startup-prediction")

    params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 50,
        "min_samples_leaf": 20,
        "max_features": "sqrt",
        "class_weight": "balanced",
        "random_state": 42,
    }

    with mlflow.start_run() as run:
        model = RandomForestClassifier(**params, n_jobs=-1)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        p100 = precision_at_k(y_val, y_proba, 100)
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        print(
            f"Precision@100: {p100:.1%}  AUC: {auc:.3f}"
            f"  F1: {f1:.3f}  Acc: {acc:.3f}"
        )

        mlflow.log_params(params)
        mlflow.log_metric("precision_at_100", p100)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "model")

        run_id = run.info.run_id
        print(f"Run ID: {run_id}")
        print(f"Artifact URI: {mlflow.get_artifact_uri()}")

    # Save model locally — baked into Docker image by CI/CD
    print("Creating deployment-ready model...")
    if DEPLOYMENT_MODEL_PATH.exists():
        shutil.rmtree(DEPLOYMENT_MODEL_PATH)
    DEPLOYMENT_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    mlflow.sklearn.save_model(model, str(DEPLOYMENT_MODEL_PATH))
    print(f"Model saved to: {DEPLOYMENT_MODEL_PATH}")

    with open("run_id.txt", "w") as f:
        f.write(run_id)

    return run_id


def main():
    df = load_data()
    df_temp, df_test = train_test_split(
        df, test_size=0.15, random_state=42, stratify=df["high_traction"]
    )
    val_ratio = 0.15 / (1 - 0.15)
    df_train, df_val = train_test_split(
        df_temp,
        test_size=val_ratio,
        random_state=42,
        stratify=df_temp["high_traction"],
    )
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_features(
        df_train, df_val, df_test
    )
    train_and_log(X_train, X_val, y_train, y_val)
    print("Training complete.")


if __name__ == "__main__":
    main()
