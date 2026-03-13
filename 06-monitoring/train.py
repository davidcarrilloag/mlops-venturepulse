"""Train VenturePulse startup traction model and log to MLflow.

Mirrors professor's 04-deployment/train.py pattern:
1. Load data
2. Prepare features (same as notebooks)
3. Train Random Forest
4. Log params, metrics, and ONE artifact: 'model'
5. Write run_id.txt for serving
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
import mlflow.sklearn


MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "venturepulse-startup-prediction"
DATA_PATH = "../data/raw/venturepulse_dataset.csv"

CATEGORICAL_COLS = [
    "sector", "location", "funding_stage",
    "founder_background", "team_diversity",
]


# ──────────────────────────────────────────────
# Helpers — identical to notebooks
# ──────────────────────────────────────────────
def precision_at_k(y_true: pd.Series, y_pred_proba: np.ndarray, k: int = 100) -> float:
    """Primary business metric: Precision@100."""
    actual_k = min(k, len(y_true))
    top_k_idx = np.argsort(y_pred_proba)[::-1][:actual_k]
    return float(y_true.iloc[top_k_idx].sum() / actual_k)


def load_data(filepath: str = DATA_PATH) -> pd.DataFrame:
    """Load VenturePulse dataset."""
    print("📥 Loading data...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df):,} rows | Success rate: {df['high_traction'].mean():.1%}")
    return df


def split_data(df: pd.DataFrame, random_state: int = 42):
    """Stratified 70/15/15 split — identical to notebooks."""
    df_temp, df_test = train_test_split(
        df, test_size=0.15, random_state=random_state,
        stratify=df["high_traction"],
    )
    val_ratio = 0.15 / (1 - 0.15)
    df_train, df_val = train_test_split(
        df_temp, test_size=val_ratio, random_state=random_state,
        stratify=df_temp["high_traction"],
    )
    print(f"✓ Train: {len(df_train):,} | Val: {len(df_val):,} | Test: {len(df_test):,}")
    return df_train, df_val, df_test


def prepare_features(df_train, df_val, df_test):
    """Feature engineering — identical to notebooks (no leakage)."""
    print("🔧 Preparing features...")
    feature_cols = [c for c in df_train.columns if c != "high_traction"]

    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    X_test = df_test[feature_cols].copy()

    y_train = df_train["high_traction"].copy()
    y_val = df_val["high_traction"].copy()
    y_test = df_test["high_traction"].copy()

    # Median imputation from training stats only
    numeric_cols = X_train.select_dtypes(include=[np.number]).columns
    impute_values = X_train[numeric_cols].median()
    for col in numeric_cols:
        X_train[col] = X_train[col].fillna(impute_values[col])
        X_val[col] = X_val[col].fillna(impute_values[col])
        X_test[col] = X_test[col].fillna(impute_values[col])

    # One-hot encoding — drop_first=False to match notebooks exactly
    X_train = pd.get_dummies(X_train, columns=CATEGORICAL_COLS, drop_first=False)
    X_val = pd.get_dummies(X_val, columns=CATEGORICAL_COLS, drop_first=False)
    X_test = pd.get_dummies(X_test, columns=CATEGORICAL_COLS, drop_first=False)

    # Align columns
    X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    print(f"✓ Features: {X_train.shape[1]}")
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_and_log(X_train, X_val, y_train, y_val) -> str:
    """Train Random Forest, evaluate, and log everything to MLflow."""
    print("🚀 Training model...")

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

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

        # Evaluate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        p100 = precision_at_k(y_val, y_proba, 100)
        auc = roc_auc_score(y_val, y_proba)
        f1 = f1_score(y_val, y_pred)
        acc = accuracy_score(y_val, y_pred)

        print(f"✓ Precision@100: {p100:.1%}  AUC: {auc:.3f}  F1: {f1:.3f}  Acc: {acc:.3f}")

        # Log to MLflow
        mlflow.log_params(params)
        mlflow.log_metric("precision_at_100", p100)
        mlflow.log_metric("auc_roc", auc)
        mlflow.log_metric("f1_score", f1)
        mlflow.log_metric("accuracy", acc)
        mlflow.set_tag("primary_metric", "precision_at_100")

        # Single artifact: the trained model
        mlflow.sklearn.log_model(model, artifact_path="model")

        run_id = run.info.run_id
        with open("run_id.txt", "w") as f:
            f.write(run_id)

        print(f"💾 Saved run_id.txt (run: {run_id})")
        print("🖥  View MLflow UI: http://localhost:5000")
        return run_id


def main():
    print("\n=== VenturePulse Training ===\n")
    df = load_data()
    df_train, df_val, df_test = split_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_features(
        df_train, df_val, df_test
    )
    train_and_log(X_train, X_val, y_train, y_val)
    print("\n✅ Training complete. Next: python app.py\n")


if __name__ == "__main__":
    main()
