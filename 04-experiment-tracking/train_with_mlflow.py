"""
VenturePulse - MLflow Experiment Tracking (DEFINITIVE VERSION)

This script replicates EXACTLY the same models and parameters from the 
notebooks (01, 02, 03) but with complete MLflow tracking.

Author: VenturePulse Team (Group 4)
Date: March 2026
MLflow Scenario: Team Collaboration (Centralized Server)
"""

import pandas as pd
import numpy as np
import pickle
import mlflow
import mlflow.sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import time
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("VENTUREPULSE - MLFLOW EXPERIMENT TRACKING")
print("Replicating notebook models with complete tracking")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================

MLFLOW_TRACKING_URI = "http://127.0.0.1:5000"
EXPERIMENT_NAME = "venturepulse-startup-prediction"
RANDOM_SEED = 42

# Set MLflow tracking
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
print(f"\n📊 MLflow Tracking URI: {MLFLOW_TRACKING_URI}")
print(f"📁 Experiment: {EXPERIMENT_NAME}")

# Create/get experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✅ Experiment configured\n")

# ============================================================================
# DATA LOADING & PREPARATION (Same as notebooks)
# ============================================================================

def load_data(filepath='../data/raw/venturepulse_dataset.csv'):
    """Load VenturePulse dataset"""
    print("📂 Loading data...")
    df = pd.read_csv(filepath)
    print(f"   Shape: {df.shape}")
    print(f"   Success rate: {df['high_traction'].mean():.1%}")
    return df

def split_data(df, test_size=0.15, val_size=0.15, random_state=42):
    """Split data - EXACTLY as in notebooks"""
    df_temp, df_test = train_test_split(
        df, test_size=test_size, random_state=random_state,
        stratify=df['high_traction']
    )
    
    val_ratio = val_size / (1 - test_size)
    df_train, df_val = train_test_split(
        df_temp, test_size=val_ratio, random_state=random_state,
        stratify=df_temp['high_traction']
    )
    
    print(f"\n📊 Data Split (same as notebooks):")
    print(f"   Train: {len(df_train):,} ({len(df_train)/len(df):.1%})")
    print(f"   Val:   {len(df_val):,} ({len(df_val)/len(df):.1%})")
    print(f"   Test:  {len(df_test):,} ({len(df_test)/len(df):.1%})")
    
    return df_train, df_val, df_test

def prepare_features(df_train, df_val, df_test):
    """Prepare features - EXACTLY as in notebooks"""
    
    feature_cols = [col for col in df_train.columns if col != 'high_traction']
    
    X_train = df_train[feature_cols].copy()
    X_val = df_val[feature_cols].copy()
    X_test = df_test[feature_cols].copy()
    
    y_train = df_train['high_traction'].copy()
    y_val = df_val['high_traction'].copy()
    y_test = df_test['high_traction'].copy()
    
    # Identify column types
    categorical_cols = X_train.select_dtypes(include=['object']).columns.tolist()
    numerical_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    
    # Handle missing values (median imputation)
    if len(numerical_cols) > 0:
        impute_values = X_train[numerical_cols].median()
        for col in numerical_cols:
            X_train[col].fillna(impute_values[col], inplace=True)
            X_val[col].fillna(impute_values[col], inplace=True)
            X_test[col].fillna(impute_values[col], inplace=True)
    
    # One-hot encoding
    if len(categorical_cols) > 0:
        X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=False, sparse=False)
        X_val = pd.get_dummies(X_val, columns=categorical_cols, drop_first=False, sparse=False)
        X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=False, sparse=False)
        
        # Align columns
        X_val = X_val.reindex(columns=X_train.columns, fill_value=0)
        X_test = X_test.reindex(columns=X_train.columns, fill_value=0)
    
    print(f"\n🔧 Feature Engineering:")
    print(f"   Categorical cols: {len(categorical_cols)}")
    print(f"   Numerical cols: {len(numerical_cols)}")
    print(f"   Final features: {X_train.shape[1]}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def precision_at_k(y_true, y_pred_proba, k=100):
    """Calculate Precision@K - PRIMARY METRIC"""
    actual_k = min(k, len(y_true))
    top_k_idx = np.argsort(y_pred_proba)[::-1][:actual_k]
    return y_true.iloc[top_k_idx].sum() / actual_k

# ============================================================================
# MODEL TRAINING WITH MLFLOW
# ============================================================================

def train_and_log_model(model, model_name, X_train, X_val, y_train, y_val, params):
    """Train model and log everything to MLflow"""
    
    print(f"\n{'='*80}")
    print(f"🤖 TRAINING: {model_name}")
    print(f"{'='*80}")
    
    with mlflow.start_run(run_name=f"{model_name.replace(' ', '-').lower()}") as run:
        
        # Log parameters
        mlflow.log_params(params)
        
        # Log tags
        mlflow.set_tag("model_type", model_name)
        mlflow.set_tag("dataset_version", "v2_30K_samples")
        mlflow.set_tag("training_date", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        mlflow.set_tag("random_seed", RANDOM_SEED)
        mlflow.set_tag("primary_metric", "precision_at_100")
        mlflow.set_tag("source", "notebook_replication")
        
        # Train model
        print(f"⏳ Training {model_name}...")
        start_time = time.time()
        model.fit(X_train, y_train)
        train_time = time.time() - start_time
        print(f"✅ Training completed in {train_time:.2f}s")
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_val = model.predict(X_val)
        y_pred_proba_train = model.predict_proba(X_train)[:, 1]
        y_pred_proba_val = model.predict_proba(X_val)[:, 1]
        
        # Training metrics
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, y_pred_train),
            'train_precision': precision_score(y_train, y_pred_train, zero_division=0),
            'train_recall': recall_score(y_train, y_pred_train, zero_division=0),
            'train_f1': f1_score(y_train, y_pred_train, zero_division=0),
            'train_auc': roc_auc_score(y_train, y_pred_proba_train),
            'train_precision_at_100': precision_at_k(y_train, y_pred_proba_train, 100)
        }
        
        # Validation metrics
        val_metrics = {
            'val_accuracy': accuracy_score(y_val, y_pred_val),
            'val_precision': precision_score(y_val, y_pred_val, zero_division=0),
            'val_recall': recall_score(y_val, y_pred_val, zero_division=0),
            'val_f1': f1_score(y_val, y_pred_val, zero_division=0),
            'val_auc': roc_auc_score(y_val, y_pred_proba_val),
            'val_precision_at_100': precision_at_k(y_val, y_pred_proba_val, 100)
        }
        
        # Log metrics
        mlflow.log_metric("training_time_seconds", train_time)
        mlflow.log_metric("n_features", X_train.shape[1])
        mlflow.log_metric("n_train_samples", len(X_train))
        mlflow.log_metric("n_val_samples", len(X_val))
        
        mlflow.log_metrics(train_metrics)
        mlflow.log_metrics(val_metrics)
        
        # Log model
        input_example = X_val.head(1)
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=f"venturepulse-{model_name.lower().replace(' ', '-')}",
            input_example=input_example
        )
        
        # Print results
        print(f"\n📊 Validation Results:")
        print(f"   Precision@100:  {val_metrics['val_precision_at_100']:.1%} ⭐ PRIMARY METRIC")
        print(f"   AUC-ROC:        {val_metrics['val_auc']:.4f}")
        print(f"   F1-Score:       {val_metrics['val_f1']:.4f}")
        print(f"   Accuracy:       {val_metrics['val_accuracy']:.4f}")
        print(f"   Precision:      {val_metrics['val_precision']:.4f}")
        print(f"   Recall:         {val_metrics['val_recall']:.4f}")
        
        run_id = run.info.run_id
        print(f"\n🔗 Run ID: {run_id}")
        print(f"🏃 View: {MLFLOW_TRACKING_URI}/#/experiments/{run.info.experiment_id}/runs/{run_id}")
        
        return model, val_metrics, run_id

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    
    # Load and prepare data
    df = load_data()
    df_train, df_val, df_test = split_data(df, random_state=RANDOM_SEED)
    X_train, X_val, X_test, y_train, y_val, y_test = prepare_features(df_train, df_val, df_test)
    
    # Calculate class ratio
    class_ratio = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"\n⚖️  Class imbalance ratio: {class_ratio:.2f}:1")
    
    # ========================================================================
    # DEFINE MODELS - EXACT SAME PARAMETERS AS NOTEBOOKS
    # ========================================================================
    
    models_config = [
        {
            'name': 'Logistic Regression',
            'model': LogisticRegression(
                class_weight='balanced',
                random_state=RANDOM_SEED,
                max_iter=1000
            ),
            'params': {
                'class_weight': 'balanced',
                'random_state': RANDOM_SEED,
                'max_iter': 1000,
                'solver': 'lbfgs'
            }
        },
        {
            'name': 'Decision Tree',
            'model': DecisionTreeClassifier(
                max_depth=10,
                min_samples_split=50,
                min_samples_leaf=20,
                class_weight='balanced',
                random_state=RANDOM_SEED
            ),
            'params': {
                'max_depth': 10,
                'min_samples_split': 50,
                'min_samples_leaf': 20,
                'class_weight': 'balanced',
                'random_state': RANDOM_SEED
            }
        },
        {
            'name': 'Random Forest',
            'model': RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                min_samples_split=50,      # Same as notebook
                min_samples_leaf=20,       # Same as notebook
                max_features='sqrt',       # Same as notebook
                class_weight='balanced',
                random_state=RANDOM_SEED,
                n_jobs=-1
            ),
            'params': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 50,
                'min_samples_leaf': 20,
                'max_features': 'sqrt',
                'class_weight': 'balanced',
                'random_state': RANDOM_SEED,
                'n_jobs': -1
            }
        }
    ]
    
    # ========================================================================
    # TRAIN ALL MODELS
    # ========================================================================
    
    results = []
    trained_models = {}
    run_ids = {}
    
    for config in models_config:
        model, metrics, run_id = train_and_log_model(
            config['model'],
            config['name'],
            X_train, X_val,
            y_train, y_val,
            config['params']
        )
        
        trained_models[config['name']] = model
        run_ids[config['name']] = run_id
        results.append({
            'model': config['name'],
            'precision_at_100': metrics['val_precision_at_100'],
            'auc_roc': metrics['val_auc'],
            'f1_score': metrics['val_f1'],
            'accuracy': metrics['val_accuracy'],
            'run_id': run_id
        })
    
    # ========================================================================
    # RESULTS SUMMARY
    # ========================================================================
    
    print("\n" + "="*80)
    print("🏆 TRAINING COMPLETE - RESULTS SUMMARY")
    print("="*80)
    
    results_df = pd.DataFrame(results)
    results_df_display = results_df[['model', 'precision_at_100', 'auc_roc', 'f1_score', 'accuracy']]
    
    # Format for display
    print("\n" + results_df_display.to_string(index=False, float_format=lambda x: f'{x:.4f}'))
    
    # Identify best model
    best_idx = results_df['precision_at_100'].idxmax()
    best_model_name = results_df.loc[best_idx, 'model']
    best_p100 = results_df.loc[best_idx, 'precision_at_100']
    best_run_id = results_df.loc[best_idx, 'run_id']
    
    print(f"\n{'='*80}")
    print(f"🥇 BEST MODEL: {best_model_name}")
    print(f"{'='*80}")
    print(f"   Precision@100: {best_p100:.1%}")
    print(f"   AUC-ROC:       {results_df.loc[best_idx, 'auc_roc']:.4f}")
    print(f"   F1-Score:      {results_df.loc[best_idx, 'f1_score']:.4f}")
    print(f"   Run ID:        {best_run_id}")
    
    # Performance vs targets
    print(f"\n📊 Performance vs Targets:")
    random_baseline = 0.25  # Success rate
    heuristic_baseline = 0.30
    target = 0.40
    
    print(f"   Random baseline (25%):    {((best_p100 - random_baseline) / random_baseline * 100):+.1f}%")
    print(f"   Heuristic baseline (30%): {((best_p100 - heuristic_baseline) / heuristic_baseline * 100):+.1f}%")
    
    if best_p100 >= target:
        print(f"   Target (40%):             ✅ ACHIEVED ({best_p100:.1%})")
    else:
        gap = target - best_p100
        print(f"   Target (40%):             ⚠️ Gap: {gap:.1%} ({best_p100:.1%})")
    
    # ========================================================================
    # SAVE METADATA
    # ========================================================================
    
    metadata = {
        'experiment_name': EXPERIMENT_NAME,
        'mlflow_tracking_uri': MLFLOW_TRACKING_URI,
        'training_date': datetime.now().isoformat(),
        'dataset_info': {
            'total_samples': len(df),
            'train_samples': len(X_train),
            'val_samples': len(X_val),
            'test_samples': len(X_test),
            'n_features': X_train.shape[1],
            'success_rate': float(df['high_traction'].mean()),
            'random_seed': RANDOM_SEED
        },
        'best_model': {
            'name': best_model_name,
            'run_id': best_run_id,
            'precision_at_100': float(best_p100),
            'auc_roc': float(results_df.loc[best_idx, 'auc_roc']),
            'f1_score': float(results_df.loc[best_idx, 'f1_score']),
            'beats_target': bool(best_p100 >= target)
        },
        'all_models': [
            {
                'name': r['model'],
                'precision_at_100': float(r['precision_at_100']),
                'auc_roc': float(r['auc_roc']),
                'f1_score': float(r['f1_score']),
                'accuracy': float(r['accuracy']),
                'run_id': r['run_id']
            }
            for r in results
        ]
    }
    
    metadata_path = 'mlflow_experiment_results.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\n✅ Results saved: {metadata_path}")
    
    # ========================================================================
    # FINAL INSTRUCTIONS
    # ========================================================================
    
    print("\n" + "="*80)
    print("📊 NEXT STEPS")
    print("="*80)
    print(f"\n1. View results in MLflow UI:")
    print(f"   {MLFLOW_TRACKING_URI}")
    print(f"\n2. Compare all runs:")
    print(f"   • Click experiment: '{EXPERIMENT_NAME}'")
    print(f"   • Select all 3 runs")
    print(f"   • Click 'Compare' button")
    print(f"\n3. Explore best model:")
    print(f"   • Go to run: {best_run_id}")
    print(f"   • View Parameters, Metrics, Artifacts")
    print(f"   • Download model if needed")
    print(f"\n4. Model Registry:")
    print(f"   • Click 'Models' in sidebar")
    print(f"   • Find: venturepulse-{best_model_name.lower().replace(' ', '-')}")
    print(f"   • Promote to 'Staging' or 'Production'")
    
    print("\n" + "="*80)
    print("✅ MLFLOW EXPERIMENT TRACKING COMPLETE")
    print("="*80)
    print(f"\nAll models tracked with EXACT same parameters as notebooks")
    print(f"Results consistent with notebook explorations")
    print(f"Ready for deployment: {best_model_name} @ {best_p100:.1%} P@100")
    print("="*80)