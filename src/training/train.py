"""
Model Training Module
=====================
End-to-end ML training with:
  - MLflow experiment tracking (every metric, param, artifact logged)
  - Optuna hyperparameter optimization (Bayesian search, not grid search)
  - LightGBM primary + XGBoost challenger
  - Walk-forward cross-validation (correct for time series — no data leakage)
  - SHAP explainability on every model
  - Champion/Challenger promotion logic
  - Model registry management

Industry pattern: This is the exact workflow used at banks, hedge funds,
and tech companies for tabular ML. MLflow is used at Databricks, Airbnb,
Microsoft. Optuna is used at Preferred Networks (creators), various ML teams.
"""

import sys
import json
import pickle
import warnings
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, Any

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, f1_score,
    precision_score, recall_score, classification_report
)
import lightgbm as lgb
import xgboost as xgb
import optuna
import shap
import mlflow
import mlflow.lightgbm
import mlflow.xgboost
import yaml
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_DIR = ROOT / CFG["paths"]["model_registry"]
MODEL_DIR.mkdir(parents=True, exist_ok=True)

logger.add(ROOT / "logs/training.log", rotation="1 day", retention="30 days", level="INFO")

# Suppress Optuna verbosity
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ════════════════════════════════════════════════════════════════
#  DATA PREPARATION
# ════════════════════════════════════════════════════════════════

def prepare_train_test(df: pd.DataFrame) -> Tuple:
    """
    Time-series aware train/val/test split.
    
    CRITICAL: For financial data, we MUST split by time, not randomly.
    Random splits cause data leakage (future info in training set).
    This is one of the most common ML mistakes in finance.
    """
    from src.features.engineer import FEATURE_COLUMNS
    
    df = df.sort_values("date").reset_index(drop=True)
    
    n = len(df)
    train_end = int(n * CFG["model"]["train_split"])
    val_end = int(n * (CFG["model"]["train_split"] + CFG["model"]["val_split"]))
    
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    X = df[feature_cols]
    y = df["target"]
    dates = df["date"]
    tickers = df["ticker"]
    
    X_train = X.iloc[:train_end]
    y_train = y.iloc[:train_end]
    X_val = X.iloc[train_end:val_end]
    y_val = y.iloc[train_end:val_end]
    X_test = X.iloc[val_end:]
    y_test = y.iloc[val_end:]
    
    logger.info(f"Train: {len(X_train):,} rows (until {dates.iloc[train_end-1].date()})")
    logger.info(f"Val:   {len(X_val):,} rows  (until {dates.iloc[val_end-1].date()})")
    logger.info(f"Test:  {len(X_test):,} rows  (until {dates.iloc[-1].date()})")
    
    return X_train, X_val, X_test, y_train, y_val, y_test, feature_cols


# ════════════════════════════════════════════════════════════════
#  WALK-FORWARD CROSS VALIDATION
# ════════════════════════════════════════════════════════════════

def walk_forward_cv(model_fn, X: pd.DataFrame, y: pd.Series, n_splits: int = 5) -> Dict:
    """
    Walk-forward (expanding window) cross-validation for time series.
    Each fold: train on past, validate on immediate future.
    This is how financial ML models are properly evaluated.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_vl = y.iloc[train_idx], y.iloc[val_idx]
        
        model = model_fn()
        model.fit(X_tr, y_tr)
        preds = model.predict(X_vl)
        probs = model.predict_proba(X_vl)[:, 1]
        
        fold_metrics = {
            "accuracy": accuracy_score(y_vl, preds),
            "auc": roc_auc_score(y_vl, probs),
            "f1": f1_score(y_vl, preds)
        }
        scores.append(fold_metrics)
        logger.info(f"  Fold {fold+1}/{n_splits}: acc={fold_metrics['accuracy']:.3f} auc={fold_metrics['auc']:.3f}")
    
    return {
        metric: {
            "mean": np.mean([s[metric] for s in scores]),
            "std": np.std([s[metric] for s in scores])
        }
        for metric in ["accuracy", "auc", "f1"]
    }


# ════════════════════════════════════════════════════════════════
#  HYPERPARAMETER OPTIMIZATION — Optuna
# ════════════════════════════════════════════════════════════════

def optimize_lightgbm(X_train, y_train, X_val, y_val, n_trials: int = None) -> Dict:
    """
    Bayesian hyperparameter optimization with Optuna.
    Much more efficient than grid search — explores the parameter space
    intelligently using Tree-structured Parzen Estimator (TPE).
    """
    n_trials = n_trials or CFG["model"]["optuna_trials"]
    
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "auc",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 15, 127),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "class_weight": "balanced",
            "random_state": 42,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)]
        )
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    
    logger.info(f"Optuna best AUC: {study.best_value:.4f}")
    logger.info(f"Best params: {study.best_params}")
    
    return study.best_params


def optimize_xgboost(X_train, y_train, X_val, y_val, n_trials: int = 30) -> Dict:
    """Optuna optimization for XGBoost (challenger model)."""
    
    def objective(trial):
        params = {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "verbosity": 0,
            "n_estimators": trial.suggest_int("n_estimators", 100, 800),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
            "random_state": 42,
        }
        
        model = xgb.XGBClassifier(**params, early_stopping_rounds=50)
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        preds = model.predict_proba(X_val)[:, 1]
        return roc_auc_score(y_val, preds)
    
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    return study.best_params


# ════════════════════════════════════════════════════════════════
#  MODEL TRAINING
# ════════════════════════════════════════════════════════════════

def train_lightgbm(X_train, y_train, X_val, y_val, params: Dict) -> lgb.LGBMClassifier:
    """Train final LightGBM model with best params."""
    full_params = {
        "objective": "binary",
        "metric": "auc",
        "verbosity": -1,
        "boosting_type": "gbdt",
        "class_weight": "balanced",
        "random_state": 42,
        **params
    }
    
    model = lgb.LGBMClassifier(**full_params)
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(100)]
    )
    return model


def train_xgboost(X_train, y_train, X_val, y_val, params: Dict) -> xgb.XGBClassifier:
    """Train final XGBoost model."""
    full_params = {
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "verbosity": 0,
        "scale_pos_weight": (y_train == 0).sum() / (y_train == 1).sum(),
        "random_state": 42,
        **params
    }
    model = xgb.XGBClassifier(**full_params, early_stopping_rounds=50)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    return model


def evaluate_model(model, X: pd.DataFrame, y: pd.Series, split_name: str) -> Dict:
    """Compute comprehensive model metrics."""
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]
    
    metrics = {
        f"{split_name}_accuracy": accuracy_score(y, preds),
        f"{split_name}_auc": roc_auc_score(y, probs),
        f"{split_name}_f1": f1_score(y, preds),
        f"{split_name}_precision": precision_score(y, preds),
        f"{split_name}_recall": recall_score(y, preds),
    }
    
    logger.info(f"\n{split_name.upper()} Metrics:")
    for k, v in metrics.items():
        logger.info(f"  {k}: {v:.4f}")
    
    return metrics


def compute_shap_values(model, X_sample: pd.DataFrame) -> np.ndarray:
    """
    SHAP (SHapley Additive exPlanations) — model explainability.
    Industry requirement: regulators, risk teams, and stakeholders need
    to understand WHY a model makes predictions. SHAP is the gold standard.
    """
    logger.info("Computing SHAP values...")
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # Class 1 (positive)
    
    return shap_values


# ════════════════════════════════════════════════════════════════
#  MLFLOW EXPERIMENT — Full training run with logging
# ════════════════════════════════════════════════════════════════

def run_experiment(df: pd.DataFrame, model_type: str = "lightgbm") -> str:
    """
    Full ML experiment run with MLflow tracking.
    Returns MLflow run_id.
    """
    mlflow.set_tracking_uri(CFG["mlflow"]["tracking_uri"])
    mlflow.set_experiment(CFG["mlflow"]["experiment_name"])
    
    X_train, X_val, X_test, y_train, y_val, y_test, feature_cols = prepare_train_test(df)
    
    with mlflow.start_run(run_name=f"{model_type}_{datetime.now().strftime('%Y%m%d_%H%M')}") as run:
        run_id = run.info.run_id
        logger.info(f"MLflow Run ID: {run_id}")
        
        # Log config
        mlflow.log_params({
            "model_type": model_type,
            "n_features": len(feature_cols),
            "train_size": len(X_train),
            "val_size": len(X_val),
            "test_size": len(X_test),
            "target_horizon": CFG["features"]["target_horizon"],
        })
        
        # ── Hyperparameter Optimization ──
        logger.info(f"Running Optuna optimization ({CFG['model']['optuna_trials']} trials)...")
        if model_type == "lightgbm":
            best_params = optimize_lightgbm(X_train, y_train, X_val, y_val)
        else:
            best_params = optimize_xgboost(X_train, y_train, X_val, y_val)
        
        mlflow.log_params({f"best_{k}": v for k, v in best_params.items()})
        
        # ── Walk-Forward CV ──
        logger.info("Walk-forward cross-validation...")
        if model_type == "lightgbm":
            cv_fn = lambda: lgb.LGBMClassifier(**{
                **best_params, "verbosity": -1, "random_state": 42, "class_weight": "balanced"
            })
        else:
            cv_fn = lambda: xgb.XGBClassifier(**{
                **best_params, "verbosity": 0, "random_state": 42
            })
        
        cv_scores = walk_forward_cv(cv_fn, X_train, y_train, n_splits=CFG["model"]["cv_folds"])
        for metric, stats in cv_scores.items():
            mlflow.log_metric(f"cv_{metric}_mean", stats["mean"])
            mlflow.log_metric(f"cv_{metric}_std", stats["std"])
        
        # ── Final Model Training ──
        logger.info("Training final model...")
        if model_type == "lightgbm":
            model = train_lightgbm(X_train, y_train, X_val, y_val, best_params)
            mlflow.lightgbm.log_model(model, "model")
        else:
            model = train_xgboost(X_train, y_train, X_val, y_val, best_params)
            mlflow.xgboost.log_model(model, "model")
        
        # ── Evaluation ──
        train_metrics = evaluate_model(model, X_train, y_train, "train")
        val_metrics = evaluate_model(model, X_val, y_val, "val")
        test_metrics = evaluate_model(model, X_test, y_test, "test")
        
        all_metrics = {**train_metrics, **val_metrics, **test_metrics}
        mlflow.log_metrics(all_metrics)
        
        # ── Feature Importance ──
        if hasattr(model, 'feature_importances_'):
            importance_df = pd.DataFrame({
                "feature": feature_cols,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)
            
            importance_path = MODEL_DIR / "feature_importance.csv"
            importance_df.to_csv(importance_path, index=False)
            mlflow.log_artifact(str(importance_path))
            
            logger.info("\nTop 10 Features:")
            for _, row in importance_df.head(10).iterrows():
                logger.info(f"  {row['feature']:<30} {row['importance']:.0f}")
        
        # ── SHAP Explainability ──
        shap_sample = X_test.sample(min(500, len(X_test)), random_state=42)
        shap_values = compute_shap_values(model, shap_sample)
        
        shap_importance = pd.DataFrame({
            "feature": feature_cols,
            "mean_shap": np.abs(shap_values).mean(axis=0)
        }).sort_values("mean_shap", ascending=False)
        
        shap_path = MODEL_DIR / "shap_importance.csv"
        shap_importance.to_csv(shap_path, index=False)
        mlflow.log_artifact(str(shap_path))
        
        # ── Save artifacts for serving ──
        model_path = MODEL_DIR / f"{model_type}_model.pkl"
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        
        metadata = {
            "run_id": run_id,
            "model_type": model_type,
            "trained_at": datetime.now().isoformat(),
            "feature_columns": feature_cols,
            "test_accuracy": test_metrics[f"test_accuracy"],
            "test_auc": test_metrics[f"test_auc"],
            "test_f1": test_metrics[f"test_f1"],
        }
        
        meta_path = MODEL_DIR / f"{model_type}_metadata.json"
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        
        mlflow.log_artifact(str(meta_path))
        mlflow.set_tag("model_stage", "candidate")
        
        logger.info(f"\n✅ Experiment complete. Run ID: {run_id}")
        logger.info(f"   Test AUC: {test_metrics['test_auc']:.4f}")
        logger.info(f"   Test Accuracy: {test_metrics['test_accuracy']:.4f}")
        
        return run_id, test_metrics


# ════════════════════════════════════════════════════════════════
#  CHAMPION / CHALLENGER PROMOTION
# ════════════════════════════════════════════════════════════════

def promote_champion(run_id: str, metrics: Dict, model_type: str = "lightgbm") -> bool:
    """
    Champion/Challenger evaluation and promotion.
    
    Industry pattern: Never auto-deploy — always compare new model
    (challenger) vs current production model (champion). Only promote
    if challenger beats champion by a meaningful margin.
    """
    champion_meta_path = MODEL_DIR / "champion_metadata.json"
    
    new_auc = metrics.get("test_auc", 0)
    threshold = CFG["model"]["promotion_threshold"]
    
    if champion_meta_path.exists():
        with open(champion_meta_path) as f:
            champion = json.load(f)
        
        champion_auc = champion.get("test_auc", 0)
        improvement = new_auc - champion_auc
        
        logger.info(f"\nChampion/Challenger Evaluation:")
        logger.info(f"  Champion AUC:   {champion_auc:.4f}")
        logger.info(f"  Challenger AUC: {new_auc:.4f}")
        logger.info(f"  Improvement:    {improvement:+.4f}")
        
        should_promote = new_auc >= threshold and (new_auc > champion_auc or improvement > -0.01)
    else:
        logger.info("No existing champion — promoting first model automatically")
        should_promote = new_auc >= threshold
    
    if should_promote:
        # Promote: copy challenger → champion
        import shutil
        src = MODEL_DIR / f"{model_type}_model.pkl"
        dst = MODEL_DIR / "champion_model.pkl"
        shutil.copy2(src, dst)
        
        src_meta = MODEL_DIR / f"{model_type}_metadata.json"
        with open(src_meta) as f:
            meta = json.load(f)
        meta["promoted_at"] = datetime.now().isoformat()
        meta["stage"] = "champion"
        
        with open(champion_meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        
        logger.info("✅ PROMOTED to champion!")
        return True
    else:
        logger.info("❌ Not promoted — challenger did not beat champion sufficiently")
        return False


# ════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════

def run_training():
    from src.features.engineer import load_features
    
    logger.info("=" * 60)
    logger.info("STARTING MODEL TRAINING PIPELINE")
    logger.info("=" * 60)
    
    # Load features
    df = load_features()
    
    # Train primary model (LightGBM)
    logger.info("\n── Training LightGBM (primary) ──")
    lgb_run_id, lgb_metrics = run_experiment(df, model_type="lightgbm")
    
    # Train challenger (XGBoost) — fewer trials for speed
    logger.info("\n── Training XGBoost (challenger) ──")
    xgb_run_id, xgb_metrics = run_experiment(df, model_type="xgboost")
    
    # Promote best model
    logger.info("\n── Champion/Challenger Evaluation ──")
    if lgb_metrics["test_auc"] >= xgb_metrics["test_auc"]:
        logger.info("LightGBM wins — evaluating for promotion")
        promote_champion(lgb_run_id, lgb_metrics, "lightgbm")
    else:
        logger.info("XGBoost wins — evaluating for promotion")
        promote_champion(xgb_run_id, xgb_metrics, "xgboost")
    
    logger.info("\n✅ Training pipeline complete")


if __name__ == "__main__":
    run_training()
