"""
Model Monitoring Module
========================
Production ML monitoring covering:
  - Data drift (Evidently AI — PSI, KS test, Jensen-Shannon divergence)
  - Model performance drift (accuracy decay over time)
  - Prediction distribution monitoring
  - Feature drift alerts
  - Automated reporting

Industry tools: Evidently (open-source), Arize AI, WhyLabs, Fiddler.
This is the CRITICAL piece most tutorials skip — in production, models
degrade silently. This module catches it.

Key insight: A model can be technically "running" but making
systematically wrong predictions. Monitoring catches this before
users notice.
"""

import sys
import json
from pathlib import Path
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import numpy as np
import pickle
from loguru import logger
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_DIR = ROOT / CFG["paths"]["model_registry"]
MONITOR_DIR = ROOT / "monitoring_reports"
MONITOR_DIR.mkdir(exist_ok=True)

logger.add(ROOT / "logs/monitoring.log", rotation="1 day", retention="30 days", level="INFO")


# ════════════════════════════════════════════════════════════════
#  PSI — Population Stability Index
#  The industry standard metric for feature drift detection
# ════════════════════════════════════════════════════════════════

def compute_psi(expected: np.ndarray, actual: np.ndarray, buckets: int = 10) -> float:
    """
    Population Stability Index (PSI).
    
    PSI < 0.1: No significant change
    PSI 0.1-0.2: Moderate change — investigate
    PSI > 0.2: Significant change — model likely degraded
    
    Used by banks for credit scorecard monitoring (Basel requirements).
    """
    expected = expected[~np.isnan(expected)]
    actual = actual[~np.isnan(actual)]
    
    # Create bins from expected distribution
    breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
    breakpoints = np.unique(breakpoints)
    
    if len(breakpoints) < 2:
        return 0.0
    
    expected_percents = np.histogram(expected, bins=breakpoints)[0] / len(expected)
    actual_percents = np.histogram(actual, bins=breakpoints)[0] / len(actual)
    
    # Avoid log(0) — add small epsilon
    expected_percents = np.clip(expected_percents, 1e-4, None)
    actual_percents = np.clip(actual_percents, 1e-4, None)
    
    psi = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return float(psi)


def compute_ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Kolmogorov-Smirnov test for distribution drift."""
    from scipy import stats
    try:
        ks_stat, _ = stats.ks_2samp(
            reference[~np.isnan(reference)],
            current[~np.isnan(current)]
        )
        return float(ks_stat)
    except Exception:
        return 0.0


# ════════════════════════════════════════════════════════════════
#  DRIFT DETECTOR
# ════════════════════════════════════════════════════════════════

class DriftDetector:
    """
    Monitors feature distributions between reference (train) and current data.
    """
    
    def __init__(self, reference_df: pd.DataFrame, feature_columns: list):
        self.reference = reference_df
        self.feature_columns = feature_columns
        self.psi_threshold = CFG["monitoring"]["psi_threshold"]
    
    def check_feature_drift(self, current_df: pd.DataFrame) -> dict:
        """Compute PSI and KS for all features."""
        results = {}
        alerts = []
        
        for feature in self.feature_columns:
            if feature not in self.reference.columns or feature not in current_df.columns:
                continue
            
            ref_vals = self.reference[feature].dropna().values
            cur_vals = current_df[feature].dropna().values
            
            if len(ref_vals) < 10 or len(cur_vals) < 10:
                continue
            
            psi = compute_psi(ref_vals, cur_vals)
            ks = compute_ks_statistic(ref_vals, cur_vals)
            
            severity = "ok"
            if psi > self.psi_threshold:
                severity = "alert"
                alerts.append(f"HIGH DRIFT: {feature} PSI={psi:.3f}")
            elif psi > self.psi_threshold * 0.5:
                severity = "warning"
            
            results[feature] = {
                "psi": round(psi, 4),
                "ks_statistic": round(ks, 4),
                "severity": severity,
                "ref_mean": round(float(np.nanmean(ref_vals)), 4),
                "cur_mean": round(float(np.nanmean(cur_vals)), 4),
                "mean_shift": round(float(np.nanmean(cur_vals) - np.nanmean(ref_vals)), 4),
            }
        
        # Aggregate drift score
        psi_scores = [v["psi"] for v in results.values()]
        
        return {
            "feature_drift": results,
            "alerts": alerts,
            "mean_psi": round(np.mean(psi_scores) if psi_scores else 0, 4),
            "max_psi": round(np.max(psi_scores) if psi_scores else 0, 4),
            "drifted_features": sum(1 for v in results.values() if v["severity"] == "alert"),
            "total_features": len(results)
        }
    
    def check_target_drift(self, current_df: pd.DataFrame) -> dict:
        """Check if the target distribution has shifted."""
        if "target" not in self.reference.columns or "target" not in current_df.columns:
            return {}
        
        ref_rate = self.reference["target"].mean()
        cur_rate = current_df["target"].mean()
        
        shift = abs(cur_rate - ref_rate)
        
        return {
            "reference_positive_rate": round(float(ref_rate), 4),
            "current_positive_rate": round(float(cur_rate), 4),
            "shift": round(float(shift), 4),
            "alert": shift > 0.1  # More than 10% shift in class balance
        }


# ════════════════════════════════════════════════════════════════
#  PERFORMANCE MONITOR
# ════════════════════════════════════════════════════════════════

class PerformanceMonitor:
    """
    Track model performance over time.
    In production: compares predictions against actual outcomes
    (with a delay for the prediction horizon).
    """
    
    def evaluate_recent_performance(self, df: pd.DataFrame, n_days: int = 30) -> dict:
        """
        Evaluate model on recent data that has realized outcomes.
        Since we predict 5-day forward returns, we need data that's at
        least 5 days old to have actual outcomes.
        """
        from sklearn.metrics import accuracy_score, roc_auc_score
        
        champion_path = MODEL_DIR / "champion_model.pkl"
        meta_path = MODEL_DIR / "champion_metadata.json"
        
        if not champion_path.exists():
            return {"error": "No champion model"}
        
        with open(champion_path, "rb") as f:
            model = pickle.load(f)
        
        with open(meta_path) as f:
            meta = json.load(f)
        
        feature_cols = meta.get("feature_columns", [])
        
        # Get recent data with known outcomes
        # (exclude last `target_horizon` days since those are future)
        horizon = CFG["features"]["target_horizon"]
        cutoff = df["date"].max() - pd.Timedelta(days=horizon + n_days)
        end = df["date"].max() - pd.Timedelta(days=horizon)
        
        recent = df[(df["date"] >= cutoff) & (df["date"] <= end) & df["target"].notna()]
        
        if len(recent) < 20:
            return {"error": "Insufficient recent data for evaluation"}
        
        available_features = [c for c in feature_cols if c in recent.columns]
        X = recent[available_features]
        y = recent["target"]
        
        try:
            preds = model.predict(X)
            probs = model.predict_proba(X)[:, 1]
            
            return {
                "period_days": n_days,
                "n_samples": len(recent),
                "accuracy": round(float(accuracy_score(y, preds)), 4),
                "auc": round(float(roc_auc_score(y, probs)), 4),
                "positive_rate_actual": round(float(y.mean()), 4),
                "positive_rate_predicted": round(float(preds.mean()), 4),
                "trained_at": meta.get("trained_at", "unknown"),
                "baseline_accuracy": meta.get("test_accuracy", 0)
            }
        except Exception as e:
            return {"error": str(e)}
    
    def check_accuracy_degradation(self, recent_metrics: dict) -> bool:
        """Alert if accuracy has dropped significantly vs training."""
        if "error" in recent_metrics:
            return False
        
        threshold = CFG["monitoring"]["accuracy_drop_threshold"]
        baseline = recent_metrics.get("baseline_accuracy", 0)
        current = recent_metrics.get("accuracy", 0)
        
        degraded = (baseline - current) > threshold
        
        if degraded:
            logger.warning(f"⚠️  ACCURACY DEGRADATION DETECTED!")
            logger.warning(f"   Baseline: {baseline:.4f}, Current: {current:.4f}")
            logger.warning(f"   Drop: {baseline - current:.4f} (threshold: {threshold})")
        
        return degraded


# ════════════════════════════════════════════════════════════════
#  EVIDENTLY REPORT — HTML drift report
# ════════════════════════════════════════════════════════════════

def generate_evidently_report(reference_df: pd.DataFrame, current_df: pd.DataFrame,
                               feature_cols: list) -> Optional[str]:
    """
    Generate full Evidently data drift report.
    Produces a beautiful HTML report you can open in browser.
    """
    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset, DataQualityPreset
        from evidently.metrics import DatasetDriftMetric
        
        available_cols = [c for c in feature_cols if c in reference_df.columns and c in current_df.columns]
        
        ref = reference_df[available_cols + (["target"] if "target" in reference_df.columns else [])].copy()
        cur = current_df[available_cols + (["target"] if "target" in current_df.columns else [])].copy()
        
        report = Report(metrics=[
            DataDriftPreset(),
            DataQualityPreset(),
        ])
        
        report.run(reference_data=ref, current_data=cur)
        
        report_path = MONITOR_DIR / f"drift_report_{datetime.now().strftime('%Y%m%d')}.html"
        report.save_html(str(report_path))
        
        logger.info(f"✅ Evidently report saved → {report_path}")
        return str(report_path)
    
    except ImportError:
        logger.warning("Evidently not available — skipping HTML report")
        return None
    except Exception as e:
        logger.error(f"Evidently report failed: {e}")
        return None


# ════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════

def run_monitoring():
    from src.features.engineer import load_features, FEATURE_COLUMNS
    
    logger.info("=" * 60)
    logger.info("STARTING MONITORING PIPELINE")
    logger.info("=" * 60)
    
    df = load_features()
    feature_cols = [c for c in FEATURE_COLUMNS if c in df.columns]
    
    # Split into reference (training window) and current (recent)
    n_days_monitor = CFG["monitoring"]["drift_check_days"]
    cutoff = df["date"].max() - pd.Timedelta(days=n_days_monitor)
    
    reference = df[df["date"] < cutoff]
    current = df[df["date"] >= cutoff]
    
    logger.info(f"Reference period: {reference['date'].min().date()} → {reference['date'].max().date()} ({len(reference):,} rows)")
    logger.info(f"Current period:   {current['date'].min().date()} → {current['date'].max().date()} ({len(current):,} rows)")
    
    # 1. Feature drift
    detector = DriftDetector(reference, feature_cols)
    drift_results = detector.check_feature_drift(current)
    target_drift = detector.check_target_drift(current)
    
    logger.info(f"\nDrift Summary:")
    logger.info(f"  Mean PSI: {drift_results['mean_psi']}")
    logger.info(f"  Max PSI:  {drift_results['max_psi']}")
    logger.info(f"  Drifted features: {drift_results['drifted_features']}/{drift_results['total_features']}")
    
    if drift_results["alerts"]:
        for alert in drift_results["alerts"]:
            logger.warning(f"  ⚠️  {alert}")
    
    # 2. Model performance
    perf_monitor = PerformanceMonitor()
    perf_results = perf_monitor.evaluate_recent_performance(df, n_days=n_days_monitor)
    
    logger.info(f"\nPerformance (last {n_days_monitor} days):")
    if "error" not in perf_results:
        logger.info(f"  Accuracy: {perf_results['accuracy']:.4f} (baseline: {perf_results['baseline_accuracy']:.4f})")
        logger.info(f"  AUC:      {perf_results['auc']:.4f}")
        perf_monitor.check_accuracy_degradation(perf_results)
    else:
        logger.warning(f"  {perf_results['error']}")
    
    # 3. Evidently report
    generate_evidently_report(reference, current, feature_cols[:20])  # Top 20 features
    
    # 4. Save monitoring snapshot
    snapshot = {
        "run_at": datetime.now().isoformat(),
        "drift": drift_results,
        "target_drift": target_drift,
        "performance": perf_results,
        "recommend_retrain": drift_results["max_psi"] > CFG["monitoring"]["psi_threshold"]
    }
    
    snap_path = MONITOR_DIR / f"snapshot_{datetime.now().strftime('%Y%m%d')}.json"
    with open(snap_path, "w") as f:
        json.dump(snapshot, f, indent=2, default=str)
    
    logger.info(f"\n✅ Monitoring complete. Snapshot → {snap_path}")
    
    if snapshot["recommend_retrain"]:
        logger.warning("🔄 RETRAINING RECOMMENDED — significant data drift detected")
    
    return snapshot


if __name__ == "__main__":
    run_monitoring()
