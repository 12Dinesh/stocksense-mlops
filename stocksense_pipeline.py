"""
Airflow DAG — Daily ML Pipeline
================================
Orchestrates the full ML lifecycle on a daily schedule:

  ingest → validate → features → train (weekly) → monitor → serve

Industry standard: Apache Airflow is used at Airbnb, Lyft, Twitter,
and hundreds of companies to orchestrate data + ML pipelines.
It gives you: dependency management, retry logic, alerting,
backfilling, and visual DAG monitoring in its UI.

Access Airflow UI: http://localhost:8080 (after `airflow standalone`)
"""

from datetime import datetime, timedelta
from pathlib import Path
import sys

from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.utils.dates import days_ago
from airflow.models import Variable

# ── Default args — applied to all tasks ──────────────────────────
default_args = {
    "owner": "mlops-team",
    "depends_on_past": False,
    "email_on_failure": False,       # Set to True + configure SMTP for real alerts
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(hours=2),
}

ROOT = Path(__file__).parent.parent


# ════════════════════════════════════════════════════════════════
#  TASK FUNCTIONS
# ════════════════════════════════════════════════════════════════

def task_ingest(**context):
    """Download fresh market data from Yahoo Finance."""
    sys.path.insert(0, str(ROOT))
    from src.ingestion.ingest import run_ingestion
    
    out_path = run_ingestion()
    
    # XCom: pass file path to next task (Airflow's inter-task data sharing)
    context["task_instance"].xcom_push(key="raw_data_path", value=str(out_path))
    return str(out_path)


def task_validate_data(**context):
    """Run data quality checks on raw data."""
    sys.path.insert(0, str(ROOT))
    from src.ingestion.ingest import load_latest_raw
    from src.ingestion.ingest import DataQualityChecker
    
    df = load_latest_raw()
    checker = DataQualityChecker(df)
    quality_ok = checker.run_all()
    
    if not quality_ok:
        # In production: could trigger alert, send to dead-letter queue, etc.
        # Here we proceed with a warning
        print("WARNING: Data quality issues detected")
    
    return quality_ok


def task_build_features(**context):
    """Compute and store feature set."""
    sys.path.insert(0, str(ROOT))
    from src.features.engineer import run_feature_engineering
    
    df = run_feature_engineering()
    return len(df)


def task_check_train_needed(**context):
    """
    Branch logic: should we retrain today?
    Train weekly OR if monitoring detected drift.
    Returns name of next task to execute.
    """
    import json
    from pathlib import Path
    
    # Check monitoring snapshot for drift alert
    monitor_dir = ROOT / "monitoring_reports"
    snapshots = sorted(monitor_dir.glob("snapshot_*.json"))
    
    if snapshots:
        with open(snapshots[-1]) as f:
            snapshot = json.load(f)
        
        if snapshot.get("recommend_retrain", False):
            print("Drift detected — triggering retrain")
            return "train_model"
    
    # Weekly training schedule (Monday = 0)
    if datetime.now().weekday() == 0:  # Monday
        print("Weekly training day — triggering retrain")
        return "train_model"
    
    print("No retraining needed today")
    return "skip_training"


def task_train_model(**context):
    """Full ML training pipeline with MLflow tracking."""
    sys.path.insert(0, str(ROOT))
    from src.training.train import run_training
    run_training()


def task_monitor(**context):
    """Run drift detection and performance monitoring."""
    sys.path.insert(0, str(ROOT))
    from src.monitoring.monitor import run_monitoring
    
    snapshot = run_monitoring()
    
    # Push key metrics to XCom for downstream tasks / alerting
    context["task_instance"].xcom_push(
        key="drift_score",
        value=snapshot["drift"]["max_psi"]
    )
    context["task_instance"].xcom_push(
        key="recommend_retrain",
        value=snapshot.get("recommend_retrain", False)
    )
    
    return snapshot


def task_reload_api(**context):
    """
    Signal the serving API to reload the model.
    In production: POST to the /model/reload endpoint.
    """
    import httpx
    
    try:
        resp = httpx.post("http://localhost:8000/model/reload", timeout=10)
        if resp.status_code == 200:
            print("✅ API model reloaded successfully")
        else:
            print(f"⚠️  API reload returned {resp.status_code}")
    except Exception as e:
        print(f"API not available (might not be running): {e}")
        # Don't fail the DAG — API reload is best-effort


def task_send_daily_report(**context):
    """
    Generate and log daily summary report.
    In production: send via email / Slack.
    """
    import json
    from pathlib import Path
    
    monitor_dir = ROOT / "monitoring_reports"
    snapshots = sorted(monitor_dir.glob("snapshot_*.json"))
    
    report = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "pipeline_status": "SUCCESS",
        "drift_score": context["task_instance"].xcom_pull(
            task_ids="monitor", key="drift_score"
        ),
    }
    
    print("=" * 50)
    print("DAILY ML PIPELINE REPORT")
    print("=" * 50)
    for k, v in report.items():
        print(f"  {k}: {v}")
    print("=" * 50)
    
    return report


# ════════════════════════════════════════════════════════════════
#  DAG DEFINITION
# ════════════════════════════════════════════════════════════════

with DAG(
    dag_id="stocksense_daily_pipeline",
    description="Daily ML pipeline: ingest → features → train → monitor → serve",
    default_args=default_args,
    schedule_interval="0 6 * * 1-5",    # 6 AM Mon-Fri (market days)
    start_date=days_ago(1),
    catchup=False,
    max_active_runs=1,                   # Prevent overlapping runs
    tags=["mlops", "production", "stocksense"],
) as dag:

    # ── Documentation ─────────────────────────────────────────────
    dag.doc_md = """
    ## StockSense Daily ML Pipeline
    
    **Schedule**: 6 AM, Monday–Friday
    
    **Flow**:
    1. `ingest` — Pull latest OHLCV data from Yahoo Finance
    2. `validate_data` — Run data quality checks
    3. `build_features` — Compute feature set and update feature store
    4. `check_train_needed` — Branch: retrain today? (weekly + drift-based)
    5. `train_model` (conditional) — Retrain with Optuna + MLflow
    6. `monitor` — Drift detection + performance evaluation
    7. `reload_api` — Refresh serving API with latest model
    8. `daily_report` — Log summary
    
    **Monitoring**: http://localhost:8080
    **API**: http://localhost:8000
    **MLflow**: http://localhost:5001
    """

    # ── Tasks ──────────────────────────────────────────────────────
    start = EmptyOperator(task_id="start")
    
    ingest = PythonOperator(
        task_id="ingest",
        python_callable=task_ingest,
    )
    
    validate = PythonOperator(
        task_id="validate_data",
        python_callable=task_validate_data,
    )
    
    build_features = PythonOperator(
        task_id="build_features",
        python_callable=task_build_features,
    )
    
    check_train = BranchPythonOperator(
        task_id="check_train_needed",
        python_callable=task_check_train_needed,
    )
    
    train = PythonOperator(
        task_id="train_model",
        python_callable=task_train_model,
        execution_timeout=timedelta(hours=3),
    )
    
    skip_train = EmptyOperator(task_id="skip_training")
    
    join = EmptyOperator(
        task_id="join_after_branch",
        trigger_rule="none_failed_min_one_success"  # Proceed regardless of which branch ran
    )
    
    monitor = PythonOperator(
        task_id="monitor",
        python_callable=task_monitor,
        trigger_rule="none_failed_min_one_success",
    )
    
    reload_api = PythonOperator(
        task_id="reload_api",
        python_callable=task_reload_api,
    )
    
    daily_report = PythonOperator(
        task_id="daily_report",
        python_callable=task_send_daily_report,
    )
    
    end = EmptyOperator(task_id="end")
    
    # ── Dependencies (DAG structure) ───────────────────────────────
    start >> ingest >> validate >> build_features >> check_train
    check_train >> [train, skip_train] >> join
    join >> monitor >> reload_api >> daily_report >> end
