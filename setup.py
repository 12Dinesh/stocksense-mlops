"""
setup.py — One-shot system initialization
==========================================
Run this once to set up the entire MLOps system.
After running this, you have a fully functional end-to-end ML system.
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent


def print_banner():
    print("""
╔══════════════════════════════════════════════════════════╗
║           StockSense MLOps — System Setup                ║
║   End-to-End ML: Ingest → Features → Train → Serve      ║
╚══════════════════════════════════════════════════════════╝
    """)


def check_python():
    version = sys.version_info
    assert version.major == 3 and version.minor >= 9, f"Python 3.9+ required, got {sys.version}"
    print(f"✓ Python {version.major}.{version.minor}.{version.micro}")


def create_dirs():
    dirs = [
        "data/raw", "data/processed", "data/features",
        "models/registry", "logs", "mlruns",
        "monitoring_reports", "notebooks"
    ]
    for d in dirs:
        (ROOT / d).mkdir(parents=True, exist_ok=True)
    
    # Create __init__.py files
    for pkg in ["src", "src/ingestion", "src/features", "src/training", "src/serving", "src/monitoring"]:
        init = ROOT / pkg / "__init__.py"
        init.parent.mkdir(parents=True, exist_ok=True)
        init.touch()
    
    print("✓ Directory structure created")


def setup_airflow():
    """Initialize Airflow with minimal config."""
    os.environ["AIRFLOW_HOME"] = str(ROOT / "airflow")
    
    airflow_cfg = ROOT / "airflow" / "airflow.cfg"
    if not airflow_cfg.exists():
        # Create minimal airflow config
        cfg_content = f"""
[core]
dags_folder = {ROOT}/airflow/dags
executor = LocalExecutor
sql_alchemy_conn = sqlite:///{ROOT}/airflow/airflow.db
load_examples = False

[webserver]
web_server_port = 8080

[scheduler]
dag_dir_list_interval = 30
"""
        airflow_cfg.parent.mkdir(parents=True, exist_ok=True)
        airflow_cfg.write_text(cfg_content)
    
    print("✓ Airflow configured")
    print("  → Start with: AIRFLOW_HOME=./airflow airflow standalone")


def run_quick_pipeline():
    """Run the pipeline to get a working model."""
    print("\n" + "="*50)
    print("Running initial pipeline...")
    print("="*50)
    
    steps = [
        ("Data Ingestion", "src.ingestion.ingest", "run_ingestion"),
        ("Feature Engineering", "src.features.engineer", "run_feature_engineering"),
        ("Model Training", "src.training.train", "run_training"),
        ("Monitoring", "src.monitoring.monitor", "run_monitoring"),
    ]
    
    for name, module, func in steps:
        print(f"\n→ {name}...")
        try:
            exec_code = f"from {module} import {func}; {func}()"
            result = subprocess.run(
                [sys.executable, "-c", exec_code],
                capture_output=False,
                cwd=ROOT
            )
            if result.returncode == 0:
                print(f"  ✓ {name} complete")
            else:
                print(f"  ⚠ {name} had issues (see output above)")
        except Exception as e:
            print(f"  ✗ {name} failed: {e}")
    
    return True


def print_next_steps():
    print("""
╔══════════════════════════════════════════════════════════╗
║                    SETUP COMPLETE! 🎉                    ║
╠══════════════════════════════════════════════════════════╣
║                                                          ║
║  Start all services:                                     ║
║                                                          ║
║  1. Infrastructure:                                      ║
║     docker-compose up -d                                 ║
║                                                          ║
║  2. API Server (terminal 1):                             ║
║     uvicorn src.serving.api:app --port 8000              ║
║                                                          ║
║  3. Airflow (terminal 2):                                ║
║     AIRFLOW_HOME=./airflow airflow standalone            ║
║                                                          ║
║  4. Dashboard (terminal 3):                              ║
║     streamlit run dashboard.py                           ║
║                                                          ║
║  Service URLs:                                           ║
║  • Dashboard:  http://localhost:8501                     ║
║  • API Docs:   http://localhost:8000/docs                ║
║  • MLflow:     http://localhost:5001                     ║
║  • Airflow:    http://localhost:8080                     ║
║  • Grafana:    http://localhost:3000                     ║
║  • Prometheus: http://localhost:9090                     ║
║                                                          ║
║  Daily refresh: Airflow DAG runs at 6 AM Mon-Fri        ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    print_banner()
    
    print("Step 1: Checking environment...")
    check_python()
    
    print("\nStep 2: Creating project structure...")
    create_dirs()
    
    print("\nStep 3: Configuring Airflow...")
    setup_airflow()
    
    print("\nStep 4: Running initial ML pipeline...")
    print("(This downloads data, trains the model — takes ~5-10 mins)")
    
    run_quick = input("\nRun the full pipeline now? [Y/n]: ").strip().lower()
    if run_quick != "n":
        run_quick_pipeline()
    
    print_next_steps()
