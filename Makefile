# ══════════════════════════════════════════════════════════════
#  StockSense MLOps — Makefile
#  Usage: make <target>
# ══════════════════════════════════════════════════════════════

PYTHON = python
AIRFLOW_HOME = $(PWD)/airflow

.PHONY: help setup install infra stop ingest features train monitor api dashboard test clean

help:
	@echo ""
	@echo "StockSense MLOps Commands:"
	@echo ""
	@echo "  make install     → Install Python dependencies"
	@echo "  make infra       → Start Docker services (Redis, MLflow, Prometheus, Grafana)"
	@echo "  make stop        → Stop Docker services"
	@echo "  make pipeline    → Run full ML pipeline (ingest → features → train → monitor)"
	@echo "  make ingest      → Download market data"
	@echo "  make features    → Build feature store"
	@echo "  make train       → Train models (with MLflow + Optuna)"
	@echo "  make monitor     → Run drift detection"
	@echo "  make api         → Start serving API"
	@echo "  make airflow     → Start Airflow scheduler"
	@echo "  make dashboard   → Launch Streamlit UI"
	@echo "  make test        → Run test suite"
	@echo "  make clean       → Remove generated data and models"
	@echo ""

install:
	pip install -r requirements.txt

infra:
	docker-compose up -d
	@echo "Services started:"
	@echo "  MLflow:     http://localhost:5001"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000"
	@echo "  Redis:      localhost:6379"

stop:
	docker-compose down

pipeline: ingest features train monitor
	@echo "✅ Full pipeline complete"

ingest:
	$(PYTHON) -c "from src.ingestion.ingest import run_ingestion; run_ingestion()"

features:
	$(PYTHON) -c "from src.features.engineer import run_feature_engineering; run_feature_engineering()"

train:
	$(PYTHON) -c "from src.training.train import run_training; run_training()"

monitor:
	$(PYTHON) -c "from src.monitoring.monitor import run_monitoring; run_monitoring()"

api:
	uvicorn src.serving.api:app --host 0.0.0.0 --port 8000 --reload

airflow:
	AIRFLOW_HOME=$(AIRFLOW_HOME) airflow standalone

dashboard:
	streamlit run dashboard.py --server.port 8501

test:
	pytest tests/ -v --tb=short

clean:
	rm -rf data/raw/*.parquet
	rm -rf data/features/*.parquet
	rm -rf models/registry/*.pkl
	rm -rf models/registry/*.json
	rm -rf monitoring_reports/*.json
	rm -rf monitoring_reports/*.html
	rm -rf logs/*.log
	@echo "Cleaned generated files (mlruns preserved)"
