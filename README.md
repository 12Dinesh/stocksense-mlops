# 📈 StockSense MLOps — End-to-End ML/AI System

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        DATA LAYER                               │
│  Yahoo Finance API → Raw Parquet → Feature Store (Feast-lite)  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   ORCHESTRATION LAYER                           │
│              Apache Airflow (DAGs, scheduling)                  │
│         Daily: ingest → features → train → evaluate → deploy   │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      ML LAYER                                   │
│  MLflow Tracking → LightGBM/XGBoost → Optuna Tuning            │
│  Model Registry → Champion/Challenger evaluation                │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                   SERVING LAYER                                 │
│  FastAPI (REST) → Redis Cache → Prometheus Metrics              │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                  MONITORING LAYER                               │
│  Evidently AI (drift) → Grafana Dashboard → Alerting           │
└──────────────────────────┬──────────────────────────────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                    UI LAYER                                     │
│         Streamlit Dashboard (predictions + monitoring)          │
└─────────────────────────────────────────────────────────────────┘
```

## Tools \& Why They're Industry Standard

|Layer|Tool|Industry Equivalent|
|-|-|-|
|Orchestration|Apache Airflow|Airflow (used by Airbnb, Twitter, Lyft)|
|Experiment Tracking|MLflow|MLflow (Databricks), W\&B|
|Feature Store|Custom + Parquet|Feast, Tecton, Hopsworks|
|Model Training|LightGBM + Optuna|Same tools at industry scale|
|Serving|FastAPI + Uvicorn|Same pattern as Netflix, Uber|
|Monitoring|Evidently AI|Evidently, Arize, WhyLabs|
|Containerization|Docker Compose|Docker/K8s in production|
|Metrics|Prometheus + Grafana|Industry standard observability|
|Data Format|Parquet + DuckDB|Same as Spark/BigQuery output|
|UI|Streamlit|Internal ML dashboards|

## Quick Start

```bash
pip install -r requirements.txt
docker-compose up -d          # Redis + Prometheus + Grafana
python setup.py               # Initialize DB, feature store
airflow standalone            # Start scheduler (separate terminal)
python -m src.ingestion.run   # First data pull
python -m src.training.run    # Train first model
uvicorn src.serving.api:app   # Start API server
streamlit run dashboard.py    # Launch UI  







\# StockSense MLOps



End-to-end ML system for stock direction prediction.



\## What This Does

\- Downloads real stock data (AAPL, MSFT, NVDA, GOOGL etc.)

\- Engineers 46 technical indicators as ML features

\- Trains LightGBM and XGBoost models with Optuna tuning

\- Serves predictions via FastAPI REST API

\- Monitors model drift with PSI scoring

\- Displays everything in a Streamlit web dashboard



\## Tech Stack

\- Data: Yahoo Finance, Parquet, DuckDB

\- ML: LightGBM, XGBoost, Optuna, SHAP, MLflow

\- Serving: FastAPI, Uvicorn

\- Dashboard: Streamlit, Plotly

\- Monitoring: PSI drift detection



\## How to Run

1\. pip install -r requirements.txt

2\. py -m src.ingestion.ingest

3\. py -m src.features.engineer

4\. py -m src.training.train

5\. uvicorn src.serving.api:app --port 8000

6\. streamlit run dashboard.py



\## Built By

Dinesh Saraswat
```

