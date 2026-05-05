"""
Model Serving API
==================
FastAPI REST API for real-time and batch predictions.

Industry patterns implemented:
  - Redis caching (avoid recomputing features for same ticker/date)
  - Prometheus metrics (latency, throughput, error rate)
  - Health checks (liveness + readiness probes, like Kubernetes)
  - Model versioning via metadata
  - Input validation with Pydantic
  - Async endpoints for performance
  - Structured logging for observability

This is the pattern used at Uber (Michelangelo), Netflix, Airbnb for
serving ML models — FastAPI + Redis + Prometheus is the open-source
equivalent of their proprietary stacks.
"""

import json
import pickle
import time
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, List
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
# import redis
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from prometheus_client import Counter, Histogram, Gauge, make_asgi_app
from loguru import logger
import yaml

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

MODEL_DIR = ROOT / CFG["paths"]["model_registry"]


# ════════════════════════════════════════════════════════════════
#  PROMETHEUS METRICS
#  These are scraped by Prometheus and visualized in Grafana
# ════════════════════════════════════════════════════════════════

PREDICTIONS_TOTAL = Counter(
    "stocksense_predictions_total",
    "Total number of predictions served",
    ["ticker", "direction"]
)

PREDICTION_LATENCY = Histogram(
    "stocksense_prediction_latency_seconds",
    "Time to generate a prediction",
    buckets=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
)

CACHE_HITS = Counter("stocksense_cache_hits_total", "Redis cache hits")
CACHE_MISSES = Counter("stocksense_cache_misses_total", "Redis cache misses")
MODEL_CONFIDENCE = Histogram("stocksense_model_confidence", "Distribution of prediction probabilities")
API_ERRORS = Counter("stocksense_api_errors_total", "API errors", ["error_type"])
MODEL_LOAD_STATUS = Gauge("stocksense_model_loaded", "Whether champion model is loaded (1=yes)")


# ════════════════════════════════════════════════════════════════
#  MODEL LOADER
# ════════════════════════════════════════════════════════════════

class ModelStore:
    """
    Singleton model store — loads model once at startup,
    reloads on demand when new champion is promoted.
    """
    
    def __init__(self):
        self.model = None
        self.metadata = None
        self.feature_columns = None
        self.loaded_at = None
    
    def load(self) -> bool:
        champion_path = MODEL_DIR / "champion_model.pkl"
        meta_path = MODEL_DIR / "champion_metadata.json"
        
        if not champion_path.exists():
            logger.warning("No champion model found. Run training first.")
            MODEL_LOAD_STATUS.set(0)
            return False
        
        with open(champion_path, "rb") as f:
            self.model = pickle.load(f)
        
        if meta_path.exists():
            with open(meta_path) as f:
                self.metadata = json.load(f)
            self.feature_columns = self.metadata.get("feature_columns", [])
        
        self.loaded_at = datetime.now()
        MODEL_LOAD_STATUS.set(1)
        logger.info(f"✅ Champion model loaded: {self.metadata.get('model_type', 'unknown')}")
        logger.info(f"   Test AUC: {self.metadata.get('test_auc', 'N/A')}")
        return True
    
    def predict(self, features: pd.DataFrame) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not loaded")
        
        # Ensure correct feature order
        if self.feature_columns:
            available = [c for c in self.feature_columns if c in features.columns]
            features = features[available]
        
        return self.model.predict_proba(features)[:, 1]  # P(up)


model_store = ModelStore()


# ════════════════════════════════════════════════════════════════
#  REDIS CACHE
# ════════════════════════════════════════════════════════════════

def get_redis():
    try:
        r = redis.Redis(host="localhost", port=6379, db=0, decode_responses=True)
        r.ping()
        return r
    except Exception:
        logger.warning("Redis unavailable — running without cache")
        return None


redis_client = get_redis()


def cache_key(ticker: str, date: str) -> str:
    model_version = model_store.metadata.get("trained_at", "v0")[:10] if model_store.metadata else "v0"
    return f"prediction:{model_version}:{ticker}:{date}"


# ════════════════════════════════════════════════════════════════
#  PYDANTIC SCHEMAS
# ════════════════════════════════════════════════════════════════

class PredictionRequest(BaseModel):
    ticker: str = Field(..., example="AAPL", description="Stock ticker symbol")
    date: Optional[str] = Field(None, example="2024-01-15", description="Date (default: latest)")


class BatchPredictionRequest(BaseModel):
    tickers: List[str] = Field(..., example=["AAPL", "MSFT", "GOOGL"])


class PredictionResponse(BaseModel):
    ticker: str
    date: str
    prediction: str          # "UP" or "DOWN"
    probability: float        # P(UP)
    confidence: str           # "HIGH", "MEDIUM", "LOW"
    model_version: str
    cached: bool
    latency_ms: float


class ModelInfoResponse(BaseModel):
    model_type: str
    trained_at: str
    test_auc: float
    test_accuracy: float
    features_count: int
    model_status: str


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    cache_available: bool
    timestamp: str


# ════════════════════════════════════════════════════════════════
#  APP LIFECYCLE
# ════════════════════════════════════════════════════════════════

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup, cleanup on shutdown."""
    logger.info("🚀 Starting StockSense API...")
    model_store.load()
    yield
    logger.info("Shutting down StockSense API")


app = FastAPI(
    title="StockSense MLOps API",
    description="Stock direction prediction with daily-refreshed ML models",
    version="1.0.0",
    lifespan=lifespan
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount Prometheus metrics endpoint
metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)


# ════════════════════════════════════════════════════════════════
#  PREDICTION LOGIC
# ════════════════════════════════════════════════════════════════

def get_prediction_for_ticker(ticker: str, date: Optional[str] = None) -> dict:
    """Core prediction logic — shared between single and batch endpoints."""
    from src.features.engineer import load_features, FEATURE_COLUMNS
    
    start_time = time.time()
    
    # Load features
    features_df = load_features()
    
    if date:
        row = features_df[
            (features_df["ticker"] == ticker) & 
            (features_df["date"].astype(str).str[:10] == date)
        ]
    else:
        row = features_df[features_df["ticker"] == ticker].sort_values("date").tail(1)
    
    if row.empty:
        raise ValueError(f"No features found for {ticker}")
    
    feature_cols = [c for c in FEATURE_COLUMNS if c in row.columns]
    X = row[feature_cols]
    
    prob_up = float(model_store.predict(X)[0])
    direction = "UP" if prob_up >= 0.5 else "DOWN"
    
    # Confidence tiers
    conf_score = abs(prob_up - 0.5) * 2  # 0 = no confidence, 1 = max confidence
    if conf_score > 0.4:
        confidence = "HIGH"
    elif conf_score > 0.2:
        confidence = "MEDIUM"
    else:
        confidence = "LOW"
    
    pred_date = str(row["date"].values[0])[:10]
    latency = (time.time() - start_time) * 1000
    
    return {
        "ticker": ticker,
        "date": pred_date,
        "prediction": direction,
        "probability": round(prob_up, 4),
        "confidence": confidence,
        "model_version": model_store.metadata.get("trained_at", "unknown")[:10] if model_store.metadata else "unknown",
        "cached": False,
        "latency_ms": round(latency, 2)
    }


# ════════════════════════════════════════════════════════════════
#  API ENDPOINTS
# ════════════════════════════════════════════════════════════════

@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Liveness probe — is the API alive?"""
    return {
        "status": "healthy",
        "model_loaded": model_store.model is not None,
        "cache_available": redis_client is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/ready", tags=["System"])
async def readiness_check():
    """Readiness probe — is the API ready to serve traffic?"""
    if model_store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ready"}


@app.get("/model/info", response_model=ModelInfoResponse, tags=["Model"])
async def model_info():
    """Get current champion model metadata."""
    if not model_store.metadata:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    m = model_store.metadata
    return {
        "model_type": m.get("model_type", "unknown"),
        "trained_at": m.get("trained_at", "unknown"),
        "test_auc": m.get("test_auc", 0),
        "test_accuracy": m.get("test_accuracy", 0),
        "features_count": len(m.get("feature_columns", [])),
        "model_status": m.get("stage", "unknown")
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict(request: PredictionRequest):
    """
    Get prediction for a single ticker.
    Checks Redis cache first — only runs model if cache miss.
    """
    if model_store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded. Run training pipeline first.")
    
    ticker = request.ticker.upper()
    date = request.date
    
    # Check cache
    if redis_client and not date:
        key = cache_key(ticker, "latest")
        cached = redis_client.get(key)
        if cached:
            CACHE_HITS.inc()
            result = json.loads(cached)
            result["cached"] = True
            return result
        CACHE_MISSES.inc()
    
    try:
        with PREDICTION_LATENCY.time():
            result = get_prediction_for_ticker(ticker, date)
    except ValueError as e:
        API_ERRORS.labels(error_type="not_found").inc()
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        API_ERRORS.labels(error_type="prediction_error").inc()
        logger.error(f"Prediction error for {ticker}: {e}")
        raise HTTPException(status_code=500, detail="Prediction failed")
    
    # Update metrics
    PREDICTIONS_TOTAL.labels(ticker=ticker, direction=result["prediction"]).inc()
    MODEL_CONFIDENCE.observe(result["probability"])
    
    # Cache result
    if redis_client and not date:
        key = cache_key(ticker, "latest")
        redis_client.setex(key, CFG["serving"]["cache_ttl_seconds"], json.dumps(result))
    
    return result


@app.post("/predict/batch", tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """Get predictions for multiple tickers at once."""
    if model_store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    results = []
    errors = []
    
    for ticker in request.tickers:
        try:
            result = get_prediction_for_ticker(ticker.upper())
            results.append(result)
        except Exception as e:
            errors.append({"ticker": ticker, "error": str(e)})
    
    return {
        "predictions": results,
        "errors": errors,
        "total": len(results),
        "timestamp": datetime.now().isoformat()
    }


@app.get("/predict/all", tags=["Predictions"])
async def predict_all():
    """Get latest predictions for all tickers in universe."""
    if model_store.model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    tickers = CFG["universe"]["tickers"]
    results = []
    
    for ticker in tickers:
        try:
            result = get_prediction_for_ticker(ticker)
            results.append(result)
        except Exception as e:
            logger.warning(f"Skipping {ticker}: {e}")
    
    return {
        "predictions": sorted(results, key=lambda x: x["probability"], reverse=True),
        "timestamp": datetime.now().isoformat(),
        "model_version": model_store.metadata.get("trained_at", "unknown")[:10] if model_store.metadata else "unknown"
    }


@app.post("/model/reload", tags=["Model"])
async def reload_model(background_tasks: BackgroundTasks):
    """
    Trigger model reload from registry.
    Called after training pipeline promotes new champion.
    """
    background_tasks.add_task(model_store.load)
    return {"status": "Model reload initiated"}


@app.get("/tickers", tags=["Data"])
async def list_tickers():
    """List all tracked tickers."""
    return {"tickers": CFG["universe"]["tickers"]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "src.serving.api:app",
        host=CFG["serving"]["host"],
        port=CFG["serving"]["port"],
        reload=False,
        workers=1,
        log_level="info"
    )
