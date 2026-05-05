"""
Test Suite — StockSense MLOps
================================
Industry-grade tests covering:
  - Data quality and schema validation
  - Feature engineering correctness
  - Model API contract tests
  - Data leakage detection
  - Prediction sanity checks

Run: pytest tests/ -v --tb=short
"""

import sys
import json
import pickle
from pathlib import Path
from datetime import datetime, timedelta

import pytest
import pandas as pd
import numpy as np
import yaml

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)


# ════════════════════════════════════════════════════════════════
#  FIXTURES
# ════════════════════════════════════════════════════════════════

@pytest.fixture
def sample_ohlcv():
    """Generate synthetic OHLCV data for testing."""
    np.random.seed(42)
    n = 252  # 1 year of trading days
    
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    tickers = ["AAPL", "MSFT", "GOOGL"]
    
    rows = []
    for ticker in tickers:
        price = 100
        prices = []
        for _ in range(n):
            price *= (1 + np.random.normal(0, 0.015))
            prices.append(price)
        
        for i, (date, close) in enumerate(zip(dates, prices)):
            high = close * (1 + abs(np.random.normal(0, 0.01)))
            low = close * (1 - abs(np.random.normal(0, 0.01)))
            open_ = prices[i-1] if i > 0 else close
            volume = int(np.random.uniform(1e6, 1e7))
            
            rows.append({
                "date": date,
                "ticker": ticker,
                "open": open_,
                "high": high,
                "low": low,
                "close": close,
                "volume": volume
            })
    
    return pd.DataFrame(rows)


@pytest.fixture
def sample_features(sample_ohlcv):
    """Build features from sample OHLCV data."""
    from src.features.engineer import build_feature_set
    return build_feature_set(sample_ohlcv)


# ════════════════════════════════════════════════════════════════
#  DATA QUALITY TESTS
# ════════════════════════════════════════════════════════════════

class TestDataQuality:
    
    def test_ohlcv_schema(self, sample_ohlcv):
        """Data must have all required columns."""
        required = ["date", "open", "high", "low", "close", "volume", "ticker"]
        for col in required:
            assert col in sample_ohlcv.columns, f"Missing column: {col}"
    
    def test_high_gte_low(self, sample_ohlcv):
        """High must always be >= Low."""
        assert (sample_ohlcv["high"] >= sample_ohlcv["low"]).all(), \
            "Found rows where high < low"
    
    def test_positive_prices(self, sample_ohlcv):
        """All prices must be positive."""
        price_cols = ["open", "high", "low", "close"]
        assert (sample_ohlcv[price_cols] > 0).all().all(), \
            "Negative prices found"
    
    def test_positive_volume(self, sample_ohlcv):
        """Volume must be non-negative."""
        assert (sample_ohlcv["volume"] >= 0).all(), "Negative volume found"
    
    def test_close_within_range(self, sample_ohlcv):
        """Close must be between Low and High."""
        assert ((sample_ohlcv["close"] <= sample_ohlcv["high"]) & 
                (sample_ohlcv["close"] >= sample_ohlcv["low"])).all(), \
            "Close price outside High-Low range"
    
    def test_no_duplicate_date_ticker(self, sample_ohlcv):
        """No duplicate (date, ticker) combinations."""
        dups = sample_ohlcv.duplicated(subset=["date", "ticker"])
        assert not dups.any(), f"Found {dups.sum()} duplicate rows"
    
    def test_data_quality_checker(self, sample_ohlcv):
        """DataQualityChecker should pass on clean data."""
        from src.ingestion.ingest import DataQualityChecker
        checker = DataQualityChecker(sample_ohlcv)
        # May fail freshness check (synthetic data) but schema should pass
        result = checker.run_all()
        assert checker.passed >= 6, f"Too many quality failures: {checker.failed}"


# ════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING TESTS
# ════════════════════════════════════════════════════════════════

class TestFeatureEngineering:
    
    def test_features_not_empty(self, sample_features):
        """Feature set must not be empty."""
        assert len(sample_features) > 0
    
    def test_target_is_binary(self, sample_features):
        """Target must be binary (0 or 1)."""
        unique_vals = sample_features["target"].dropna().unique()
        assert set(unique_vals).issubset({0, 1}), \
            f"Non-binary target values: {unique_vals}"
    
    def test_rsi_range(self, sample_features):
        """RSI must be between 0 and 100."""
        rsi = sample_features["rsi_14"].dropna()
        assert (rsi >= 0).all() and (rsi <= 100).all(), \
            "RSI out of [0, 100] range"
    
    def test_no_future_leakage(self, sample_features):
        """
        CRITICAL: Test for data leakage.
        Features must not include any future information.
        This is the most important test in financial ML.
        """
        from src.features.engineer import FEATURE_COLUMNS
        
        # All feature columns should be computable from past data only
        # The target (future return) should NOT appear in feature columns
        for col in FEATURE_COLUMNS:
            assert col != "target", "Target in feature columns = DATA LEAKAGE"
            assert col != "future_return", "Future return in features = DATA LEAKAGE"
            assert "future" not in col.lower(), f"Suspicious column name: {col}"
    
    def test_feature_count(self, sample_features):
        """Should have a reasonable number of features."""
        from src.features.engineer import FEATURE_COLUMNS
        available = [c for c in FEATURE_COLUMNS if c in sample_features.columns]
        assert len(available) >= 20, f"Too few features: {len(available)}"
    
    def test_feature_store_save_load(self, sample_features, tmp_path):
        """Feature store save/load round-trip."""
        from src.features.engineer import save_features, load_features
        import pyarrow.parquet as pq
        
        # Monkeypatch FEATURE_DIR to tmp_path
        import src.features.engineer as fe
        original_dir = fe.FEATURE_DIR
        fe.FEATURE_DIR = tmp_path
        
        try:
            save_features(sample_features, version="test")
            loaded = load_features(version="test")
            assert len(loaded) == len(sample_features)
        finally:
            fe.FEATURE_DIR = original_dir
    
    def test_bollinger_band_position(self, sample_features):
        """BB position should mostly be in [0, 1] range."""
        bb_pos = sample_features["bb_position"].dropna()
        # Allow some outliers (price can spike outside bands)
        in_range = ((bb_pos >= -0.5) & (bb_pos <= 1.5)).mean()
        assert in_range > 0.9, f"Too many BB position outliers: {1-in_range:.1%}"
    
    def test_volume_ratio_positive(self, sample_features):
        """Volume ratio must be positive."""
        vr = sample_features["volume_ratio"].dropna()
        assert (vr > 0).all(), "Negative volume ratio found"
    
    def test_target_balance(self, sample_features):
        """Class balance shouldn't be too extreme."""
        rate = sample_features["target"].mean()
        assert 0.3 < rate < 0.7, \
            f"Severely imbalanced target: {rate:.2%} positive rate"


# ════════════════════════════════════════════════════════════════
#  MODEL TESTS
# ════════════════════════════════════════════════════════════════

class TestModelBehavior:
    
    def test_psi_zero_same_distribution(self):
        """PSI should be 0 (or near-0) when comparing identical distributions."""
        from src.monitoring.monitor import compute_psi
        
        np.random.seed(42)
        data = np.random.normal(0, 1, 1000)
        psi = compute_psi(data, data)
        assert psi < 0.01, f"PSI on identical distributions should be ~0, got {psi}"
    
    def test_psi_high_different_distribution(self):
        """PSI should be high when distributions are very different."""
        from src.monitoring.monitor import compute_psi
        
        np.random.seed(42)
        ref = np.random.normal(0, 1, 1000)
        shifted = np.random.normal(3, 1, 1000)  # Very different mean
        psi = compute_psi(ref, shifted)
        assert psi > 0.2, f"PSI on different distributions should be >0.2, got {psi}"
    
    def test_train_test_split_no_overlap(self, sample_features):
        """Train and test sets must not overlap (time series integrity)."""
        from src.training.train import prepare_train_test
        
        X_train, X_val, X_test, y_train, y_val, y_test, _ = prepare_train_test(sample_features)
        
        # In a proper time split, there should be no index overlap
        assert len(set(X_train.index) & set(X_test.index)) == 0, \
            "Train/test index overlap detected — potential data leakage"
        assert len(set(X_val.index) & set(X_test.index)) == 0, \
            "Val/test index overlap detected"
    
    def test_prediction_probability_range(self, sample_features):
        """Model predictions must be valid probabilities [0, 1]."""
        model_path = ROOT / CFG["paths"]["model_registry"] / "champion_model.pkl"
        
        if not model_path.exists():
            pytest.skip("No trained model — run training first")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        meta_path = ROOT / CFG["paths"]["model_registry"] / "champion_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        
        feature_cols = [c for c in meta.get("feature_columns", []) if c in sample_features.columns]
        X = sample_features[feature_cols].dropna().head(100)
        
        probs = model.predict_proba(X)
        assert probs.shape[1] == 2, "Expected binary probabilities"
        assert (probs >= 0).all(), "Negative probabilities"
        assert (probs <= 1).all(), "Probabilities > 1"
        assert np.allclose(probs.sum(axis=1), 1.0), "Probabilities don't sum to 1"
    
    def test_model_consistency(self, sample_features):
        """Same input must always produce same output (determinism)."""
        model_path = ROOT / CFG["paths"]["model_registry"] / "champion_model.pkl"
        
        if not model_path.exists():
            pytest.skip("No trained model — run training first")
        
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        
        meta_path = ROOT / CFG["paths"]["model_registry"] / "champion_metadata.json"
        with open(meta_path) as f:
            meta = json.load(f)
        
        feature_cols = [c for c in meta.get("feature_columns", []) if c in sample_features.columns]
        X = sample_features[feature_cols].dropna().head(10)
        
        probs1 = model.predict_proba(X)
        probs2 = model.predict_proba(X)
        
        np.testing.assert_array_equal(probs1, probs2, "Model is non-deterministic!")


# ════════════════════════════════════════════════════════════════
#  API TESTS (integration — require running API)
# ════════════════════════════════════════════════════════════════

class TestAPIIntegration:
    """
    These tests require the API to be running.
    Skip gracefully if not available.
    """
    
    @pytest.fixture(autouse=True)
    def check_api(self):
        try:
            import httpx
            resp = httpx.get("http://localhost:8000/health", timeout=2)
            if resp.status_code != 200:
                pytest.skip("API not running")
        except Exception:
            pytest.skip("API not running — start with: uvicorn src.serving.api:app")
    
    def test_health_check(self):
        import httpx
        resp = httpx.get("http://localhost:8000/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_prediction_schema(self):
        import httpx
        resp = httpx.post("http://localhost:8000/predict",
                          json={"ticker": "AAPL"})
        
        if resp.status_code == 503:
            pytest.skip("Model not loaded")
        
        assert resp.status_code == 200
        data = resp.json()
        assert "ticker" in data
        assert "prediction" in data
        assert data["prediction"] in ["UP", "DOWN"]
        assert 0 <= data["probability"] <= 1
        assert data["confidence"] in ["HIGH", "MEDIUM", "LOW"]
    
    def test_batch_prediction(self):
        import httpx
        resp = httpx.post("http://localhost:8000/predict/batch",
                          json={"tickers": ["AAPL", "MSFT"]})
        assert resp.status_code in [200, 503]
        
        if resp.status_code == 200:
            data = resp.json()
            assert "predictions" in data
            assert "errors" in data
    
    def test_invalid_ticker(self):
        import httpx
        resp = httpx.post("http://localhost:8000/predict",
                          json={"ticker": "INVALID_TICKER_XYZ"})
        assert resp.status_code in [404, 503], \
            "Expected 404 for unknown ticker"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
