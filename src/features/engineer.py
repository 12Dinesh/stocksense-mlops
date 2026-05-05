"""
Feature Engineering Module
===========================
Transforms raw OHLCV data into ML-ready features.

This module implements a lightweight Feature Store pattern:
  - Features computed once, stored as Parquet
  - Feature definitions are versioned via config
  - Train/serve feature parity (same code runs at training and inference)

Industry tools this mirrors: Feast, Tecton, Hopsworks, SageMaker Feature Store.
Key principle: Features must be IDENTICAL at training time and serving time
to avoid training-serving skew (the #1 cause of ML model failures in production).
"""

import sys
from pathlib import Path
from typing import Optional
import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import duckdb
import ta  # Technical Analysis library
import pyarrow as pa
import pyarrow.parquet as pq
import yaml
from loguru import logger

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

FEATURE_DIR = ROOT / CFG["paths"]["feature_store"]
FEATURE_DIR.mkdir(parents=True, exist_ok=True)

logger.add(ROOT / "logs/features.log", rotation="1 day", retention="30 days", level="INFO")


# ════════════════════════════════════════════════════════════════
#  FEATURE DEFINITIONS
#  Each function = one feature group. Versioned in config.yaml.
# ════════════════════════════════════════════════════════════════

def compute_return_features(df: pd.DataFrame) -> pd.DataFrame:
    """Momentum / return-based features."""
    for w in CFG["features"]["price_windows"]:
        df[f"return_{w}d"] = df.groupby("ticker")["close"].pct_change(w)
        df[f"return_{w}d_log"] = np.log1p(df[f"return_{w}d"])
    
    # Overnight gap
    df["gap"] = df.groupby("ticker").apply(
        lambda g: (g["open"] - g["close"].shift(1)) / g["close"].shift(1)
    ).reset_index(level=0, drop=True)
    
    return df


def compute_moving_averages(df: pd.DataFrame) -> pd.DataFrame:
    """SMA, EMA ratios — trend features."""
    for w in CFG["features"]["price_windows"]:
        df[f"sma_{w}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(w).mean()
        )
        df[f"ema_{w}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=w, adjust=False).mean()
        )
        # Price relative to MA (normalized)
        df[f"price_to_sma_{w}"] = df["close"] / df[f"sma_{w}"] - 1
        df[f"price_to_ema_{w}"] = df["close"] / df[f"ema_{w}"] - 1
    
    # Golden/Death cross signal
    df["sma_5_20_cross"] = (df["sma_5"] - df["sma_20"]).apply(np.sign)
    df["sma_10_50_cross"] = (df["sma_10"] - df["sma_50"]).apply(np.sign)
    
    return df


def compute_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volatility and range features."""
    for w in CFG["features"]["volatility_windows"]:
        df[f"volatility_{w}d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().rolling(w).std() * np.sqrt(252)
        )
    
    # High-Low range (intraday volatility proxy)
    df["hl_range"] = (df["high"] - df["low"]) / df["close"]
    df["hl_range_5d_avg"] = df.groupby("ticker")["hl_range"].transform(
        lambda x: x.rolling(5).mean()
    )
    
    # ATR (Average True Range)
    df["true_range"] = df.groupby("ticker").apply(lambda g: pd.concat([
        g["high"] - g["low"],
        (g["high"] - g["close"].shift()).abs(),
        (g["low"] - g["close"].shift()).abs()
    ], axis=1).max(axis=1)).reset_index(level=0, drop=True)
    
    df["atr_14"] = df.groupby("ticker")["true_range"].transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    )
    
    return df


def compute_volume_features(df: pd.DataFrame) -> pd.DataFrame:
    """Volume-based features — institutional activity signals."""
    df["volume_sma_20"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20).mean()
    )
    df["volume_ratio"] = df["volume"] / df["volume_sma_20"]
    df["volume_change"] = df.groupby("ticker")["volume"].pct_change()
    
    # On-Balance Volume (OBV)
    def compute_obv(group):
        direction = np.sign(group["close"].diff())
        obv = (direction * group["volume"]).cumsum()
        return obv
    
    df["obv"] = df.groupby("ticker").apply(compute_obv).reset_index(level=0, drop=True)
    df["obv_sma_20"] = df.groupby("ticker")["obv"].transform(
        lambda x: x.rolling(20).mean()
    )
    df["obv_ratio"] = df["obv"] / (df["obv_sma_20"].abs() + 1)
    
    return df


def compute_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """RSI, MACD, Bollinger Bands — classic TA features."""
    results = []
    
    for ticker in df["ticker"].unique():
        tdf = df[df["ticker"] == ticker].copy().sort_values("date")
        close = tdf["close"]
        
        # RSI
        tdf["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        tdf["rsi_overbought"] = (tdf["rsi_14"] > 70).astype(int)
        tdf["rsi_oversold"] = (tdf["rsi_14"] < 30).astype(int)
        
        # MACD
        macd = ta.trend.MACD(
            close,
            window_fast=CFG["features"]["macd_fast"],
            window_slow=CFG["features"]["macd_slow"],
            window_sign=CFG["features"]["macd_signal"]
        )
        tdf["macd"] = macd.macd()
        tdf["macd_signal"] = macd.macd_signal()
        tdf["macd_diff"] = macd.macd_diff()
        tdf["macd_bullish"] = (tdf["macd"] > tdf["macd_signal"]).astype(int)
        
        # Bollinger Bands
        bb = ta.volatility.BollingerBands(
            close,
            window=CFG["features"]["bollinger_period"],
            window_dev=CFG["features"]["bollinger_std"]
        )
        tdf["bb_upper"] = bb.bollinger_hband()
        tdf["bb_lower"] = bb.bollinger_lband()
        tdf["bb_mid"] = bb.bollinger_mavg()
        tdf["bb_width"] = (tdf["bb_upper"] - tdf["bb_lower"]) / tdf["bb_mid"]
        tdf["bb_position"] = (close - tdf["bb_lower"]) / (tdf["bb_upper"] - tdf["bb_lower"] + 1e-9)
        
        # Stochastic Oscillator
        stoch = ta.momentum.StochasticOscillator(tdf["high"], tdf["low"], close, window=14)
        tdf["stoch_k"] = stoch.stoch()
        tdf["stoch_d"] = stoch.stoch_signal()
        
        results.append(tdf)
    
    return pd.concat(results, ignore_index=True)


def compute_market_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cross-sectional features: how a stock behaves relative to market.
    This is key for equity ML — stocks don't move in isolation.
    """
    # Get S&P 500 data as market benchmark
    benchmark = CFG["universe"]["benchmark"]
    
    if benchmark in df["ticker"].unique():
        spy_ret = df[df["ticker"] == benchmark][["date", "return_5d"]].rename(
            columns={"return_5d": "market_return_5d"}
        )
        df = df.merge(spy_ret, on="date", how="left")
        
        # Beta proxy (rolling correlation with market)
        # Simplified: relative return vs market
        df["excess_return_5d"] = df["return_5d"] - df["market_return_5d"]
    else:
        df["market_return_5d"] = np.nan
        df["excess_return_5d"] = np.nan
    
    # Day of week / month effects (calendar features)
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter
    
    return df


def compute_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create prediction target.
    Target: 5-day forward return direction (binary classification).
    
    Note: This uses FUTURE data — must only be computed at training time,
    never at inference time. The serving pipeline uses features only.
    """
    horizon = CFG["features"]["target_horizon"]
    
    df["future_return"] = df.groupby("ticker")["close"].transform(
        lambda x: x.shift(-horizon).div(x) - 1
    )
    
    # Binary target: 1 = price goes up, 0 = price goes down
    df["target"] = (df["future_return"] > 0).astype(int)
    
    # Also keep regression target for optional regression models
    df["target_return"] = df["future_return"]
    
    return df


# ════════════════════════════════════════════════════════════════
#  FEATURE STORE — save/load feature sets
# ════════════════════════════════════════════════════════════════

# Canonical list of features the model uses (feature contract)
FEATURE_COLUMNS = [
    # Returns
    "return_5d", "return_10d", "return_20d", "return_50d",
    "return_5d_log", "return_10d_log", "gap",
    # MAs
    "price_to_sma_5", "price_to_sma_10", "price_to_sma_20", "price_to_sma_50",
    "price_to_ema_5", "price_to_ema_10", "price_to_ema_20", "price_to_ema_50",
    "sma_5_20_cross", "sma_10_50_cross",
    # Volatility
    "volatility_10d", "volatility_20d", "hl_range", "hl_range_5d_avg", "atr_14",
    # Volume
    "volume_ratio", "volume_change", "obv_ratio",
    # Technical
    "rsi_14", "rsi_overbought", "rsi_oversold",
    "macd", "macd_signal", "macd_diff", "macd_bullish",
    "bb_width", "bb_position",
    "stoch_k", "stoch_d",
    # Market
    "excess_return_5d", "market_return_5d",
    # Calendar
    "day_of_week", "month", "quarter",
]

META_COLUMNS = ["date", "ticker", "close", "target", "target_return"]


def build_feature_set(df: pd.DataFrame) -> pd.DataFrame:
    """Run all feature engineering steps and return clean feature dataframe."""
    logger.info("Building features...")
    
    df = df.copy()
    df = df.sort_values(["ticker", "date"]).reset_index(drop=True)
    
    # Run all feature groups
    logger.info("  → Return features")
    df = compute_return_features(df)
    logger.info("  → Moving averages")
    df = compute_moving_averages(df)
    logger.info("  → Volatility")
    df = compute_volatility_features(df)
    logger.info("  → Volume")
    df = compute_volume_features(df)
    logger.info("  → Technical indicators")
    df = compute_technical_indicators(df)
    logger.info("  → Market features")
    df = compute_market_features(df)
    logger.info("  → Target variable")
    df = compute_target(df)
    
    # Select only needed columns
    all_cols = META_COLUMNS + FEATURE_COLUMNS
    available = [c for c in all_cols if c in df.columns]
    df = df[available]
    
    # Drop rows with NaN features (warmup period for rolling windows)
    n_before = len(df)
    df = df.dropna(subset=[c for c in FEATURE_COLUMNS if c in df.columns])
    logger.info(f"  Dropped {n_before - len(df)} warmup rows, {len(df):,} rows remain")
    
    # Filter out benchmark ticker (not a tradeable stock)
    df = df[df["ticker"] != CFG["universe"]["benchmark"]]
    
    logger.info(f"✅ Feature set: {len(df):,} rows × {len(available)} columns")
    return df


def save_features(df: pd.DataFrame, version: str = "latest") -> Path:
    """Save feature set to feature store."""
    path = FEATURE_DIR / f"features_{version}.parquet"
    table = pa.Table.from_pandas(df)
    pq.write_table(table, path, compression="snappy")
    logger.info(f"Feature store saved → {path}")
    return path


def load_features(version: str = "latest") -> pd.DataFrame:
    """Load features from feature store."""
    path = FEATURE_DIR / f"features_{version}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Feature store not found: {path}")
    df = pd.read_parquet(path)
    logger.info(f"Loaded feature store: {len(df):,} rows")
    return df


def get_latest_features_for_serving(ticker: str = None) -> pd.DataFrame:
    """
    Get most recent feature row(s) for serving.
    This is what the prediction API calls — features without target.
    """
    df = load_features()
    latest_date = df["date"].max()
    latest = df[df["date"] == latest_date]
    
    if ticker:
        latest = latest[latest["ticker"] == ticker]
    
    return latest[FEATURE_COLUMNS]


# ════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════

def run_feature_engineering():
    from src.ingestion.ingest import load_all_raw
    
    logger.info("=" * 60)
    logger.info("STARTING FEATURE ENGINEERING PIPELINE")
    logger.info("=" * 60)
    
    raw_df = load_all_raw()
    feature_df = build_feature_set(raw_df)
    save_features(feature_df, version="latest")
    
    # Print feature summary stats
    logger.info("\nFeature Statistics (sample):")
    sample_cols = ["return_5d", "rsi_14", "volatility_20d", "bb_position"]
    available_sample = [c for c in sample_cols if c in feature_df.columns]
    if available_sample:
        logger.info(f"\n{feature_df[available_sample].describe().round(4)}")
    
    return feature_df


if __name__ == "__main__":
    run_feature_engineering()
