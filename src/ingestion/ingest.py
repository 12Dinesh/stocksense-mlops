"""
Data Ingestion Module
=====================
Downloads OHLCV data from Yahoo Finance, validates it with Great Expectations,
and stores in Parquet format (columnar, fast, industry-standard).

Industry pattern: This mirrors what ETL pipelines at hedge funds / fintech
companies do — pull from market data vendors (Bloomberg, Refinitiv) and land
in a data lake (S3/GCS as Parquet). We use yfinance + local Parquet here.
"""

import os
import sys
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import yfinance as yf
import pyarrow as pa
import pyarrow.parquet as pq
import duckdb
from loguru import logger
from tenacity import retry, stop_after_attempt, wait_exponential
import yaml

# ─── Setup ──────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

RAW_DIR = ROOT / CFG["paths"]["raw_data"]
RAW_DIR.mkdir(parents=True, exist_ok=True)

logger.remove()
logger.add(ROOT / "logs/ingestion.log", rotation="1 day", retention="30 days", level="INFO")
logger.add(sys.stdout, colorize=True, level="INFO")


# ════════════════════════════════════════════════════════════════
#  DOWNLOADER
# ════════════════════════════════════════════════════════════════

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=2, max=10))
def download_ticker(ticker: str, start: str, end: str) -> pd.DataFrame:
    """Download OHLCV + adjusted data for a single ticker."""
    logger.info(f"Downloading {ticker} from {start} to {end}")
    df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
    
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    
    # Flatten MultiIndex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0].lower() for col in df.columns]
    else:
        df.columns = [c.lower() for c in df.columns]
    
    df.index.name = "date"
    df = df.reset_index()
    df["ticker"] = ticker
    df["date"] = pd.to_datetime(df["date"])
    
    logger.info(f"  ✓ {ticker}: {len(df)} rows")
    return df


def download_all(lookback_days: int = None) -> pd.DataFrame:
    """Download all tickers in the universe."""
    lookback = lookback_days or CFG["universe"]["lookback_days"]
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=lookback)).strftime("%Y-%m-%d")
    
    tickers = CFG["universe"]["tickers"] + [CFG["universe"]["benchmark"]]
    frames = []
    
    for ticker in tickers:
        try:
            df = download_ticker(ticker, start, end)
            frames.append(df)
        except Exception as e:
            logger.error(f"Failed {ticker}: {e}")
    
    combined = pd.concat(frames, ignore_index=True)
    logger.info(f"Total rows downloaded: {len(combined):,}")
    return combined


# ════════════════════════════════════════════════════════════════
#  DATA QUALITY — Great Expectations style checks
# ════════════════════════════════════════════════════════════════

class DataQualityChecker:
    """
    Lightweight data contract validation.
    Industry tools: Great Expectations, Soda, dbt tests.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = []
        self.passed = 0
        self.failed = 0
    
    def expect(self, name: str, condition: bool, severity: str = "error"):
        status = "✓ PASS" if condition else "✗ FAIL"
        self.results.append({"check": name, "status": status, "severity": severity})
        if condition:
            self.passed += 1
        else:
            self.failed += 1
            if severity == "error":
                logger.error(f"Data quality FAIL: {name}")
            else:
                logger.warning(f"Data quality WARN: {name}")
    
    def run_all(self) -> bool:
        df = self.df
        
        # Schema checks
        self.expect("required_columns_present",
                    all(c in df.columns for c in ["date", "open", "high", "low", "close", "volume", "ticker"]))
        
        # Completeness
        self.expect("no_null_close_prices", df["close"].isna().sum() == 0)
        self.expect("no_null_dates", df["date"].isna().sum() == 0)
        
        # Business rules
        self.expect("high_gte_low", (df["high"] >= df["low"]).all())
        self.expect("close_within_high_low", 
                    ((df["close"] <= df["high"]) & (df["close"] >= df["low"])).all())
        self.expect("positive_volume", (df["volume"] >= 0).all())
        self.expect("positive_prices", (df[["open","high","low","close"]] > 0).all().all())
        
        # Freshness check
        latest = df["date"].max()
        days_old = (datetime.today() - pd.Timestamp(latest)).days
        self.expect("data_freshness_within_5_days", days_old <= 5, severity="warning")
        
        # Coverage
        expected_tickers = set(CFG["universe"]["tickers"])
        actual_tickers = set(df["ticker"].unique())
        self.expect("all_tickers_present", expected_tickers.issubset(actual_tickers))
        
        # Print summary
        logger.info(f"\n{'─'*50}")
        logger.info(f"DATA QUALITY REPORT: {self.passed} passed, {self.failed} failed")
        for r in self.results:
            logger.info(f"  {r['status']} {r['check']}")
        logger.info(f"{'─'*50}")
        
        return self.failed == 0


# ════════════════════════════════════════════════════════════════
#  STORAGE — Parquet + DuckDB
# ════════════════════════════════════════════════════════════════

def save_raw(df: pd.DataFrame) -> Path:
    """
    Save raw data as partitioned Parquet.
    Industry pattern: Hive-partitioned parquet is the standard
    data lake format (used in Spark, Delta Lake, AWS Athena).
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RAW_DIR / f"ohlcv_{timestamp}.parquet"
    
    table = pa.Table.from_pandas(df)
    pq.write_table(table, out_path, compression="snappy")
    
    logger.info(f"Saved raw data → {out_path} ({out_path.stat().st_size / 1024:.1f} KB)")
    return out_path


def load_latest_raw() -> pd.DataFrame:
    """
    Load most recent raw data file using DuckDB for fast querying.
    DuckDB is the in-process analytical DB used by data engineers
    as a local alternative to BigQuery/Snowflake.
    """
    parquet_files = sorted(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No raw data found. Run ingestion first.")
    
    latest = parquet_files[-1]
    con = duckdb.connect()
    df = con.execute(f"SELECT * FROM read_parquet('{latest}')").df()
    logger.info(f"Loaded {len(df):,} rows from {latest.name}")
    return df


def load_all_raw() -> pd.DataFrame:
    """Load and deduplicate all raw Parquet files (data lake merge pattern)."""
    parquet_files = list(RAW_DIR.glob("*.parquet"))
    if not parquet_files:
        raise FileNotFoundError("No raw data found.")
    
    con = duckdb.connect()
    files_str = str([str(f) for f in parquet_files])
    
    # DuckDB can read multiple parquets and deduplicate
    df = con.execute(f"""
        SELECT * FROM (
            SELECT *, ROW_NUMBER() OVER (
                PARTITION BY ticker, date 
                ORDER BY date DESC
            ) as rn
            FROM read_parquet({files_str})
        ) WHERE rn = 1
        ORDER BY ticker, date
    """).df()
    
    logger.info(f"Loaded {len(df):,} unique rows from {len(parquet_files)} files")
    return df


# ════════════════════════════════════════════════════════════════
#  ENTRYPOINT
# ════════════════════════════════════════════════════════════════

def run_ingestion():
    logger.info("=" * 60)
    logger.info("STARTING DATA INGESTION PIPELINE")
    logger.info("=" * 60)
    
    # 1. Download
    df = download_all()
    
    # 2. Quality checks
    checker = DataQualityChecker(df)
    quality_ok = checker.run_all()
    
    if not quality_ok:
        logger.warning("Data quality issues found — proceeding with caution")
    
    # 3. Save
    out_path = save_raw(df)
    
    logger.info("✅ Ingestion complete")
    return out_path


if __name__ == "__main__":
    run_ingestion()
