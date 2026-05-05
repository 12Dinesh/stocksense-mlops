import streamlit as st
import pandas as pd
import numpy as np
import pickle
import warnings
import sys
from pathlib import Path
from datetime import datetime, timedelta
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="StockSense MLOps",
    page_icon="📈",
    layout="wide"
)

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

# ── Load config ──────────────────────────────────────────────
import yaml
with open(ROOT / "config.yaml", encoding="utf-8") as f:
    CFG = yaml.safe_load(f)

TICKERS = CFG["universe"]["tickers"]
BENCHMARK = CFG["universe"]["benchmark"]
MODEL_PATH = ROOT / "models/registry/champion_model.pkl"

# ── Header ────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#1B3A5C;'>📈 StockSense MLOps Dashboard</h1>
<p style='color:#666;'>Live stock direction predictions — powered by LightGBM</p>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Download fresh data ───────────────────────────────────────
@st.cache_data(ttl=3600)
def download_data():
    import yfinance as yf
    end = datetime.today().strftime("%Y-%m-%d")
    start = (datetime.today() - timedelta(days=365)).strftime("%Y-%m-%d")

    frames = []
    for ticker in TICKERS + [BENCHMARK]:
        try:
            df = yf.download(
                ticker, start=start, end=end,
                auto_adjust=True, progress=False
            )
            if df.empty:
                continue
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = [col[0].lower() for col in df.columns]
            else:
                df.columns = [c.lower() for c in df.columns]
            df.index.name = "date"
            df = df.reset_index()
            df["ticker"] = ticker
            df["date"] = pd.to_datetime(df["date"])
            frames.append(df)
        except Exception as e:
            st.warning(f"Could not download {ticker}: {e}")
            continue

    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)

# ── Build features ────────────────────────────────────────────
def build_features(df):
    import ta

    df = df.copy().sort_values(["ticker","date"]).reset_index(drop=True)

    # Returns
    for w in [5, 10, 20, 50]:
        df[f"return_{w}d"] = df.groupby("ticker")["close"].pct_change(w)
        df[f"return_{w}d_log"] = np.log1p(df[f"return_{w}d"].fillna(0))

    # Gap
    df["gap"] = df.groupby("ticker").apply(
        lambda g: (g["open"] - g["close"].shift(1)) / (g["close"].shift(1) + 1e-9)
    ).reset_index(level=0, drop=True)

    # Moving averages
    for w in [5, 10, 20, 50]:
        df[f"sma_{w}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.rolling(w, min_periods=1).mean()
        )
        df[f"ema_{w}"] = df.groupby("ticker")["close"].transform(
            lambda x: x.ewm(span=w, adjust=False).mean()
        )
        df[f"price_to_sma_{w}"] = df["close"] / (df[f"sma_{w}"] + 1e-9) - 1
        df[f"price_to_ema_{w}"] = df["close"] / (df[f"ema_{w}"] + 1e-9) - 1

    df["sma_5_20_cross"] = np.sign(df["sma_5"] - df["sma_20"])
    df["sma_10_50_cross"] = np.sign(df["sma_10"] - df["sma_50"])

    # Volatility
    for w in [10, 20]:
        df[f"volatility_{w}d"] = df.groupby("ticker")["close"].transform(
            lambda x: x.pct_change().rolling(w, min_periods=1).std() * np.sqrt(252)
        )
    df["hl_range"] = (df["high"] - df["low"]) / (df["close"] + 1e-9)
    df["hl_range_5d_avg"] = df.groupby("ticker")["hl_range"].transform(
        lambda x: x.rolling(5, min_periods=1).mean()
    )
    df["true_range"] = df.groupby("ticker").apply(
        lambda g: pd.concat([
            g["high"] - g["low"],
            (g["high"] - g["close"].shift()).abs(),
            (g["low"] - g["close"].shift()).abs()
        ], axis=1).max(axis=1)
    ).reset_index(level=0, drop=True)
    df["atr_14"] = df.groupby("ticker")["true_range"].transform(
        lambda x: x.ewm(span=14, adjust=False).mean()
    )

    # Volume
    df["volume_sma_20"] = df.groupby("ticker")["volume"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df["volume_ratio"] = df["volume"] / (df["volume_sma_20"] + 1e-9)
    df["volume_change"] = df.groupby("ticker")["volume"].pct_change()

    def compute_obv(group):
        direction = np.sign(group["close"].diff())
        return (direction * group["volume"]).cumsum()

    df["obv"] = df.groupby("ticker").apply(compute_obv).reset_index(level=0, drop=True)
    df["obv_sma_20"] = df.groupby("ticker")["obv"].transform(
        lambda x: x.rolling(20, min_periods=1).mean()
    )
    df["obv_ratio"] = df["obv"] / (df["obv_sma_20"].abs() + 1)

    # Technical indicators
    results = []
    for ticker in df["ticker"].unique():
        tdf = df[df["ticker"] == ticker].copy().sort_values("date")
        close = tdf["close"]

        tdf["rsi_14"] = ta.momentum.RSIIndicator(close, window=14).rsi()
        tdf["rsi_overbought"] = (tdf["rsi_14"] > 70).astype(int)
        tdf["rsi_oversold"] = (tdf["rsi_14"] < 30).astype(int)

        macd = ta.trend.MACD(close, window_fast=12, window_slow=26, window_sign=9)
        tdf["macd"] = macd.macd()
        tdf["macd_signal"] = macd.macd_signal()
        tdf["macd_diff"] = macd.macd_diff()
        tdf["macd_bullish"] = (tdf["macd"] > tdf["macd_signal"]).astype(int)

        bb = ta.volatility.BollingerBands(close, window=20, window_dev=2)
        tdf["bb_upper"] = bb.bollinger_hband()
        tdf["bb_lower"] = bb.bollinger_lband()
        tdf["bb_mid"] = bb.bollinger_mavg()
        tdf["bb_width"] = (tdf["bb_upper"] - tdf["bb_lower"]) / (tdf["bb_mid"] + 1e-9)
        tdf["bb_position"] = (close - tdf["bb_lower"]) / (tdf["bb_upper"] - tdf["bb_lower"] + 1e-9)

        stoch = ta.momentum.StochasticOscillator(tdf["high"], tdf["low"], close, window=14)
        tdf["stoch_k"] = stoch.stoch()
        tdf["stoch_d"] = stoch.stoch_signal()

        results.append(tdf)

    df = pd.concat(results, ignore_index=True)

    # Market features
    if BENCHMARK in df["ticker"].unique():
        spy = df[df["ticker"] == BENCHMARK][["date","return_5d"]].rename(
            columns={"return_5d": "market_return_5d"}
        )
        df = df.merge(spy, on="date", how="left")
        df["excess_return_5d"] = df["return_5d"] - df["market_return_5d"]
    else:
        df["market_return_5d"] = 0
        df["excess_return_5d"] = 0

    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["month"] = pd.to_datetime(df["date"]).dt.month
    df["quarter"] = pd.to_datetime(df["date"]).dt.quarter

    return df

# ── Load model ────────────────────────────────────────────────
@st.cache_resource
def load_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None

# ── Feature columns ───────────────────────────────────────────
FEATURE_COLUMNS = [
    "return_5d","return_10d","return_20d","return_50d",
    "return_5d_log","return_10d_log","gap",
    "price_to_sma_5","price_to_sma_10","price_to_sma_20","price_to_sma_50",
    "price_to_ema_5","price_to_ema_10","price_to_ema_20","price_to_ema_50",
    "sma_5_20_cross","sma_10_50_cross",
    "volatility_10d","volatility_20d","hl_range","hl_range_5d_avg","atr_14",
    "volume_ratio","volume_change","obv_ratio",
    "rsi_14","rsi_overbought","rsi_oversold",
    "macd","macd_signal","macd_diff","macd_bullish",
    "bb_width","bb_position","stoch_k","stoch_d",
    "excess_return_5d","market_return_5d",
    "day_of_week","month","quarter",
]

# ── Main app ──────────────────────────────────────────────────
model = load_model()

if model is None:
    st.error("Model file not found. Please ensure models/registry/champion_model.pkl is committed to GitHub.")
    st.stop()

with st.spinner("Downloading live stock data from Yahoo Finance..."):
    raw = download_data()

if raw is None:
    st.error("Could not download stock data. Check internet connection.")
    st.stop()

with st.spinner("Computing 40+ features..."):
    featured = build_features(raw)

# Get latest row per ticker
latest = featured[featured["ticker"] != BENCHMARK].groupby("ticker").last().reset_index()
feature_cols = [c for c in FEATURE_COLUMNS if c in latest.columns]
X = latest[feature_cols].fillna(0)

# Predict
probs = model.predict_proba(X)[:, 1]
preds = (probs > 0.5).astype(int)

# ── Predictions table ─────────────────────────────────────────
st.subheader("🎯 Live Predictions — 5-Day Direction")

cols = st.columns(4)
for i, (_, row) in enumerate(latest.iterrows()):
    prob = float(probs[i])
    direction = "📈 UP" if preds[i] == 1 else "📉 DOWN"
    color = "green" if preds[i] == 1 else "red"
    with cols[i % 4]:
        st.metric(
            label=row["ticker"],
            value=direction,
            delta=f"Confidence: {prob:.1%}"
        )

st.markdown("---")

# ── Summary table ─────────────────────────────────────────────
st.subheader("📊 Prediction Summary")
summary = pd.DataFrame({
    "Ticker": latest["ticker"].values,
    "Last Close": latest["close"].round(2).values,
    "Prediction": ["UP ✅" if p == 1 else "DOWN ❌" for p in preds],
    "Probability": [f"{p:.1%}" for p in probs],
    "RSI": latest["rsi_14"].round(1).values if "rsi_14" in latest.columns else ["N/A"]*len(latest),
    "MACD Signal": ["Bullish 📈" if m == 1 else "Bearish 📉"
                    for m in (latest["macd_bullish"].values if "macd_bullish" in latest.columns
                    else [0]*len(latest))]
})
st.dataframe(summary, use_container_width=True)

st.markdown("---")

# ── Price chart ───────────────────────────────────────────────
st.subheader("📉 Price History")
import plotly.graph_objects as go

selected = st.selectbox("Select stock", TICKERS)
stock_df = featured[featured["ticker"] == selected].sort_values("date").tail(60)

fig = go.Figure()
fig.add_trace(go.Scatter(
    x=stock_df["date"], y=stock_df["close"],
    name="Close Price", line=dict(color="#1B3A5C", width=2)
))
if "bb_upper" in stock_df.columns:
    fig.add_trace(go.Scatter(
        x=stock_df["date"], y=stock_df["bb_upper"],
        name="BB Upper", line=dict(color="rgba(200,100,100,0.5)", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=stock_df["date"], y=stock_df["bb_lower"],
        name="BB Lower", line=dict(color="rgba(100,200,100,0.5)", dash="dash"),
        fill="tonexty", fillcolor="rgba(150,150,200,0.1)"
    ))
fig.update_layout(
    title=f"{selected} — Last 60 Days with Bollinger Bands",
    xaxis_title="Date", yaxis_title="Price (USD)",
    height=400, template="plotly_white"
)
st.plotly_chart(fig, use_container_width=True)

# ── RSI chart ─────────────────────────────────────────────────
if "rsi_14" in stock_df.columns:
    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=stock_df["date"], y=stock_df["rsi_14"],
        name="RSI 14", line=dict(color="#2E6DA4", width=2)
    ))
    fig2.add_hline(y=70, line_dash="dash", line_color="red", annotation_text="Overbought")
    fig2.add_hline(y=30, line_dash="dash", line_color="green", annotation_text="Oversold")
    fig2.update_layout(
        title=f"{selected} — RSI",
        height=250, template="plotly_white"
    )
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("---")
st.caption(f"Data refreshes every hour. Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M UTC')}")
st.caption("Built with LightGBM + Streamlit | StockSense MLOps")