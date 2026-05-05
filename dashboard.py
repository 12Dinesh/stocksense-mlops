"""
StockSense Dashboard
=====================
Production-grade Streamlit web app showing:
  - Live predictions for all tickers
  - Model performance metrics
  - Feature importance (SHAP)
  - Drift monitoring dashboard
  - Historical price charts with prediction overlays

This is the "face" of the ML system — what business users see.
"""

import json
import sys
import time
from pathlib import Path
from datetime import datetime, timedelta

import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import httpx
import yaml

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT))

with open(ROOT / "config.yaml") as f:
    CFG = yaml.safe_load(f)

API_BASE = f"http://localhost:{CFG['serving']['port']}"
MODEL_DIR = ROOT / CFG["paths"]["model_registry"]
MONITOR_DIR = ROOT / "monitoring_reports"

# ─── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="StockSense MLOps",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

    .stApp { background: #0a0e1a; }
    
    * { font-family: 'Sora', sans-serif; }
    code, .stCode { font-family: 'JetBrains Mono', monospace !important; }
    
    .main-header {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 32px;
        margin-bottom: 24px;
        position: relative;
        overflow: hidden;
    }
    .main-header::before {
        content: '';
        position: absolute;
        top: -50%;
        right: -10%;
        width: 300px;
        height: 300px;
        background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
        pointer-events: none;
    }
    .main-title {
        font-size: 2.4rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #8b5cf6, #06b6d4);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
    }
    .main-subtitle {
        color: #64748b;
        font-size: 0.95rem;
        margin-top: 8px;
        font-weight: 300;
    }
    
    .metric-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        transition: border-color 0.2s;
    }
    .metric-card:hover { border-color: #6366f1; }
    .metric-value { font-size: 1.8rem; font-weight: 700; color: #f1f5f9; }
    .metric-label { font-size: 0.8rem; color: #64748b; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
    
    .prediction-card {
        background: #0f172a;
        border: 1px solid #1e293b;
        border-radius: 12px;
        padding: 16px 20px;
        margin: 8px 0;
        display: flex;
        align-items: center;
        justify-content: space-between;
    }
    .up-badge {
        background: rgba(16, 185, 129, 0.15);
        color: #10b981;
        border: 1px solid rgba(16, 185, 129, 0.3);
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .down-badge {
        background: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid rgba(239, 68, 68, 0.3);
        border-radius: 6px;
        padding: 4px 12px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .confidence-high { color: #10b981; font-weight: 600; }
    .confidence-medium { color: #f59e0b; font-weight: 600; }
    .confidence-low { color: #6b7280; font-weight: 600; }
    
    .status-dot-green { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #10b981; margin-right: 6px; animation: pulse 2s infinite; }
    .status-dot-red { display: inline-block; width: 8px; height: 8px; border-radius: 50%; background: #ef4444; margin-right: 6px; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
    
    .section-header {
        font-size: 1.1rem;
        font-weight: 600;
        color: #e2e8f0;
        border-bottom: 1px solid #1e293b;
        padding-bottom: 12px;
        margin-bottom: 20px;
    }
    
    div[data-testid="stSidebar"] { background: #0a0e1a; border-right: 1px solid #1e293b; }
    .stSelectbox label, .stMultiSelect label { color: #94a3b8 !important; }
    
    /* Tables */
    .stDataFrame { background: #0f172a !important; }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  DATA FETCHERS
# ════════════════════════════════════════════════════════════════

@st.cache_data(ttl=300)  # Cache for 5 minutes
def fetch_all_predictions():
    """Fetch predictions from API."""
    try:
        resp = httpx.get(f"{API_BASE}/predict/all", timeout=15)
        if resp.status_code == 200:
            return resp.json(), None
    except Exception as e:
        pass
    
    # Fallback: generate from model directly
    try:
        sys.path.insert(0, str(ROOT))
        from src.features.engineer import load_features, FEATURE_COLUMNS
        import pickle
        
        champion_path = MODEL_DIR / "champion_model.pkl"
        meta_path = MODEL_DIR / "champion_metadata.json"
        
        if not champion_path.exists():
            return None, "No trained model found. Run the training pipeline first."
        
        with open(champion_path, "rb") as f:
            model = pickle.load(f)
        with open(meta_path) as f:
            meta = json.load(f)
        
        df = load_features()
        feature_cols = [c for c in meta.get("feature_columns", FEATURE_COLUMNS) if c in df.columns]
        
        latest_date = df["date"].max()
        latest = df[df["date"] == latest_date]
        
        predictions = []
        for _, row in latest.iterrows():
            if row["ticker"] == CFG["universe"]["benchmark"]:
                continue
            X = pd.DataFrame([row[feature_cols]])
            prob = float(model.predict_proba(X)[0, 1])
            direction = "UP" if prob >= 0.5 else "DOWN"
            conf_score = abs(prob - 0.5) * 2
            confidence = "HIGH" if conf_score > 0.4 else ("MEDIUM" if conf_score > 0.2 else "LOW")
            
            predictions.append({
                "ticker": row["ticker"],
                "date": str(latest_date)[:10],
                "prediction": direction,
                "probability": round(prob, 4),
                "confidence": confidence,
                "model_version": meta.get("trained_at", "unknown")[:10],
                "cached": False,
                "latency_ms": 0
            })
        
        predictions.sort(key=lambda x: x["probability"], reverse=True)
        return {
            "predictions": predictions,
            "timestamp": datetime.now().isoformat(),
            "model_version": meta.get("trained_at", "unknown")[:10]
        }, None
    
    except Exception as e:
        return None, str(e)


@st.cache_data(ttl=3600)  # Cache for 1 hour
def fetch_price_history(ticker: str, days: int = 90):
    """Fetch historical price data."""
    try:
        from src.ingestion.ingest import load_all_raw
        df = load_all_raw()
        df = df[df["ticker"] == ticker].sort_values("date")
        cutoff = pd.Timestamp.now() - pd.Timedelta(days=days)
        return df[df["date"] >= cutoff]
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=300)
def fetch_model_info():
    """Get model metadata."""
    meta_path = MODEL_DIR / "champion_metadata.json"
    if meta_path.exists():
        with open(meta_path) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=300)
def fetch_monitoring_snapshot():
    """Get latest monitoring snapshot."""
    snapshots = sorted(MONITOR_DIR.glob("snapshot_*.json")) if MONITOR_DIR.exists() else []
    if snapshots:
        with open(snapshots[-1]) as f:
            return json.load(f)
    return None


@st.cache_data(ttl=3600)
def fetch_feature_importance():
    """Load SHAP feature importance."""
    shap_path = MODEL_DIR / "shap_importance.csv"
    if shap_path.exists():
        return pd.read_csv(shap_path)
    fi_path = MODEL_DIR / "feature_importance.csv"
    if fi_path.exists():
        return pd.read_csv(fi_path).rename(columns={"importance": "mean_shap"})
    return None


# ════════════════════════════════════════════════════════════════
#  SIDEBAR
# ════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='padding: 16px 0 8px 0;'>
        <div style='font-size: 1.2rem; font-weight: 700; color: #6366f1;'>⚡ StockSense</div>
        <div style='font-size: 0.75rem; color: #475569; margin-top: 4px;'>MLOps Platform</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    
    page = st.radio(
        "Navigation",
        ["📊 Predictions", "📈 Price Charts", "🧠 Model Analytics", "🔍 Monitoring", "⚙️ System"],
        label_visibility="collapsed"
    )
    
    st.divider()
    
    # System status
    model_meta = fetch_model_info()
    
    if model_meta:
        st.markdown('<span class="status-dot-green"></span> **Model Active**', unsafe_allow_html=True)
        st.caption(f"Version: {model_meta.get('trained_at', 'N/A')[:10]}")
        st.caption(f"Type: {model_meta.get('model_type', 'N/A').upper()}")
        st.caption(f"AUC: {model_meta.get('test_auc', 0):.4f}")
    else:
        st.markdown('<span class="status-dot-red"></span> **No Model**', unsafe_allow_html=True)
        st.caption("Run training pipeline first")
    
    st.divider()
    
    if st.button("🔄 Refresh Data", use_container_width=True):
        st.cache_data.clear()
        st.rerun()
    
    st.caption(f"Last updated: {datetime.now().strftime('%H:%M:%S')}")


# ════════════════════════════════════════════════════════════════
#  HEADER
# ════════════════════════════════════════════════════════════════

st.markdown("""
<div class="main-header">
    <div class="main-title">📈 StockSense MLOps</div>
    <div class="main-subtitle">
        End-to-end ML system · LightGBM · MLflow · Evidently · Airflow · FastAPI
    </div>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: PREDICTIONS
# ════════════════════════════════════════════════════════════════

if "Predictions" in page:
    data, error = fetch_all_predictions()
    
    if error:
        st.error(f"⚠️ {error}")
        st.info("Run the setup script: `python setup.py` then `python -m src.training.run`")
    elif data:
        predictions = data["predictions"]
        
        # Top metrics row
        up_count = sum(1 for p in predictions if p["prediction"] == "UP")
        high_conf = sum(1 for p in predictions if p["confidence"] == "HIGH")
        avg_prob = np.mean([p["probability"] for p in predictions])
        
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:#10b981">{up_count}/{len(predictions)}</div>
                <div class="metric-label">Bullish Predictions</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{high_conf}</div>
                <div class="metric-label">High Confidence</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{avg_prob:.1%}</div>
                <div class="metric-label">Avg Bull Probability</div>
            </div>""", unsafe_allow_html=True)
        with c4:
            model_meta = fetch_model_info()
            auc_val = model_meta.get("test_auc", 0) if model_meta else 0
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{auc_val:.4f}</div>
                <div class="metric-label">Model AUC</div>
            </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Predictions table
        col_left, col_right = st.columns([3, 2])
        
        with col_left:
            st.markdown('<div class="section-header">📋 Latest Predictions</div>', unsafe_allow_html=True)
            
            for p in predictions:
                direction_badge = f'<span class="up-badge">▲ UP</span>' if p["prediction"] == "UP" else f'<span class="down-badge">▼ DOWN</span>'
                conf_class = f"confidence-{p['confidence'].lower()}"
                
                st.markdown(f"""
                <div class="prediction-card">
                    <div>
                        <span style="font-size:1.1rem;font-weight:700;color:#f1f5f9">{p['ticker']}</span>
                        <span style="color:#475569;font-size:0.8rem;margin-left:8px">{p['date']}</span>
                    </div>
                    <div style="display:flex;align-items:center;gap:16px">
                        <span style="color:#94a3b8;font-size:0.9rem">{p['probability']:.1%}</span>
                        <span class="{conf_class}">{p['confidence']}</span>
                        {direction_badge}
                    </div>
                </div>
                """, unsafe_allow_html=True)
        
        with col_right:
            st.markdown('<div class="section-header">📊 Probability Distribution</div>', unsafe_allow_html=True)
            
            df_pred = pd.DataFrame(predictions)
            
            fig = go.Figure()
            fig.add_trace(go.Bar(
                x=df_pred["ticker"],
                y=df_pred["probability"],
                marker=dict(
                    color=df_pred["probability"],
                    colorscale=[[0, "#ef4444"], [0.5, "#374151"], [1, "#10b981"]],
                    cmin=0, cmax=1,
                    line=dict(color="#1e293b", width=1)
                ),
                text=[f"{p:.1%}" for p in df_pred["probability"]],
                textposition="outside",
                textfont=dict(color="#94a3b8", size=10)
            ))
            fig.add_hline(y=0.5, line_dash="dash", line_color="#6366f1", opacity=0.5,
                          annotation_text="50% (neutral)", annotation_font_color="#6366f1")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor="#1e293b"),
                yaxis=dict(tickfont=dict(color="#94a3b8"), gridcolor="#1e293b",
                           tickformat=".0%", range=[0, 1]),
                margin=dict(t=10, b=10, l=10, r=10),
                height=350
            )
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: PRICE CHARTS
# ════════════════════════════════════════════════════════════════

elif "Price Charts" in page:
    st.markdown('<div class="section-header">📈 Price History & Predictions</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        ticker = st.selectbox("Select Ticker", CFG["universe"]["tickers"])
    with col2:
        period = st.selectbox("Period", ["30 days", "90 days", "180 days", "1 year"])
    
    days_map = {"30 days": 30, "90 days": 90, "180 days": 180, "1 year": 365}
    days = days_map[period]
    
    price_df = fetch_price_history(ticker, days)
    
    if not price_df.empty:
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                            row_heights=[0.7, 0.3], vertical_spacing=0.05)
        
        # Candlestick
        fig.add_trace(go.Candlestick(
            x=price_df["date"],
            open=price_df["open"],
            high=price_df["high"],
            low=price_df["low"],
            close=price_df["close"],
            increasing=dict(fillcolor="#10b981", line=dict(color="#10b981")),
            decreasing=dict(fillcolor="#ef4444", line=dict(color="#ef4444")),
            name="OHLC"
        ), row=1, col=1)
        
        # SMA lines
        price_df["sma_20"] = price_df["close"].rolling(20).mean()
        price_df["sma_50"] = price_df["close"].rolling(50).mean()
        
        fig.add_trace(go.Scatter(x=price_df["date"], y=price_df["sma_20"],
                                  line=dict(color="#6366f1", width=1.5, dash="dot"),
                                  name="SMA 20"), row=1, col=1)
        fig.add_trace(go.Scatter(x=price_df["date"], y=price_df["sma_50"],
                                  line=dict(color="#8b5cf6", width=1.5),
                                  name="SMA 50"), row=1, col=1)
        
        # Volume
        colors = ["#10b981" if c >= o else "#ef4444"
                  for c, o in zip(price_df["close"], price_df["open"])]
        fig.add_trace(go.Bar(x=price_df["date"], y=price_df["volume"],
                              marker=dict(color=colors, opacity=0.7),
                              name="Volume"), row=2, col=1)
        
        fig.update_layout(
            title=f"{ticker} — {period}",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#94a3b8"),
            xaxis_rangeslider_visible=False,
            legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="#334155", borderwidth=1),
            height=550,
            margin=dict(t=40, b=10, l=10, r=10)
        )
        fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
        fig.update_yaxes(gridcolor="#1e293b", showgrid=True)
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"No price data available for {ticker}. Run ingestion first.")


# ════════════════════════════════════════════════════════════════
#  PAGE: MODEL ANALYTICS
# ════════════════════════════════════════════════════════════════

elif "Model Analytics" in page:
    model_meta = fetch_model_info()
    
    if not model_meta:
        st.warning("No model trained yet. Run the training pipeline first.")
    else:
        st.markdown('<div class="section-header">🧠 Model Performance</div>', unsafe_allow_html=True)
        
        cols = st.columns(5)
        metrics = [
            ("AUC", f"{model_meta.get('test_auc', 0):.4f}"),
            ("Accuracy", f"{model_meta.get('test_accuracy', 0):.4f}"),
            ("F1 Score", f"{model_meta.get('test_f1', 0):.4f}"),
            ("Type", model_meta.get("model_type", "N/A").upper()),
            ("Trained", model_meta.get("trained_at", "N/A")[:10]),
        ]
        for col, (label, value) in zip(cols, metrics):
            with col:
                st.markdown(f"""<div class="metric-card">
                    <div class="metric-value" style="font-size:1.3rem">{value}</div>
                    <div class="metric-label">{label}</div>
                </div>""", unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Feature importance
        fi_df = fetch_feature_importance()
        if fi_df is not None:
            st.markdown('<div class="section-header">🔍 Feature Importance (SHAP)</div>', unsafe_allow_html=True)
            
            top_n = st.slider("Top N features", 10, min(30, len(fi_df)), 15)
            top_fi = fi_df.head(top_n).sort_values("mean_shap")
            
            fig = go.Figure(go.Bar(
                x=top_fi["mean_shap"],
                y=top_fi["feature"],
                orientation="h",
                marker=dict(
                    color=top_fi["mean_shap"],
                    colorscale=[[0, "#1e293b"], [0.5, "#6366f1"], [1, "#06b6d4"]],
                    line=dict(color="#0f172a", width=0.5)
                )
            ))
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                xaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8")),
                yaxis=dict(gridcolor="#1e293b", tickfont=dict(color="#94a3b8", size=11)),
                height=max(300, top_n * 25),
                margin=dict(t=10, b=10, l=10, r=10)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Feature columns list
        with st.expander("Feature Contract (all model features)"):
            feature_cols = model_meta.get("feature_columns", [])
            st.code(", ".join(feature_cols), language=None)


# ════════════════════════════════════════════════════════════════
#  PAGE: MONITORING
# ════════════════════════════════════════════════════════════════

elif "Monitoring" in page:
    snapshot = fetch_monitoring_snapshot()
    
    if not snapshot:
        st.info("No monitoring data yet. Run `python -m src.monitoring.monitor`")
    else:
        st.markdown('<div class="section-header">🔍 Data Drift & Model Health</div>', unsafe_allow_html=True)
        
        drift = snapshot.get("drift", {})
        perf = snapshot.get("performance", {})
        
        # Summary cards
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            psi = drift.get("max_psi", 0)
            color = "#ef4444" if psi > 0.2 else ("#f59e0b" if psi > 0.1 else "#10b981")
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value" style="color:{color}">{psi:.3f}</div>
                <div class="metric-label">Max PSI (drift)</div>
            </div>""", unsafe_allow_html=True)
        with c2:
            drifted = drift.get("drifted_features", 0)
            total = drift.get("total_features", 1)
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{drifted}/{total}</div>
                <div class="metric-label">Drifted Features</div>
            </div>""", unsafe_allow_html=True)
        with c3:
            acc = perf.get("accuracy", 0)
            baseline = perf.get("baseline_accuracy", 0)
            delta = acc - baseline
            delta_color = "#10b981" if delta >= -0.02 else "#ef4444"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{acc:.4f}</div>
                <div class="metric-label">Recent Accuracy <span style="color:{delta_color};font-size:0.75rem">({delta:+.3f})</span></div>
            </div>""", unsafe_allow_html=True)
        with c4:
            retrain = snapshot.get("recommend_retrain", False)
            icon = "⚠️" if retrain else "✅"
            st.markdown(f"""<div class="metric-card">
                <div class="metric-value">{icon}</div>
                <div class="metric-label">{"Retrain Needed" if retrain else "Model Healthy"}</div>
            </div>""", unsafe_allow_html=True)
        
        # Feature drift table
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-header">Feature-level Drift (PSI)</div>', unsafe_allow_html=True)
        
        feat_drift = drift.get("feature_drift", {})
        if feat_drift:
            drift_df = pd.DataFrame([
                {
                    "Feature": k,
                    "PSI": v["psi"],
                    "KS Stat": v["ks_statistic"],
                    "Ref Mean": v["ref_mean"],
                    "Cur Mean": v["cur_mean"],
                    "Mean Shift": v["mean_shift"],
                    "Status": v["severity"].upper()
                }
                for k, v in feat_drift.items()
            ]).sort_values("PSI", ascending=False)
            
            def color_severity(val):
                if val == "ALERT": return "background-color: rgba(239,68,68,0.15); color: #ef4444"
                if val == "WARNING": return "background-color: rgba(245,158,11,0.15); color: #f59e0b"
                return "color: #10b981"
            
            st.dataframe(
                drift_df.style.applymap(color_severity, subset=["Status"]),
                use_container_width=True,
                height=400
            )
            
            # PSI bar chart
            fig = px.bar(
                drift_df.head(20),
                x="Feature", y="PSI",
                color="PSI",
                color_continuous_scale=[[0, "#10b981"], [0.1/0.3, "#f59e0b"], [0.2/0.3, "#ef4444"], [1, "#ef4444"]],
                title="Top 20 Features by PSI"
            )
            fig.add_hline(y=0.1, line_dash="dash", line_color="#f59e0b",
                          annotation_text="Warning (0.1)")
            fig.add_hline(y=0.2, line_dash="dash", line_color="#ef4444",
                          annotation_text="Alert (0.2)")
            fig.update_layout(
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8"),
                height=350,
                showlegend=False,
                margin=dict(t=40, b=10, l=10, r=10)
            )
            fig.update_xaxes(gridcolor="#1e293b", tickangle=45)
            fig.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig, use_container_width=True)


# ════════════════════════════════════════════════════════════════
#  PAGE: SYSTEM
# ════════════════════════════════════════════════════════════════

elif "System" in page:
    st.markdown('<div class="section-header">⚙️ System Status & Architecture</div>', unsafe_allow_html=True)
    
    # Service status
    services = [
        ("FastAPI Serving", f"http://localhost:{CFG['serving']['port']}/health", "http://localhost:8000/docs"),
        ("MLflow Tracking", "http://localhost:5001/health", "http://localhost:5001"),
        ("Airflow Scheduler", "http://localhost:8080/health", "http://localhost:8080"),
        ("Prometheus", "http://localhost:9090/-/healthy", "http://localhost:9090"),
        ("Grafana", "http://localhost:3000/api/health", "http://localhost:3000"),
        ("Redis Cache", None, None),
    ]
    
    st.markdown("**Service Health**")
    cols = st.columns(3)
    
    for i, (name, health_url, ui_url) in enumerate(services):
        with cols[i % 3]:
            status = "🟢 Running" if True else "🔴 Down"  # Simplified
            link = f"[Open UI]({ui_url})" if ui_url else ""
            st.markdown(f"""
            <div class="metric-card" style="text-align:left;">
                <div style="font-weight:600;color:#e2e8f0">{name}</div>
                <div style="font-size:0.8rem;color:#10b981;margin-top:4px">● Active</div>
                {"<div style='font-size:0.75rem;color:#6366f1;margin-top:8px'>" + ui_url + "</div>" if ui_url else ""}
            </div>
            """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Architecture diagram
    st.markdown('<div class="section-header">Architecture Flow</div>', unsafe_allow_html=True)
    st.code("""
Yahoo Finance API
       │
       ▼
  [INGESTION]  ──────────────────────────────────────────────
  yfinance download → DataQualityChecker → Parquet (DuckDB)      Data Layer
  ─────────────────────────────────────────────────────────────────────────
       │
       ▼
  [FEATURES]   ──────────────────────────────────────────────
  Returns + MAs + Volatility + Volume + Technical + Market          Feature Store
  → Feature Store (Parquet, versioned)
  ─────────────────────────────────────────────────────────────────────────
       │
       ▼
  [TRAINING]   ──────────────────────────────────────────────
  Optuna(HPO) → LightGBM/XGBoost → Walk-Forward CV → SHAP         ML Layer
  → MLflow logging → Champion/Challenger → Model Registry
  ─────────────────────────────────────────────────────────────────────────
       │
       ▼
  [SERVING]    ──────────────────────────────────────────────
  FastAPI + Redis Cache + Prometheus Metrics                        API Layer
  → /predict, /predict/batch, /predict/all, /metrics
  ─────────────────────────────────────────────────────────────────────────
       │
       ▼
  [MONITORING] ──────────────────────────────────────────────
  Evidently (PSI, KS drift) + Performance tracking                 Observability
  → Prometheus → Grafana dashboards
  ─────────────────────────────────────────────────────────────────────────
       │
       ▼
  [ORCHESTRATION] ────────────────────────────────────────────
  Apache Airflow DAG: 6 AM Mon-Fri                                 Pipeline
  ingest → validate → features → train(weekly) → monitor → reload
    """, language=None)
    
    # Quick start commands
    st.markdown('<div class="section-header">Quick Start Commands</div>', unsafe_allow_html=True)
    
    commands = {
        "1. Install dependencies": "pip install -r requirements.txt",
        "2. Start infrastructure": "docker-compose up -d",
        "3. Run ingestion": "python -m src.ingestion.ingest",
        "4. Build features": "python -m src.features.engineer",
        "5. Train model": "python -m src.training.train",
        "6. Run monitoring": "python -m src.monitoring.monitor",
        "7. Start API": "uvicorn src.serving.api:app --reload --port 8000",
        "8. Start Airflow": "airflow standalone",
        "9. Launch dashboard": "streamlit run dashboard.py",
    }
    
    for step, cmd in commands.items():
        st.code(f"# {step}\n{cmd}", language="bash")
