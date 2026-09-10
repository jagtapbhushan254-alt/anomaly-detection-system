"""
Streamlit Live Dashboard — Real-Time Anomaly Detection
=======================================================
Author: Bhushan Jagtap
"""

from dataclasses import asdict
from producer.stream_producer import (
    generate_normal_transaction,
    generate_fraud_transaction,
)
import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time
import random
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")
API_URL = "https://anomaly-detection-system-h4jj.onrender.com"

st.markdown(
    """
<style>
    .block-container {
        padding: 1.1rem 1.7rem 2rem;
        max-width: 1600px;
    }
    .hero {
        position: relative;
        overflow: hidden;
        padding: 1.45rem 1.6rem;
        border-radius: 18px;
        background: linear-gradient(135deg, #101827 0%, #18263d 55%, #101827 100%);
        border: 1px solid #293a55;
        box-shadow: 0 12px 35px rgba(0,0,0,.18);
        margin-bottom: 1.15rem;
    }
    .hero:after {
        content: "";
        position: absolute;
        width: 220px;
        height: 220px;
        right: -80px;
        top: -120px;
        border-radius: 50%;
        background: rgba(59,130,246,.12);
        filter: blur(4px);
    }
    .hero-title { font-size: 2rem; font-weight: 800; margin: 0; letter-spacing: -.03em; }
    .hero-subtitle { color:#9fb0c7; margin-top:.35rem; font-size:.9rem; }
    .live-pill {
        display:inline-block; margin-top:.75rem; padding:.25rem .6rem;
        border-radius:999px; background:#102a24; color:#4ade80;
        border:1px solid #1d5c45; font-size:.72rem; font-weight:700;
    }
    .section-title { font-size:1.08rem; font-weight:750; margin:.25rem 0 .7rem; }
    .mini-card {
        background:#101722; border:1px solid #263449; border-radius:14px;
        padding:.8rem 1rem; min-height:78px;
    }
    .mini-label { color:#8494aa; font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; }
    .mini-value { font-size:1.35rem; font-weight:750; margin-top:.2rem; }
    .mini-sub { color:#718198; font-size:.7rem; margin-top:.15rem; }
    .alert-card {
        background:#101722; border:1px solid #29384d; border-left:4px solid #ef4444;
        border-radius:11px; padding:.7rem .8rem; margin:.42rem 0;
    }
    .risk-high { color:#f87171; font-weight:750; }
    .risk-medium { color:#fbbf24; font-weight:750; }
    .risk-low { color:#4ade80; font-weight:750; }
    .muted { color:#8494aa; font-size:.74rem; }
    .status-online {
        background:#0e2a20; border:1px solid #1c5b43; color:#4ade80;
        border-radius:10px; padding:.65rem .75rem; font-weight:700;
    }
    .status-offline {
        background:#321b22; border:1px solid #6d3342; color:#fb7185;
        border-radius:10px; padding:.65rem .75rem; font-weight:700;
    }
    .side-brand { font-size:1.18rem; font-weight:800; }
    .side-caption { color:#8796aa; font-size:.76rem; margin-bottom:1rem; }
    .table-head {
        color:#8291a6; font-size:.7rem; text-transform:uppercase;
        letter-spacing:.06em; padding:.25rem 0;
    }
</style>
""",
    unsafe_allow_html=True,
)

for key, val in [
    ("transactions", []),
    ("anomalies", []),
    ("total", 0),
    ("running", False),
]:
    if key not in st.session_state:
        st.session_state[key] = val


def call_api(txn_dict):
    fields = [
        "amount",
        "merchant_category",
        "hour_of_day",
        "day_of_week",
        "transaction_count_1h",
        "avg_amount_7d",
        "distance_from_home_km",
    ]
    payload = {k: txn_dict[k] for k in fields}
    try:
        r = requests.post(f"{API_URL}/predict", json=payload, timeout=2)
        return r.json() if r.status_code == 200 else None
    except BaseException:
        return None


# Sidebar
with st.sidebar:
    st.markdown(
        '<div class="side-brand">⚡ Sentinel Monitor</div>', unsafe_allow_html=True
    )
    st.markdown(
        '<div class="side-caption">Financial risk operations console</div>',
        unsafe_allow_html=True,
    )
    st.markdown("### 🎛️ Simulation Controls")
    fraud_rate = st.slider("Fraud Rate (%)", 1, 30, 5) / 100
    refresh_rate = st.slider("Refresh Speed (s)", 1, 5, 2)
    max_history = st.slider("History Window", 20, 200, 50)
    st.markdown("---")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("▶️ Start", use_container_width=True):
            st.session_state.running = True
    with c2:
        if st.button("⏹️ Stop", use_container_width=True):
            st.session_state.running = False
    if st.button("🔄 Reset", use_container_width=True):
        st.session_state.transactions = []
        st.session_state.anomalies = []
        st.session_state.total = 0
    st.markdown("---")
    st.markdown("### 🧠 Detection Engine")
    try:
        h = requests.get(f"{API_URL}/health", timeout=1).json()
        st.markdown(
            '<div class="status-online">● API ONLINE</div>',
            unsafe_allow_html=True,
        )
        st.caption(
            f"Isolation Forest {'✓' if h.get('isolation_forest') else '✗'} • "
            f"Autoencoder {'✓' if h.get('autoencoder') else '✗'}"
        )
    except BaseException:
        st.markdown(
            '<div class="status-offline">● API OFFLINE</div>',
            unsafe_allow_html=True,
        )
        st.caption("Start FastAPI on port 8000 to enable detection.")
    st.markdown("---")
    st.caption("Monitoring mode • Real-time API inference • Dual-model ensemble")

# Header
st.markdown(
    """
<div class="hero">
    <div class="hero-title">🛡️ Sentinel Risk Intelligence</div>
    <div class="hero-subtitle">
        Real-time transaction surveillance • Dual-model anomaly detection •
        Automated risk classification
    </div>
    <div class="live-pill">● LIVE RISK MONITORING</div>
</div>
""",
    unsafe_allow_html=True,
)

# KPIs
total = st.session_state.total
n_an = len(st.session_state.anomalies)
rate = (n_an / total * 100) if total > 0 else 0
high = sum(1 for a in st.session_state.anomalies if a.get("risk_level") == "HIGH")
medium = sum(1 for a in st.session_state.anomalies if a.get("risk_level") == "MEDIUM")
avg_conf = (
    sum(float(a.get("ensemble_confidence", 0)) for a in st.session_state.transactions)
    / len(st.session_state.transactions)
    if st.session_state.transactions
    else 0
)

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Transactions screened</div><div class="mini-value">{
            total:,    }</div><div class="mini-sub">Live transaction volume</div></div>',
        unsafe_allow_html=True,
    )
with k2:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Anomalies detected</div><div class="mini-value">{
            n_an:,    }</div><div class="mini-sub">Potentially suspicious</div></div>',
        unsafe_allow_html=True,
    )
with k3:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Anomaly rate</div><div class="mini-value">{
            rate:.1f}%</div><div class="mini-sub">Share of screened activity</div></div>',
        unsafe_allow_html=True,
    )
with k4:
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">High-risk alerts</div><div class="mini-value">{high}</div><div class="mini-sub">{medium} medium-risk alerts</div></div>',
        unsafe_allow_html=True,
    )
with k5:
    system = "RUNNING" if st.session_state.running else "PAUSED"
    st.markdown(
        f'<div class="mini-card"><div class="mini-label">Detection engine</div>'
        f'<div class="mini-value">{system}</div>'
        f'<div class="mini-sub">Avg confidence {avg_conf * 100:.1f}%</div></div>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

# Charts
ch, al = st.columns([1.75, 1])
with ch:
    st.markdown(
        '<div class="section-title">📈 Live Transaction Stream</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.transactions:
        df = pd.DataFrame(st.session_state.transactions[-max_history:])
        fig = go.Figure()
        for flag, color, sym, name in [
            (False, "#00cc44", "circle", "Normal"),
            (True, "#ff4444", "x", "Anomaly"),
        ]:
            sub = df[df["is_anomaly"] == flag]
            fig.add_trace(
                go.Scatter(
                    x=sub.index,
                    y=sub["amount"],
                    mode="markers",
                    name=name,
                    marker=dict(
                        color=color,
                        size=9 if flag else 6,
                        symbol=sym,
                        line=dict(width=2, color=color) if flag else dict(width=0),
                    ),
                )
            )
        fig.update_layout(
            height=300,
            paper_bgcolor="#0e1117",
            plot_bgcolor="#0e1117",
            font_color="white",
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis=dict(showgrid=False),
            yaxis=dict(title="Amount ($)", gridcolor="#333"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Press ▶️ Start to begin monitoring.")

with al:
    st.markdown(
        '<div class="section-title">🚨 Live Security Alerts</div>',
        unsafe_allow_html=True,
    )
    if st.session_state.anomalies:
        for a in reversed(st.session_state.anomalies[-8:]):
            rc = f"risk-{a.get('risk_level', 'LOW').lower()}"
            st.markdown(
                f"""<div class="alert-card">
                🚨 <b>${a.get('amount', 0):,.2f}</b>
                <span class="{rc}"> {a.get('risk_level', '?')}</span><br>
                <span class="muted">
                    IF Score: {a.get('anomaly_score', 0):.3f}
                    &nbsp; • &nbsp;
                    AE Error: {a.get('reconstruction_error', 0):.4f}
                    &nbsp; • &nbsp;
                    Confidence: {a.get('ensemble_confidence', 0) * 100:.1f}%
                </span>
            </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No anomalies yet")

# Recent transaction activity
if st.session_state.transactions:
    st.markdown(
        '<div class="section-title">🧾 Recent Transaction Activity</div>',
        unsafe_allow_html=True,
    )
    recent = pd.DataFrame(st.session_state.transactions[-8:][::-1]).copy()
    recent["Status"] = recent["is_anomaly"].map({True: "🚨 ALERT", False: "✓ NORMAL"})
    recent["Amount"] = recent["amount"].map(lambda x: f"${x:,.2f}")
    recent["Confidence"] = recent["ensemble_confidence"].map(
        lambda x: f"{x * 100:.1f}%"
    )
    recent["Risk"] = recent["risk_level"].map(lambda x: str(x))
    recent = recent[["Status", "Amount", "merchant_category", "Risk", "Confidence"]]
    recent.columns = ["Status", "Amount", "Merchant", "Risk", "Confidence"]
    st.dataframe(recent, use_container_width=True, hide_index=True)

# Distribution charts
if len(st.session_state.transactions) > 10:
    st.markdown(
        '<div class="section-title">📊 Model Signal Analysis</div>',
        unsafe_allow_html=True,
    )
    df_all = pd.DataFrame(st.session_state.transactions)
    d1, d2 = st.columns(2)
    with d1:
        fig2 = px.histogram(
            df_all,
            x="anomaly_score",
            color="is_anomaly",
            nbins=30,
            title="Isolation Forest Scores",
            template="plotly_dark",
            color_discrete_map={False: "#00cc44", True: "#ff4444"},
        )
        fig2.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig2, use_container_width=True)
    with d2:
        fig3 = px.histogram(
            df_all,
            x="reconstruction_error",
            color="is_anomaly",
            nbins=30,
            title="Autoencoder Reconstruction Error",
            template="plotly_dark",
            color_discrete_map={False: "#00cc44", True: "#ff4444"},
        )
        fig3.update_layout(height=250, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig3, use_container_width=True)

    st.markdown(
        '<div class="section-title">🎯 Ensemble Confidence</div>',
        unsafe_allow_html=True,
    )
    fig4 = px.histogram(
        df_all,
        x="ensemble_confidence",
        nbins=20,
        title="Combined Model Confidence",
        template="plotly_dark",
    )
    fig4.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=30, b=0),
        xaxis_title="Confidence",
        yaxis_title="Transactions",
    )
    st.plotly_chart(fig4, use_container_width=True)

# Simulation loop
if st.session_state.running:
    is_fraud = random.random() < fraud_rate
    txn = generate_fraud_transaction() if is_fraud else generate_normal_transaction()
    result = call_api(asdict(txn))
    if result:
        row = {
            "amount": txn.amount,
            "merchant_category": txn.merchant_category,
            "hour_of_day": txn.hour_of_day,
            "is_anomaly": result["is_anomaly"],
            "anomaly_score": result["anomaly_score"],
            "reconstruction_error": result["reconstruction_error"],
            "risk_level": result["risk_level"],
            "ensemble_confidence": result["ensemble_confidence"],
        }
        st.session_state.transactions.append(row)
        st.session_state.total += 1
        if result["is_anomaly"]:
            st.session_state.anomalies.append({**row, **result})
        if len(st.session_state.transactions) > max_history * 2:
            st.session_state.transactions = st.session_state.transactions[-max_history:]
    time.sleep(refresh_rate)
    st.rerun()
