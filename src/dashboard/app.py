"""
Streamlit Live Dashboard — Real-Time Anomaly Detection
=======================================================
Author: Bhushan Jagtap
"""

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import time, random, sys, os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from producer.stream_producer import (
    generate_normal_transaction,
    generate_fraud_transaction,
)
from dataclasses import asdict

st.set_page_config(page_title="Anomaly Detection", page_icon="🔍", layout="wide")
API_URL = "http://localhost:8000"

st.markdown(
    """
<style>
.anomaly-alert{background:#2d0000;border-radius:8px;padding:.75rem 1rem;
border-left:4px solid #ff4444;color:#ff8888;margin:.3rem 0;font-size:.85rem}
.risk-high{color:#ff4444;font-weight:bold}
.risk-medium{color:#ffaa00;font-weight:bold}
.risk-low{color:#00cc44;font-weight:bold}
</style>""",
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
    except:
        return None


# Sidebar
with st.sidebar:
    st.markdown("## ⚙️ Controls")
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
    try:
        h = requests.get(f"{API_URL}/health", timeout=1).json()
        st.success("✅ API Online")
        st.caption(
            f"IF: {'✅' if h['isolation_forest'] else '❌'}  AE: {'✅' if h['autoencoder'] else '❌'}"
        )
    except:
        st.error("❌ API Offline")
    st.markdown("---")
    st.markdown("**Author:** Bhushan Prabhakar Jagtap")
    st.markdown("**IEEE ICCCNT 2025** | **ICASET 2026**")

# Header
st.markdown("# 🔍 Real-Time Anomaly Detection System")
st.markdown("Financial Fraud Detection — Isolation Forest + PyTorch Autoencoder")
st.markdown("---")

# KPIs
k1, k2, k3, k4, k5 = st.columns(5)
total = st.session_state.total
n_an = len(st.session_state.anomalies)
rate = (n_an / total * 100) if total > 0 else 0
high = sum(1 for a in st.session_state.anomalies if a.get("risk_level") == "HIGH")
k1.metric("Transactions", f"{total:,}")
k2.metric(
    "Anomalies",
    f"{n_an:,}",
    delta=f"+{min(n_an,1)}" if n_an else None,
    delta_color="inverse",
)
k3.metric("Anomaly Rate", f"{rate:.1f}%")
k4.metric("High Risk", f"{high}", delta_color="inverse")
k5.metric("Status", "🟢 Running" if st.session_state.running else "🔴 Stopped")
st.markdown("---")

# Charts
ch, al = st.columns([2, 1])
with ch:
    st.markdown("### 📈 Transaction Stream")
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
                        size=8 if flag else 6,
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
    st.markdown("### 🚨 Live Alerts")
    if st.session_state.anomalies:
        for a in reversed(st.session_state.anomalies[-8:]):
            rc = f"risk-{a.get('risk_level','LOW').lower()}"
            st.markdown(
                f"""<div class="anomaly-alert">
                🚨 <b>${a.get('amount',0):,.0f}</b> —
                <span class="{rc}">{a.get('risk_level','?')}</span><br>
                Score: {a.get('anomaly_score',0):.3f} | AE: {a.get('reconstruction_error',0):.4f}
            </div>""",
                unsafe_allow_html=True,
            )
    else:
        st.caption("No anomalies yet")

# Distribution charts
if len(st.session_state.transactions) > 10:
    st.markdown("### 📊 Score Distributions")
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
        }
        st.session_state.transactions.append(row)
        st.session_state.total += 1
        if result["is_anomaly"]:
            st.session_state.anomalies.append({**row, **result})
        if len(st.session_state.transactions) > max_history * 2:
            st.session_state.transactions = st.session_state.transactions[-max_history:]
    time.sleep(refresh_rate)
    st.rerun()
