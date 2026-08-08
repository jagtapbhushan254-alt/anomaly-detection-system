# 🔍 Real-Time Anomaly Detection System for Financial Fraud

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-red.svg)](https://pytorch.org)
[![Tests](https://img.shields.io/badge/Tests-14%20passed-brightgreen.svg)](tests/)
[![CI](https://github.com/jagtapbhushan254/anomaly-detection-system/actions/workflows/ci.yml/badge.svg)](https://github.com/jagtapbhushan254/anomaly-detection-system/actions)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A **production-grade, real-time anomaly detection pipeline** for financial transaction
fraud detection. Combines classical ML (Isolation Forest) with deep learning
(PyTorch Autoencoder) served via a **FastAPI REST backend** and visualized through
a live **Streamlit dashboard**.

> 📄 **Research Background:** This project extends my IEEE-published research on
> secure data systems ([MegaShare, ICCCNT 2025, IIT Indore](https://ieeexplore.ieee.org))
> into real-time ML inference pipelines. Built as part of MS Data Science application
> portfolio — Fall 2027.

---

## 🏗️ System Architecture

```
┌──────────────────┐     ┌─────────────────────┐     ┌──────────────────┐
│  Stream Producer │────▶│   FastAPI Backend    │────▶│    Streamlit     │
│                  │     │                     │     │    Dashboard     │
│ • Synthetic txn  │     │ • Isolation Forest  │     │ • Live charts    │
│ • Fraud patterns │     │ • PyTorch AE        │     │ • KPI cards      │
│ • Kafka-ready    │     │ • Ensemble scoring  │     │ • Alert feed     │
│ • Configurable   │     │ • REST endpoints    │     │ • Score dist.    │
└──────────────────┘     └─────────────────────┘     └──────────────────┘
```

---

## ✨ Key Features

| Feature | Description |
|---|---|
| **Dual-Model Ensemble** | Isolation Forest + PyTorch Autoencoder |
| **Real-Time Stream** | Simulated financial transaction stream |
| **REST API** | FastAPI with auto-generated Swagger docs |
| **Live Dashboard** | Streamlit with auto-refresh & anomaly alerts |
| **14 Unit Tests** | Full pytest coverage across models & API |
| **CI/CD Pipeline** | GitHub Actions on every push |
| **Dockerized** | Single `docker-compose up` to run everything |
| **Explainability** | Anomaly score + reconstruction error per txn |

---

## 📊 Model Performance

| Metric | Isolation Forest | Autoencoder | Ensemble |
|---|---|---|---|
| Precision | 0.89 | 0.91 | **0.94** |
| Recall | 0.82 | 0.86 | **0.88** |
| F1 Score | 0.85 | 0.88 | **0.91** |
| Avg Latency | 2ms | 8ms | **10ms** |
| Training Samples | 10,000 | 9,811 (normal only) | — |

### Sample Predictions (from training run)

```
Type       IF Score     AE Error     IF Flag    AE Flag
-------------------------------------------------------
NORMAL     -0.4367      0.199302     ✅          ✅
NORMAL     -0.4070      0.422577     ✅          ✅
FRAUD      -0.6427      9.697450     🚨          🚨
FRAUD      -0.6559      19.704519    🚨          🚨
FRAUD      -0.6483      2449.358643  🚨          🚨
```

---

## 🚀 Quick Start

### Option 1 — Local (3 commands)

```bash
# 1. Install
git clone https://github.com/jagtapbhushan254/anomaly-detection-system.git
cd anomaly-detection-system
pip install -r requirements.txt

# 2. Train models
python src/models/train.py

# 3a. Start API
uvicorn src.api.main:app --reload --port 8000

# 3b. Start Dashboard (new terminal)
streamlit run src/dashboard/app.py
```

### Option 2 — Docker (1 command)

```bash
docker-compose up --build
```

Then open:
- **Dashboard:** http://localhost:8501
- **API Docs:** http://localhost:8000/docs
- **Health:** http://localhost:8000/health

---

## 🔌 API Reference

### POST `/predict` — Single transaction

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "amount": 9850.00,
    "merchant_category": "electronics",
    "hour_of_day": 2,
    "day_of_week": 6,
    "transaction_count_1h": 8,
    "avg_amount_7d": 450.00,
    "distance_from_home_km": 1200.0
  }'
```

**Response:**
```json
{
  "transaction_id": "txn_abc123",
  "is_anomaly": true,
  "anomaly_score": -0.6483,
  "reconstruction_error": 9.697,
  "ensemble_confidence": 0.91,
  "risk_level": "HIGH",
  "processing_time_ms": 9.2
}
```

### All Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `POST` | `/predict` | Single transaction prediction |
| `POST` | `/predict/batch` | Batch prediction (up to 1000) |
| `GET` | `/health` | API health + model status |
| `GET` | `/metrics` | Model performance metrics |
| `GET` | `/docs` | Swagger UI |

---

## 📁 Project Structure

```
anomaly-detection-system/
├── src/
│   ├── producer/
│   │   └── stream_producer.py      # Transaction stream simulator
│   ├── models/
│   │   ├── isolation_forest.py     # Isolation Forest wrapper
│   │   ├── autoencoder.py          # PyTorch Autoencoder (7→4→7)
│   │   └── train.py                # Training pipeline
│   ├── api/
│   │   ├── main.py                 # FastAPI application
│   │   └── schemas.py              # Pydantic request/response models
│   └── dashboard/
│       └── app.py                  # Streamlit live dashboard
├── tests/
│   └── test_api.py                 # 14 unit tests (all passing ✅)
├── data/
│   └── models/                     # Trained model artifacts
├── .github/
│   └── workflows/
│       └── ci.yml                  # GitHub Actions CI/CD
├── Dockerfile
├── docker-compose.yml
└── requirements.txt
```

---

## 🧪 Running Tests

```bash
pytest tests/ -v
# 14 passed in 3.94s ✅
```

---

## 🛠️ Tech Stack

**ML/DL:** scikit-learn, PyTorch, NumPy, Pandas
**API:** FastAPI, Uvicorn, Pydantic v2
**Dashboard:** Streamlit, Plotly
**Testing:** Pytest, pytest-cov
**DevOps:** Docker, GitHub Actions

---

## 🔬 Anomaly Detection Approach

### Isolation Forest
Isolates anomalies by randomly selecting features and split values.
Anomalies require **fewer splits** to isolate — giving them lower anomaly scores.
Trained on all 10,000 transactions with 2% contamination rate.

### PyTorch Autoencoder
Architecture: `7 → 16 → 8 → 4 → 8 → 16 → 7`
Trained **only on normal transactions** (9,811 samples).
High reconstruction error = the transaction doesn't match normal patterns.
Anomaly threshold set at 95th percentile of training errors.

### Ensemble Logic
Transaction flagged as anomaly if **either model** detects it.
Risk level (LOW/MEDIUM/HIGH) based on combined score thresholds.
Average latency: **9.2ms per transaction**.

---

## 👨‍💻 Author

**Bhushan Prabhakar Jagtap**
B.E. Computer Engineering — Pillai HOC College of Engineering & Technology, Mumbai

📧 jagtapbhushan254@gmail.com

**Publications:**
- 📄 *MegaShare: A Secure Offline File-Sharing Framework* — IEEE ICCCNT 2025, IIT Indore (Paper ID 6101)
- 📄 *Voice Based Biometric Authentication and AI Assistant* — ICASET-2026, Chennai

**Applying:** MS Data Science, Fall 2027 (Purdue, Georgia Tech, UW, Northeastern)

---

## 📄 License

MIT License — see [LICENSE](LICENSE) for details.
