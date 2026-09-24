# 🔍 Real-Time Financial Anomaly Detection System

<p align="center">

**An End-to-End Machine Learning & MLOps System for Detecting Suspicious Financial Transactions**

</p>

<p align="center">

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11-blue?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-REST%20API-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?logo=docker&logoColor=white)](https://www.docker.com/)
[![GitHub Actions](https://img.shields.io/badge/CI-GitHub%20Actions-2088FF?logo=github-actions&logoColor=white)](https://github.com/features/actions)

</p>

<p align="center">

[![Tests](https://img.shields.io/badge/Tests-14%20Passing-success?logo=pytest&logoColor=white)](https://pytest.org/)
[![Black](https://img.shields.io/badge/Code%20Style-Black-black?logo=python&logoColor=white)](https://black.readthedocs.io/)
[![Flake8](https://img.shields.io/badge/Linting-Flake8-blueviolet)](https://flake8.pycqa.org/)
[![License](https://img.shields.io/badge/Status-Active%20Development-orange)]()

</p>

---

## 🎯 **Overview**

Financial transaction systems generate enormous volumes of data, making it difficult to identify **unusual, suspicious, or potentially fraudulent transactions** using manual analysis alone.

This project implements an **end-to-end financial anomaly detection system** that combines **classical machine learning, deep learning, real-time inference, API development, interactive monitoring, containerization, automated testing, and CI/CD**.

The system uses an ensemble of:

- 🌲 **Isolation Forest** for unsupervised tree-based anomaly detection
- 🧠 **PyTorch Autoencoder** for reconstruction-error-based detection
- ⚡ **FastAPI** for real-time REST API inference
- 📊 **Streamlit** for interactive transaction monitoring
- 🐳 **Docker** for containerized deployment
- 🧪 **Pytest** for automated testing
- 🔄 **GitHub Actions** for Continuous Integration

> 💡 **Core Idea:** Combine multiple anomaly detection signals to provide a more comprehensive assessment of potentially suspicious financial transactions.

---

# 🚀 **Key Features**

## 🤖 **Machine Learning**

- 🌲 **Isolation Forest** for unsupervised anomaly detection
- 🧠 **PyTorch Autoencoder** for reconstruction-based anomaly detection
- 🔗 **Ensemble detection** combining signals from both models
- 📈 **Anomaly score generation**
- 🚨 **Risk-level classification**
- ⚙️ Configurable model parameters
- 💾 Persistent trained model artifacts

---

## ⚡ **Real-Time Inference**

- 🚀 **FastAPI REST API**
- 💳 Transaction-level prediction
- ⚡ Low-overhead model inference
- 📦 Structured API responses
- ❤️ API health/status endpoints
- 📚 Interactive **Swagger/OpenAPI documentation**

---

## 📊 **Interactive Monitoring Dashboard**

The Streamlit dashboard provides an interface for monitoring transaction activity and model predictions.

### Dashboard capabilities include:

- 📈 Transaction screening statistics
- 🚨 Anomaly detection status
- 🛡️ Risk-level information
- 📊 Anomaly scores
- 💳 Transaction details
- 🧠 Model inference information
- 📡 Transaction/stream visualization

---

## 🧪 **Testing & Code Quality**

The project incorporates software engineering practices including:

- ✅ **Pytest** automated testing
- 🐍 Python **3.10 & 3.11** CI validation
- 🖤 **Black** code formatting
- 🔎 **Flake8** static analysis
- 🔄 **GitHub Actions** Continuous Integration

---

# 🏗️ **System Architecture**

```text
                    💳 Financial Transaction
                              │
                              ▼
                ┌─────────────────────────────┐
                │  ⚙️ Feature Preparation &  │
                │      Preprocessing          │
                └──────────────┬──────────────┘
                               │
                 ┌─────────────┴─────────────┐
                 │                           │
                 ▼                           ▼
        ┌──────────────────┐       ┌──────────────────┐
        │ 🌲 Isolation     │       │ 🧠 PyTorch       │
        │    Forest        │       │    Autoencoder   │
        │                  │       │                  │
        │ Tree-Based       │       │ Reconstruction   │
        │ Detection        │       │ Error Detection │
        └────────┬─────────┘       └─────────┬────────┘
                 │                           │
                 └─────────────┬─────────────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │ 🔗 Ensemble Decision        │
                │                             │
                │ 📈 Anomaly Score            │
                │ 🚨 Risk Level               │
                │ 🎯 Final Prediction         │
                └──────────────┬──────────────┘
                               │
                    ┌──────────┴──────────┐
                    │                     │
                    ▼                     ▼
           ┌────────────────┐    ┌──────────────────┐
           │ ⚡ FastAPI     │    │ 📊 Streamlit     │
           │ REST Inference │    │ Monitoring UI    │
           └────────────────┘    └──────────────────┘
```

---

# 🧠 **Machine Learning Approach**

## 1️⃣ **Isolation Forest**

**Isolation Forest** is used as a tree-based unsupervised anomaly detection model.

The model identifies observations that are easier to isolate from the rest of the transaction population.

### ⚙️ Configurable Parameters

- 🎛️ Contamination rate
- 🌲 Number of estimators
- 💾 Model persistence
- 💳 Transaction-level prediction

---

## 2️⃣ **PyTorch Autoencoder**

The Autoencoder is a neural-network-based model that learns to reconstruct transaction features.

Transactions that produce a **higher reconstruction error** can be treated as potential anomalies.

```text
💳 Input Transaction
        │
        ▼
   🧠 Encoder
        │
        ▼
   🔹 Latent Space
        │
        ▼
   🧠 Decoder
        │
        ▼
🔄 Reconstructed Transaction
        │
        ▼
📉 Reconstruction Error
        │
        ▼
🚨 Anomaly Decision
```

### ⚙️ Configurable Training Parameters

- 🔢 Number of epochs
- 📦 Batch size
- 📈 Learning rate
- 🎯 Anomaly threshold percentile

---

## 3️⃣ **Ensemble Detection**

Instead of relying on a single algorithm, the system combines signals generated by:

**🌲 Isolation Forest + 🧠 Autoencoder**

The resulting inference pipeline provides information such as:

- 🎯 Anomaly status
- 📈 Anomaly score
- 🚨 Risk level
- 📉 Reconstruction error
- 💳 Transaction information

This approach allows the system to combine **tree-based isolation** with **neural reconstruction-based detection**.

---

# 🛠️ **Technology Stack**

| 🧩 Category | 🔧 Technologies |
|---|---|
| 💻 **Language** | Python |
| 🤖 **Machine Learning** | Scikit-learn, PyTorch |
| 📊 **Data Processing** | Pandas, NumPy |
| ⚡ **API** | FastAPI, Uvicorn |
| 📈 **Dashboard** | Streamlit |
| 📊 **Visualization** | Plotly |
| 🧪 **Testing** | Pytest |
| 🖤 **Code Formatting** | Black |
| 🔎 **Linting** | Flake8 |
| 🐳 **Containerization** | Docker, Docker Compose |
| 🔄 **CI/CD** | GitHub Actions |
| 📦 **Version Control** | Git, GitHub |

---

# 🗂️ **Project Structure**

```text
🔍 anomaly-detection-system/
│
├── ⚙️ .github/
│   └── 🔄 workflows/
│       └── ci.yml
│
├── 📦 data/
│   └── 🤖 models/
│
├── 🚀 src/
│   │
│   ├── ⚡ api/
│   │   └── main.py
│   │
│   ├── 📊 dashboard/
│   │   └── app.py
│   │
│   ├── 🧠 models/
│   │   ├── autoencoder.py
│   │   ├── isolation_forest.py
│   │   └── train.py
│   │
│   └── 📡 producer/
│       └── stream_producer.py
│
├── 🧪 tests/
│   └── test_api.py
│
├── 🐳 Dockerfile
├── 🐳 docker-compose.yml
├── 📋 requirements.txt
├── 🚫 .gitignore
└── 📖 README.md
```

### 🔎 **Component Overview**

| 📁 Component | 🎯 Purpose |
|---|---|
| ⚡ **`api/`** | REST API and model inference |
| 📊 **`dashboard/`** | Interactive Streamlit monitoring interface |
| 🧠 **`models/`** | Machine learning models and training pipeline |
| 📡 **`producer/`** | Transaction data generation/streaming |
| 🧪 **`tests/`** | Automated application tests |
| 📦 **`data/models/`** | Trained model artifacts |
| ⚙️ **`.github/workflows/`** | Automated CI pipeline |
| 🐳 **Docker** | Containerized deployment |
| 📋 **`requirements.txt`** | Project dependencies |

---

# 🚀 **Getting Started**

## 📋 **Prerequisites**

Make sure you have:

- 🐍 **Python 3.10 or 3.11**
- 📦 **pip**
- 🔧 **Git**
- 🐳 **Docker** *(optional)*

---

## 1️⃣ **Clone the Repository**

```bash
git clone https://github.com/jagtapbhushan254-alt/anomaly-detection-system.git

cd anomaly-detection-system
```

---

## 2️⃣ **Create a Virtual Environment**

### 🪟 Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### 🐧 macOS / Linux

```bash
python3 -m venv .venv

source .venv/bin/activate
```

---

## 3️⃣ **Install Dependencies**

```bash
pip install -r requirements.txt
```

---

# 🧠 **Train the Models**

The training pipeline generates transaction data and trains both anomaly detection models.

Run:

```bash
python src/models/train.py
```

### 🔄 Training Pipeline

```text
📊 Generate Dataset
        ↓
⚙️ Feature Preparation
        ↓
🌲 Train Isolation Forest
        ↓
🧠 Train Autoencoder
        ↓
💾 Save Model Artifacts
        ↓
🎯 Run Sample Predictions
```

Trained model artifacts are stored under:

```text
data/models/
```

---

# ⚡ **Run the FastAPI Backend**

Start the API server:

```bash
uvicorn src.api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

### 📚 Interactive API Documentation

Open:

```text
http://127.0.0.1:8000/docs
```

The Swagger interface allows you to interactively test the available prediction endpoints.

---

# 📊 **Run the Streamlit Dashboard**

Start the dashboard:

```bash
streamlit run src/dashboard/app.py
```

The dashboard provides an interactive interface for viewing transaction screening and anomaly detection results.

---

# 🐳 **Docker Deployment**

The project supports containerized execution using Docker.

## 🔨 Build the Docker Image

```bash
docker build -t anomaly-detection-system .
```

## ▶️ Run the Container

```bash
docker run -p 8000:8000 anomaly-detection-system
```

## 🧩 Docker Compose

Alternatively:

```bash
docker compose up --build
```

---

# 🧪 **Testing**

The project includes an automated **Pytest** suite.

Run:

```bash
pytest
```

### ✅ Current Test Status

```text
14 tests passed
```

The CI pipeline additionally validates the project using:

```text
🐍 Python 3.10
🐍 Python 3.11
```

---

# 🧹 **Code Quality**

The project uses **Black** for formatting and **Flake8** for static analysis.

## 🖤 Format Code

```bash
black src/ tests/
```

## 🔎 Check Formatting

```bash
black --check src/ tests/
```

## 🧹 Run Flake8

```bash
flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503,E203
```

---

# 🔄 **Continuous Integration**

Every push and pull request is automatically validated using **GitHub Actions**.

### 🔁 CI Pipeline

```text
📤 Push / Pull Request
          │
          ▼
   ⚙️ GitHub Actions
          │
     ┌────┴────┐
     │         │
     ▼         ▼
 🧪 Tests    🧹 Lint
     │         │
 ┌───┴───┐   ┌─┴──────┐
 ▼       ▼   ▼        ▼
Py3.10  Py3.11 Black  Flake8
```

### ✅ CI Checks

- 🐍 Python **3.10** tests
- 🐍 Python **3.11** tests
- 🖤 **Black** formatting
- 🔎 **Flake8** linting
- 🔄 Automated GitHub Actions workflow

👉 **[View GitHub Actions](https://github.com/jagtapbhushan254-alt/anomaly-detection-system/actions)**

---

# 📡 **API Workflow**

A typical transaction moves through the system as follows:

```text
💳 Transaction
      │
      ▼
⚡ FastAPI Endpoint
      │
      ▼
⚙️ Feature Preparation
      │
      ├───────────────┐
      ▼               ▼
🌲 Isolation      🧠 Autoencoder
   Forest
      │               │
      └───────┬───────┘
              ▼
       🔗 Ensemble Logic
              │
              ▼
      🚨 Risk / Anomaly
           Result
              │
              ▼
       📦 API Response
```

---

# 📈 **Example Prediction Output**

A prediction response contains transaction information together with the resulting anomaly assessment.

### 💡 Conceptual Response

```json
{
  "transaction_id": "example-id",
  "is_anomaly": true,
  "risk_level": "HIGH",
  "anomaly_score": 0.XX
}
```

> ℹ️ The exact response schema is defined by the FastAPI application.

---

# 🛡️ **Engineering Practices**

This project follows several software engineering and MLOps principles:

- 🧩 **Modular source-code organization**
- 🧠 **Separate model implementations**
- 🔄 **Reusable training pipeline**
- ⚡ **REST-based model inference**
- 🐳 **Containerization**
- 🧪 **Automated testing**
- 🔄 **Continuous Integration**
- 🖤 **Code formatting**
- 🔎 **Static code analysis**
- 📦 **Reproducible dependency installation**
- 💾 **Model artifact persistence**

The objective is not only to train an ML model, but to demonstrate how a machine learning solution can be developed as an **end-to-end software system**.

---

# 🗺️ **Development Roadmap**

## ✅ **Completed**

- [x] 📊 Transaction data generation
- [x] ⚙️ Feature preprocessing
- [x] 🌲 Isolation Forest implementation
- [x] 🧠 PyTorch Autoencoder implementation
- [x] 🔄 Model training pipeline
- [x] 💾 Model artifact persistence
- [x] 🔗 Ensemble anomaly detection
- [x] ⚡ FastAPI inference service
- [x] 📊 Streamlit dashboard
- [x] 🧪 Automated tests
- [x] 🐳 Docker support
- [x] 🔄 GitHub Actions CI
- [x] 🖤 Black formatting checks
- [x] 🔎 Flake8 linting
- [x] 🐍 Python 3.10 CI testing
- [x] 🐍 Python 3.11 CI testing

## 🚧 **Future Improvements**

- [ ] 🌍 Expand real-world financial transaction datasets
- [ ] 🤖 Add additional anomaly detection algorithms
- [ ] 🎯 Improve model calibration and threshold selection
- [ ] 📊 Add experiment tracking
- [ ] 📡 Add model monitoring and drift detection
- [ ] 🧪 Expand integration testing
- [ ] 🔐 Add authentication and API security
- [ ] ☁️ Improve production deployment architecture

---

# 🎓 **Project Purpose**

This project was developed as a practical **Machine Learning Engineering and MLOps project** demonstrating the integration of:

```text
🤖 Machine Learning
        +
💻 Software Engineering
        +
⚡ APIs
        +
🐳 Deployment
        +
🧪 Testing
        +
🔄 MLOps
```

Rather than focusing exclusively on model development, the project demonstrates the supporting engineering infrastructure required to move an anomaly detection model toward an **end-to-end deployable application**.

---

# 📚 **Research Background**

The project is also informed by experience working on **secure systems, machine learning, and applied research**.

### 📄 Publications

**🔐 MegaShare: A Secure Offline File-Sharing Framework**  
IEEE ICCCNT 2025

**🎙️ Voice Based Biometric Authentication and AI Assistant**  
ICASET 2026

The financial anomaly detection system is an **independent engineering project** focused on practical machine learning, deployment, testing, and MLOps.

---

# 👨‍💻 **Author**

## **Bhushan Prabhakar Jagtap**

🎓 **B.E. Computer Engineering**  
🏫 Pillai HOC College of Engineering & Technology  
📍 Mumbai, India

📧 **Email:** `jagtapbhushan254@gmail.com`

🔗 **GitHub:**  
[github.com/jagtapbhushan254-alt](https://github.com/jagtapbhushan254-alt)

---

# ⭐ **Project Status**

🚀 **Active Development**

The core **anomaly detection pipeline, model training, API, dashboard, testing infrastructure, containerization, and CI pipeline** are implemented.

Future development is focused on:

> 📊 **Better datasets** → 🎯 **Improved model calibration** → 📡 **Model monitoring** → ☁️ **Production deployment**

---

<p align="center">

### 🔍 **Built with Machine Learning • Engineering • APIs • MLOps**

⭐ **If you find this project interesting, consider giving the repository a star!**

</p>