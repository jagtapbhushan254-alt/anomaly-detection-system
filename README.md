🔍 Real-Time Financial Anomaly Detection System

An end-to-end machine learning system for detecting anomalous financial transactions using an ensemble of Isolation Forest and PyTorch Autoencoder models, exposed through a FastAPI inference service and monitored through an interactive Streamlit dashboard.

📌 Overview

Financial transaction systems can generate large volumes of data, making it difficult to manually identify unusual or potentially fraudulent activity.

This project implements a complete anomaly detection pipeline that combines:

Synthetic financial transaction generation
Feature engineering and preprocessing
Isolation Forest anomaly detection
PyTorch Autoencoder-based anomaly detection
Ensemble risk assessment
REST API inference using FastAPI
Interactive monitoring using Streamlit
Docker-based deployment
Automated testing with Pytest
Code quality checks with Black and Flake8
Continuous Integration using GitHub Actions

The project is designed with a focus on machine learning engineering, API development, deployment, testing, and MLOps practices.

🎯 Key Features
🤖 Machine Learning
Isolation Forest for unsupervised anomaly detection
PyTorch Autoencoder for reconstruction-error-based anomaly detection
Ensemble-based anomaly assessment
Transaction risk classification
Anomaly score generation
Feature preprocessing for model inference
⚡ Real-Time Inference
FastAPI REST API
Transaction-level prediction
Low-overhead model inference
Structured prediction responses
API health/status endpoints
📊 Monitoring Dashboard

The Streamlit dashboard provides an interactive interface for monitoring transaction activity and anomaly predictions.

Features include:

Transaction screening statistics
Anomaly detection status
Risk-level information
Anomaly scores
Transaction details
Model inference information
Real-time/streaming transaction visualization
🧪 Testing & Code Quality

The repository includes automated:

Pytest test suite
Python 3.10 testing
Python 3.11 testing
Black formatting validation
Flake8 linting
GitHub Actions CI

The current test suite contains 14 passing tests.

🏗️ System Architecture
                    ┌─────────────────────────┐
                    │   Financial Transaction │
                    │          Input          │
                    └────────────┬────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │ Feature Preparation &   │
                    │    Preprocessing        │
                    └────────────┬────────────┘
                                 │
                  ┌──────────────┴──────────────┐
                  │                             │
                  ▼                             ▼
       ┌─────────────────────┐       ┌─────────────────────┐
       │   Isolation Forest  │       │  PyTorch Autoencoder│
       │                     │       │                     │
       │  Tree-based        │       │ Reconstruction      │
       │  anomaly detection │       │ error detection     │
       └──────────┬──────────┘       └──────────┬──────────┘
                  │                             │
                  └──────────────┬──────────────┘
                                 │
                                 ▼
                    ┌─────────────────────────┐
                    │    Ensemble Decision    │
                    │                         │
                    │ Anomaly Score / Risk    │
                    │ Level / Prediction      │
                    └────────────┬────────────┘
                                 │
                    ┌────────────┴────────────┐
                    │                         │
                    ▼                         ▼
          ┌──────────────────┐      ┌──────────────────┐
          │   FastAPI API    │      │    Streamlit     │
          │                  │      │    Dashboard     │
          │ REST Inference   │      │ Monitoring/UI    │
          └──────────────────┘      └──────────────────┘
🧠 Machine Learning Approach
1. Isolation Forest

Isolation Forest is used as a tree-based unsupervised anomaly detection model.

The model identifies observations that are easier to isolate from the rest of the transaction population.

The implementation supports configurable parameters such as:

Contamination rate
Number of estimators
Model persistence
Transaction-level prediction
2. PyTorch Autoencoder

A neural-network-based Autoencoder learns to reconstruct transaction features.

The reconstruction error is used as an anomaly signal:

Input Transaction
       │
       ▼
   Encoder
       │
       ▼
 Latent Space
       │
       ▼
   Decoder
       │
       ▼
Reconstructed Transaction
       │
       ▼
Reconstruction Error
       │
       ▼
 Anomaly Decision

The implementation uses PyTorch and supports configurable training parameters including:

Number of epochs
Batch size
Learning rate
Anomaly threshold percentile
3. Ensemble Detection

The system combines signals from both models to produce a more comprehensive anomaly assessment.

The inference pipeline produces information such as:

Anomaly status
Anomaly score
Risk level
Reconstruction error
Transaction information

This allows the system to use both tree-based anomaly detection and neural reconstruction-based detection.

⚙️ Technology Stack
Category	Technologies
Language	Python
Machine Learning	Scikit-learn, PyTorch
Data Processing	Pandas, NumPy
API	FastAPI, Uvicorn
Dashboard	Streamlit
Visualization	Plotly
Testing	Pytest
Code Quality	Black, Flake8
Containerization	Docker
CI/CD	GitHub Actions
Version Control	Git, GitHub
📁 Project Structure
anomaly-detection-system/
│
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── data/
│   └── models/
│
├── src/
│   ├── api/
│   │   └── main.py
│   │
│   ├── dashboard/
│   │   └── app.py
│   │
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── isolation_forest.py
│   │   └── train.py
│   │
│   └── producer/
│       └── stream_producer.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── .gitignore
└── README.md
🚀 Getting Started
Prerequisites

Make sure you have:

Python 3.10 or 3.11
pip
Git

Clone the repository:

git clone https://github.com/jagtapbhushan254-alt/anomaly-detection-system.git
cd anomaly-detection-system
🔧 Create Virtual Environment
Windows
python -m venv .venv
.venv\Scripts\activate
macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
📦 Install Dependencies
pip install -r requirements.txt
🧠 Train the Models

The training pipeline generates transaction data and trains both anomaly detection models.

Run:

python src/models/train.py

The pipeline performs:

Generate Dataset
       ↓
Feature Preparation
       ↓
Train Isolation Forest
       ↓
Train Autoencoder
       ↓
Save Model Artifacts
       ↓
Run Sample Predictions

Model artifacts are stored under:

data/models/
⚡ Run the FastAPI Backend

Start the API using:

uvicorn src.api.main:app --reload

The API will be available at:

http://127.0.0.1:8000

FastAPI documentation:

http://127.0.0.1:8000/docs

The interactive Swagger interface can be used to test the prediction endpoints.

📊 Run the Streamlit Dashboard

Start the dashboard with:

streamlit run src/dashboard/app.py

The dashboard provides an interactive interface for viewing transaction screening and anomaly detection results.

🐳 Docker

The project also includes Docker support.

Build the image:

docker build -t anomaly-detection-system .

Run the container:

docker run -p 8000:8000 anomaly-detection-system

If using Docker Compose:

docker compose up --build
🧪 Testing

The project includes an automated Pytest suite.

Run all tests:

pytest

Current local test status:

14 passed

The CI pipeline additionally validates the project against:

Python 3.10
Python 3.11
🧹 Code Quality

The project uses Black for formatting and Flake8 for static code analysis.

Run Black:

black src/ tests/

Check formatting:

black --check src/ tests/

Run Flake8:

flake8 src/ tests/ --max-line-length=100 --ignore=E501,W503,E203
🔄 Continuous Integration

GitHub Actions automatically validates changes pushed to the repository.

The CI pipeline performs:

Git Push / Pull Request
          │
          ▼
    GitHub Actions
          │
     ┌────┴────┐
     ▼         ▼
   Tests      Lint
     │         │
     ├─ Py3.10 ├─ Black
     └─ Py3.11 └─ Flake8
Current CI checks
✅ Python 3.10 tests
✅ Python 3.11 tests
✅ Black formatting
✅ Flake8 linting
✅ Automated GitHub Actions workflow

View GitHub Actions →

📡 API Workflow

A typical transaction flows through the system as follows:

Transaction
    │
    ▼
FastAPI Endpoint
    │
    ▼
Feature Preparation
    │
    ├───────────────┐
    ▼               ▼
Isolation Forest   Autoencoder
    │               │
    └───────┬───────┘
            ▼
      Ensemble Logic
            │
            ▼
   Risk / Anomaly Result
            │
            ▼
       API Response
📈 Example Prediction Output

A prediction response contains information about the transaction and the resulting anomaly assessment.

Conceptually:

{
  "transaction_id": "example-id",
  "is_anomaly": true,
  "risk_level": "HIGH",
  "anomaly_score": 0.XX
}

The exact response schema is defined by the FastAPI application.

🛡️ Engineering Practices

This project follows several software engineering and MLOps practices:

Modular source-code organization
Separate model implementations
Reusable training pipeline
REST-based model inference
Containerization
Automated unit testing
Continuous Integration
Code formatting
Static code analysis
Reproducible dependency installation
Model artifact persistence
🗺️ Development Roadmap

The original development roadmap has evolved into the current implementation.

Completed
 Repository initialization
 Transaction data generation
 Feature preprocessing
 Isolation Forest implementation
 PyTorch Autoencoder implementation
 Model training pipeline
 Model artifact persistence
 Ensemble anomaly detection
 FastAPI inference service
 Streamlit dashboard
 Automated tests
 Docker support
 GitHub Actions CI
 Black formatting checks
 Flake8 linting
 Python 3.10 CI testing
 Python 3.11 CI testing
Future Improvements
 Expand real-world financial transaction datasets
 Add additional anomaly detection models
 Improve model calibration and threshold selection
 Add experiment tracking
 Add model monitoring and drift detection
 Expand automated integration testing
 Add authentication and API security
 Improve production deployment architecture
🎓 Project Purpose

This project was developed as a practical machine learning engineering project to demonstrate the integration of:

Machine Learning + Software Engineering + APIs + Deployment + MLOps

Rather than focusing only on model training, the project implements the surrounding engineering infrastructure required to take an anomaly detection model toward an end-to-end application.

📚 Research Background

The project is also informed by experience working on secure systems and applied machine learning research.

Publications

MegaShare: A Secure Offline File-Sharing Framework
IEEE ICCCNT 2025

Voice Based Biometric Authentication and AI Assistant
ICASET 2026

The financial anomaly detection system is an independent engineering project focused on practical machine learning, deployment, and MLOps.

👨‍💻 Author
Bhushan Prabhakar Jagtap

B.E. Computer Engineering
Pillai HOC College of Engineering & Technology
Mumbai, India

📧 jagtapbhushan254@gmail.com

🔗 GitHub

⭐ Project Status

Active Development

The core anomaly detection pipeline, model training, API, dashboard, testing infrastructure, containerization, and CI pipeline are implemented. Further improvements are focused on production hardening, model monitoring, and extending the ML pipeline.

🔗 Repository

View the project on GitHub →