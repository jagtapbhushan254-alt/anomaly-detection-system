"""
API Endpoint Tests
==================
Tests for FastAPI anomaly detection endpoints.
Run: pytest tests/ -v

Author: Bhushan Jagtap
"""

import pytest
import sys, os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

SAMPLE_NORMAL = {
    "amount": 45.50,
    "merchant_category": "groceries",
    "hour_of_day": 14,
    "day_of_week": 2,
    "transaction_count_1h": 1,
    "avg_amount_7d": 52.00,
    "distance_from_home_km": 3.5,
}

SAMPLE_FRAUD = {
    "amount": 9850.00,
    "merchant_category": "electronics",
    "hour_of_day": 2,
    "day_of_week": 6,
    "transaction_count_1h": 12,
    "avg_amount_7d": 450.00,
    "distance_from_home_km": 1200.0,
}


class TestSchemas:
    """Test Pydantic schema validation."""

    def test_valid_transaction_request(self):
        from src.api.schemas import TransactionRequest

        txn = TransactionRequest(**SAMPLE_NORMAL)
        assert txn.amount == 45.50
        assert txn.merchant_category == "groceries"

    def test_invalid_merchant_category(self):
        from src.api.schemas import TransactionRequest

        with pytest.raises(Exception):
            TransactionRequest(**{**SAMPLE_NORMAL, "merchant_category": "invalid_cat"})

    def test_invalid_hour(self):
        from src.api.schemas import TransactionRequest

        with pytest.raises(Exception):
            TransactionRequest(**{**SAMPLE_NORMAL, "hour_of_day": 25})

    def test_negative_amount(self):
        from src.api.schemas import TransactionRequest

        with pytest.raises(Exception):
            TransactionRequest(**{**SAMPLE_NORMAL, "amount": -100.0})


class TestStreamProducer:
    """Test transaction stream generator."""

    def test_normal_transaction_structure(self):
        from src.producer.stream_producer import generate_normal_transaction

        txn = generate_normal_transaction()
        assert txn.amount > 0
        assert txn.hour_of_day in range(24)
        assert txn.day_of_week in range(7)
        assert txn.is_fraud == False

    def test_fraud_transaction_structure(self):
        from src.producer.stream_producer import generate_fraud_transaction

        txn = generate_fraud_transaction()
        assert txn.amount > 0
        assert txn.is_fraud == True

    def test_stream_generates_correct_count(self):
        from src.producer.stream_producer import generate_stream

        txns = list(generate_stream(delay_seconds=0, total=10))
        assert len(txns) == 10

    def test_fraud_rate_approximately_correct(self):
        from src.producer.stream_producer import generate_stream

        txns = list(generate_stream(fraud_rate=0.5, delay_seconds=0, total=1000))
        fraud_count = sum(1 for t in txns if t.is_fraud)
        # With 50% fraud rate, expect 400-600 fraudulent in 1000
        assert 300 <= fraud_count <= 700

    def test_dataset_generation(self, tmp_path):
        from src.producer.stream_producer import generate_training_dataset

        save_path = str(tmp_path / "test_data.csv")
        df = generate_training_dataset(
            n_samples=100, fraud_rate=0.1, save_path=save_path
        )
        assert len(df) == 100
        assert "is_fraud" in df.columns
        assert df["is_fraud"].sum() > 0


class TestIsolationForest:
    """Test Isolation Forest model."""

    def test_train_and_predict(self):
        from src.producer.stream_producer import generate_training_dataset
        from src.models.isolation_forest import IsolationForestDetector
        import tempfile, os

        df = generate_training_dataset(
            n_samples=500, fraud_rate=0.05, save_path="data/test_transactions.csv"
        )
        model = IsolationForestDetector(contamination=0.05, n_estimators=10)
        summary = model.train(df)

        assert summary["n_samples"] == 500
        assert "anomaly_rate" in summary

        features = {
            "amount": 45.5,
            "hour_of_day": 14,
            "day_of_week": 2,
            "transaction_count_1h": 1,
            "avg_amount_7d": 52.0,
            "distance_from_home_km": 3.5,
            "merchant_category_encoded": 0,
        }
        result = model.predict(features)
        assert "is_anomaly" in result
        assert "anomaly_score" in result
        assert "risk_level" in result
        assert result["risk_level"] in ["LOW", "MEDIUM", "HIGH"]

    def test_untrained_model_raises(self):
        from src.models.isolation_forest import IsolationForestDetector

        model = IsolationForestDetector()
        with pytest.raises(RuntimeError):
            model.predict({"amount": 100})


class TestAutoencoder:
    """Test PyTorch Autoencoder model."""

    def test_model_architecture(self):
        import torch
        from src.models.autoencoder import TransactionAutoencoder

        model = TransactionAutoencoder(input_dim=7)
        x = torch.randn(32, 7)
        out = model(x)
        assert out.shape == (32, 7)

    def test_reconstruction_error_shape(self):
        import torch
        from src.models.autoencoder import TransactionAutoencoder

        model = TransactionAutoencoder(input_dim=7)
        x = torch.randn(16, 7)
        errors = model.reconstruction_error(x)
        assert errors.shape == (16,)
        assert all(e >= 0 for e in errors)

    def test_train_and_predict(self):
        from src.producer.stream_producer import generate_training_dataset
        from src.models.autoencoder import AutoencoderDetector

        df = generate_training_dataset(
            n_samples=300, fraud_rate=0.05, save_path="data/test_ae.csv"
        )
        model = AutoencoderDetector(threshold_percentile=95)
        summary = model.train(df, epochs=5, batch_size=64)

        assert "final_loss" in summary
        assert summary["anomaly_threshold"] > 0

        features = {
            "amount": 45.5,
            "hour_of_day": 14,
            "day_of_week": 2,
            "transaction_count_1h": 1,
            "avg_amount_7d": 52.0,
            "distance_from_home_km": 3.5,
            "merchant_category_encoded": 0,
        }
        result = model.predict(features)
        assert "is_anomaly" in result
        assert "reconstruction_error" in result
        assert result["reconstruction_error"] >= 0
