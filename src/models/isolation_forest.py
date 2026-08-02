"""
Isolation Forest Anomaly Detector
===================================
Wraps scikit-learn's IsolationForest with training, 
prediction, persistence, and explainability utilities.

Author: Bhushan Jagtap
"""

import numpy as np
import pandas as pd
import joblib
import os
import logging
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = [
    "amount",
    "hour_of_day",
    "day_of_week",
    "transaction_count_1h",
    "avg_amount_7d",
    "distance_from_home_km",
    "merchant_category_encoded",
]

MODEL_PATH  = "data/models/isolation_forest.joblib"
SCALER_PATH = "data/models/scaler_if.joblib"


class IsolationForestDetector:
    """
    Anomaly detector using Isolation Forest.

    Isolation Forest isolates observations by randomly 
    selecting a feature and then randomly selecting a 
    split value — anomalies need fewer splits to be isolated.
    """

    def __init__(self, contamination: float = 0.02, n_estimators: int = 200):
        """
        Args:
            contamination:  Expected fraction of anomalies in training data
            n_estimators:   Number of trees in the forest
        """
        self.contamination = contamination
        self.n_estimators  = n_estimators
        self.model  = None
        self.scaler = None
        self.is_trained = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, df: pd.DataFrame) -> dict:
        """
        Fit scaler and Isolation Forest on training data.

        Args:
            df: DataFrame with FEATURE_COLUMNS present

        Returns:
            Training summary dict
        """
        logger.info("Training Isolation Forest...")
        X = df[FEATURE_COLUMNS].fillna(0).values

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
            max_features=1.0,
            bootstrap=False,
            random_state=42,
            n_jobs=-1,
        )
        self.model.fit(X_scaled)
        self.is_trained = True

        # Compute training scores
        scores = self.model.score_samples(X_scaled)
        predictions = self.model.predict(X_scaled)
        n_anomalies = (predictions == -1).sum()

        summary = {
            "n_samples":      len(X),
            "n_anomalies":    int(n_anomalies),
            "anomaly_rate":   round(n_anomalies / len(X), 4),
            "score_mean":     round(float(scores.mean()), 4),
            "score_std":      round(float(scores.std()), 4),
            "contamination":  self.contamination,
            "n_estimators":   self.n_estimators,
        }
        logger.info(f"Training complete: {summary}")
        return summary

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, features: dict) -> dict:
        """
        Predict whether a single transaction is anomalous.

        Args:
            features: Dict with transaction feature values

        Returns:
            Dict with is_anomaly, anomaly_score, risk_level
        """
        self._check_trained()
        X = self._features_to_array(features)
        X_scaled = self.scaler.transform(X)

        prediction = self.model.predict(X_scaled)[0]   # 1 = normal, -1 = anomaly
        score      = float(self.model.score_samples(X_scaled)[0])

        is_anomaly = prediction == -1
        risk_level = self._score_to_risk(score)

        return {
            "is_anomaly":     bool(is_anomaly),
            "anomaly_score":  round(score, 4),
            "risk_level":     risk_level,
            "model":          "isolation_forest",
        }

    def predict_batch(self, df: pd.DataFrame) -> pd.DataFrame:
        """Run prediction on a DataFrame."""
        self._check_trained()
        X = df[FEATURE_COLUMNS].fillna(0).values
        X_scaled = self.scaler.transform(X)

        predictions = self.model.predict(X_scaled)
        scores      = self.model.score_samples(X_scaled)

        result = df.copy()
        result["is_anomaly"]    = predictions == -1
        result["anomaly_score"] = scores
        result["risk_level"]    = [self._score_to_risk(s) for s in scores]
        return result

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """Save model and scaler to disk."""
        self._check_trained()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        joblib.dump(self.model,  model_path)
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Model saved to {model_path}")

    def load(self, model_path: str = MODEL_PATH, scaler_path: str = SCALER_PATH):
        """Load model and scaler from disk."""
        self.model      = joblib.load(model_path)
        self.scaler     = joblib.load(scaler_path)
        self.is_trained = True
        logger.info(f"Model loaded from {model_path}")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _features_to_array(self, features: dict) -> np.ndarray:
        return np.array([[features.get(col, 0) for col in FEATURE_COLUMNS]])

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")

    @staticmethod
    def _score_to_risk(score: float) -> str:
        """Convert anomaly score to human-readable risk level."""
        if score < -0.3:   return "HIGH"
        if score < -0.1:   return "MEDIUM"
        return "LOW"
