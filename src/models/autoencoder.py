"""
PyTorch Autoencoder for Anomaly Detection
==========================================
Learns compressed representation of NORMAL transactions.
High reconstruction error = anomalous transaction.

Architecture: 7 → 16 → 8 → 4 → 8 → 16 → 7
Trained only on normal data (unsupervised anomaly detection).

Author: Bhushan Jagtap
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib
import os
import logging

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

MODEL_PATH = "data/models/autoencoder.pt"
SCALER_PATH = "data/models/scaler_ae.joblib"
THRESHOLD_PATH = "data/models/ae_threshold.joblib"


class TransactionAutoencoder(nn.Module):
    """
    Symmetric autoencoder with bottleneck for anomaly detection.
    Encoder compresses 7D input → 4D latent space.
    Decoder reconstructs 4D → 7D.
    Anomaly score = MSE reconstruction error.
    """

    def __init__(self, input_dim: int = 7):
        super().__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 4),
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(
            nn.Linear(4, 8),
            nn.ReLU(),
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, input_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))

    def reconstruction_error(self, x: torch.Tensor) -> torch.Tensor:
        """Return per-sample MSE reconstruction error."""
        reconstructed = self.forward(x)
        return torch.mean((x - reconstructed) ** 2, dim=1)


class AutoencoderDetector:
    """
    Wrapper around TransactionAutoencoder with training,
    prediction, and persistence utilities.
    """

    def __init__(self, threshold_percentile: float = 95):
        self.model = TransactionAutoencoder()
        self.scaler = None
        self.threshold = None
        self.threshold_percentile = threshold_percentile
        self.is_trained = False
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    # ── Training ─────────────────────────────────────────────────────────────

    def train(
        self,
        df: pd.DataFrame,
        epochs: int = 50,
        batch_size: int = 256,
        lr: float = 1e-3,
    ) -> dict:
        """
        Train autoencoder on NORMAL transactions only.
        Anomaly threshold = 95th percentile of training errors.
        """
        logger.info("Training Autoencoder on normal transactions only...")

        # Use only normal transactions for training
        normal_df = df[df["is_fraud"] == False] if "is_fraud" in df.columns else df
        X = normal_df[FEATURE_COLUMNS].fillna(0).values.astype(np.float32)

        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X).astype(np.float32)

        dataset = torch.FloatTensor(X_scaled).to(self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )
        criterion = nn.MSELoss()

        self.model.train()
        losses = []

        for epoch in range(epochs):
            # Mini-batch training
            perm = torch.randperm(len(dataset))
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(dataset), batch_size):
                batch = dataset[perm[i : i + batch_size]]
                optimizer.zero_grad()
                reconstructed = self.model(batch)
                loss = criterion(reconstructed, batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()
                n_batches += 1

            avg_loss = epoch_loss / n_batches
            losses.append(avg_loss)
            scheduler.step(avg_loss)

            if (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.6f}")

        # Set anomaly threshold from training errors
        self.model.eval()
        with torch.no_grad():
            errors = self.model.reconstruction_error(dataset).cpu().numpy()

        self.threshold = float(np.percentile(errors, self.threshold_percentile))
        self.is_trained = True

        summary = {
            "n_normal_samples": len(X),
            "epochs": epochs,
            "final_loss": round(losses[-1], 6),
            "anomaly_threshold": round(self.threshold, 6),
            "threshold_percentile": self.threshold_percentile,
        }
        logger.info(f"Training complete: {summary}")
        return summary

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, features: dict) -> dict:
        """Predict anomaly for a single transaction."""
        self._check_trained()
        X = np.array([[features.get(c, 0) for c in FEATURE_COLUMNS]], dtype=np.float32)
        X_scaled = self.scaler.transform(X).astype(np.float32)
        tensor = torch.FloatTensor(X_scaled).to(self.device)

        self.model.eval()
        with torch.no_grad():
            error = float(self.model.reconstruction_error(tensor).cpu().numpy()[0])

        is_anomaly = error > self.threshold

        return {
            "is_anomaly": bool(is_anomaly),
            "reconstruction_error": round(error, 6),
            "threshold": round(self.threshold, 6),
            "model": "autoencoder",
        }

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(
        self,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold_path=THRESHOLD_PATH,
    ):
        self._check_trained()
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(self.model.state_dict(), model_path)
        joblib.dump(self.scaler, scaler_path)
        joblib.dump(self.threshold, threshold_path)
        logger.info(f"Autoencoder saved to {model_path}")

    def load(
        self,
        model_path=MODEL_PATH,
        scaler_path=SCALER_PATH,
        threshold_path=THRESHOLD_PATH,
    ):
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        self.scaler = joblib.load(scaler_path)
        self.threshold = joblib.load(threshold_path)
        self.is_trained = True
        logger.info(f"Autoencoder loaded from {model_path}")

    def _check_trained(self):
        if not self.is_trained:
            raise RuntimeError("Model not trained. Call train() or load() first.")
