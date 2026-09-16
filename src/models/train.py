"""
Training Pipeline — Isolation Forest + Autoencoder
====================================================
Run this script once to train both models and save
artifacts to data/models/ directory.

Usage:
    python src/models/train.py

Author: Bhushan Jagtap
"""

from models.autoencoder import AutoencoderDetector
from models.isolation_forest import IsolationForestDetector
from producer.stream_producer import generate_training_dataset
import logging
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s"
)
logger = logging.getLogger(__name__)


def train_all(n_samples: int = 10000, fraud_rate: float = 0.02):
    """
    Full training pipeline:
    1. Generate synthetic transaction dataset
    2. Train Isolation Forest
    3. Train PyTorch Autoencoder
    4. Save both models + scalers
    5. Print evaluation summary
    """
    print("\n" + "=" * 60)
    print("  ANOMALY DETECTION SYSTEM — TRAINING PIPELINE")
    print("  Author: Bhushan Prabhakar Jagtap")
    print("=" * 60 + "\n")

    # ── Step 1: Generate data ────────────────────────────────────────
    logger.info(
        f"Step 1/4: Generating {n_samples} transactions (fraud_rate={fraud_rate:.1%})"
    )
    df = generate_training_dataset(
        n_samples=n_samples, fraud_rate=fraud_rate, save_path="data/transactions.csv"
    )
    print(
        f" ✅ Dataset: {len(df)} rows | "
        f"{df['is_fraud'].sum()} fraud ({df['is_fraud'].mean():.1%})\n"
    )

    # ── Step 2: Train Isolation Forest ──────────────────────────────
    logger.info("Step 2/4: Training Isolation Forest...")
    if_model = IsolationForestDetector(contamination=fraud_rate, n_estimators=200)
    if_summary = if_model.train(df)
    if_model.save()
    print(
        f"  ✅ Isolation Forest trained | "
        f"Anomaly rate: {if_summary['anomaly_rate']:.1%} | "
        f"Trees: {if_summary['n_estimators']}\n"
    )

    # ── Step 3: Train Autoencoder ────────────────────────────────────
    logger.info("Step 3/4: Training PyTorch Autoencoder...")
    ae_model = AutoencoderDetector(threshold_percentile=95)
    ae_summary = ae_model.train(df, epochs=50, batch_size=256, lr=1e-3)
    ae_model.save()
    print(
        f"  ✅ Autoencoder trained | "
        f"Final loss: {ae_summary['final_loss']:.6f} | "
        f"Threshold: {ae_summary['anomaly_threshold']:.6f}\n"
    )

    # ── Step 4: Quick evaluation ─────────────────────────────────────
    logger.info("Step 4/4: Quick evaluation on test samples...")

    # Test on 5 normal + 5 fraud transactions

    normal_samples = df[~df["is_fraud"]].head(5)
    fraud_samples = df[df["is_fraud"]].head(5)

    print("\n  SAMPLE PREDICTIONS:")
    print(
        f"  {'Type':<10} {'IF Score':<12} {'AE Error':<12} "
        f"{'IF Flag':<10} {'AE Flag'}"
    )
    print("  " + "-" * 55)

    for _, row in normal_samples.iterrows():
        features = row.to_dict()
        if_pred = if_model.predict(features)
        ae_pred = ae_model.predict(features)

        print(
            f"  {'NORMAL':<10} "
            f"{if_pred['anomaly_score']:<12.4f} "
            f"{ae_pred['reconstruction_error']:<12.6f} "
            f"{'⚠️' if if_pred['is_anomaly'] else '✅':<10} "
            f"{'⚠️' if ae_pred['is_anomaly'] else '✅'}"
        )

    for _, row in fraud_samples.iterrows():
        features = row.to_dict()
        if_pred = if_model.predict(features)
        ae_pred = ae_model.predict(features)

        print(
            f"  {'FRAUD':<10} "
            f"{if_pred['anomaly_score']:<12.4f} "
            f"{ae_pred['reconstruction_error']:<12.6f} "
            f"{'🚨' if if_pred['is_anomaly'] else '❌':<10} "
            f"{'🚨' if ae_pred['is_anomaly'] else '❌'}"
        )
