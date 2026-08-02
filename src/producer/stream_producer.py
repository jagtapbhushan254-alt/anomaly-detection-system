"""
Real-Time Transaction Stream Producer
======================================
Simulates a financial transaction stream for anomaly detection.
Designed to be Kafka-ready — swap generate_stream() output to a
Kafka producer with minimal changes.

Author: Bhushan Jagtap
"""

import numpy as np
import pandas as pd
import time
import uuid
from dataclasses import dataclass, asdict
from typing import Generator
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

MERCHANT_CATEGORIES = [
    "groceries", "electronics", "restaurant", "fuel",
    "travel", "healthcare", "entertainment", "retail"
]

# Fraud patterns — unusual but realistic
FRAUD_PATTERNS = {
    "high_amount_night":   {"amount_range": (5000, 50000), "hour_range": (0, 5)},
    "rapid_transactions":  {"txn_count_1h": (10, 50),     "amount_range": (100, 500)},
    "distant_location":    {"distance_km":  (800, 5000),   "amount_range": (200, 2000)},
}


@dataclass
class Transaction:
    transaction_id: str
    amount: float
    merchant_category: str
    hour_of_day: int
    day_of_week: int
    transaction_count_1h: int
    avg_amount_7d: float
    distance_from_home_km: float
    is_fraud: bool  # Ground truth label (for evaluation only)
    timestamp: float


def generate_normal_transaction() -> Transaction:
    """Generate a typical, legitimate transaction."""
    return Transaction(
        transaction_id=str(uuid.uuid4())[:12],
        amount=round(np.random.lognormal(mean=4.5, sigma=1.2), 2),
        merchant_category=np.random.choice(MERCHANT_CATEGORIES),
        hour_of_day=int(np.random.choice(range(24), p=_hour_weights())),
        day_of_week=int(np.random.randint(0, 7)),
        transaction_count_1h=int(np.random.poisson(lam=2)),
        avg_amount_7d=round(np.random.lognormal(mean=4.0, sigma=0.8), 2),
        distance_from_home_km=round(abs(np.random.normal(loc=15, scale=20)), 2),
        is_fraud=False,
        timestamp=time.time()
    )


def generate_fraud_transaction() -> Transaction:
    """Generate a fraudulent transaction using known fraud patterns."""
    pattern_name = np.random.choice(list(FRAUD_PATTERNS.keys()))
    pattern = FRAUD_PATTERNS[pattern_name]

    amount = round(np.random.uniform(
        *pattern.get("amount_range", (500, 5000))
    ), 2)

    hour = int(np.random.uniform(
        *pattern.get("hour_range", (0, 23))
    )) if "hour_range" in pattern else int(np.random.randint(0, 24))

    txn_count = int(np.random.uniform(
        *pattern.get("txn_count_1h", (1, 3))
    ))

    distance = round(np.random.uniform(
        *pattern.get("distance_km", (50, 200))
    ), 2)

    return Transaction(
        transaction_id=str(uuid.uuid4())[:12],
        amount=amount,
        merchant_category=np.random.choice(MERCHANT_CATEGORIES),
        hour_of_day=hour,
        day_of_week=int(np.random.randint(0, 7)),
        transaction_count_1h=txn_count,
        avg_amount_7d=round(np.random.lognormal(mean=4.0, sigma=0.8), 2),
        distance_from_home_km=distance,
        is_fraud=True,
        timestamp=time.time()
    )


def _hour_weights() -> list:
    """Business hours have higher transaction probability."""
    weights = np.ones(24)
    weights[8:20] *= 4   # Business hours: 8am - 8pm
    weights[12:14] *= 2  # Lunch peak
    return (weights / weights.sum()).tolist()


def generate_stream(
    fraud_rate: float = 0.02,
    delay_seconds: float = 0.1,
    total: int = None
) -> Generator[Transaction, None, None]:
    """
    Infinite (or bounded) transaction stream generator.

    Args:
        fraud_rate:     Fraction of transactions that are fraudulent (default 2%)
        delay_seconds:  Simulated stream delay between transactions
        total:          Stop after N transactions (None = infinite)

    Yields:
        Transaction dataclass instances
    """
    count = 0
    logger.info(f"Stream started | fraud_rate={fraud_rate:.1%} | delay={delay_seconds}s")

    while total is None or count < total:
        is_fraud = np.random.random() < fraud_rate
        txn = generate_fraud_transaction() if is_fraud else generate_normal_transaction()

        if is_fraud:
            logger.warning(f"⚠️  FRAUD transaction generated: {txn.transaction_id} | ${txn.amount}")

        yield txn
        count += 1

        if delay_seconds > 0:
            time.sleep(delay_seconds)


def generate_training_dataset(
    n_samples: int = 10000,
    fraud_rate: float = 0.02,
    save_path: str = "data/transactions.csv"
) -> pd.DataFrame:
    """
    Generate a labeled dataset for model training.

    Args:
        n_samples:  Number of transactions to generate
        fraud_rate: Fraction that are fraudulent
        save_path:  Where to save the CSV

    Returns:
        DataFrame with transaction features and fraud label
    """
    logger.info(f"Generating {n_samples} training samples...")
    transactions = []

    for txn in generate_stream(fraud_rate=fraud_rate, delay_seconds=0, total=n_samples):
        transactions.append(asdict(txn))

    df = pd.DataFrame(transactions)

    # Encode categorical feature
    df["merchant_category_encoded"] = pd.Categorical(
        df["merchant_category"]
    ).codes

    logger.info(
        f"Dataset ready: {len(df)} rows | "
        f"{df['is_fraud'].sum()} fraudulent ({df['is_fraud'].mean():.1%})"
    )

    import os
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    logger.info(f"Saved to {save_path}")

    return df


if __name__ == "__main__":
    # Quick demo
    print("🚀 Starting transaction stream (5 transactions)...\n")
    for i, txn in enumerate(generate_stream(fraud_rate=0.3, delay_seconds=0.5, total=5)):
        label = "🔴 FRAUD" if txn.is_fraud else "🟢 NORMAL"
        print(f"{label} | ID: {txn.transaction_id} | Amount: ${txn.amount:,.2f} | "
              f"Hour: {txn.hour_of_day}:00 | Distance: {txn.distance_from_home_km}km")
