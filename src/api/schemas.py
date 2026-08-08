"""
Pydantic Schemas — Request & Response Models
=============================================
Author: Bhushan Jagtap
"""

from pydantic import BaseModel, Field, field_validator
from typing import Optional
from enum import Enum


class RiskLevel(str, Enum):
    LOW    = "LOW"
    MEDIUM = "MEDIUM"
    HIGH   = "HIGH"


class TransactionRequest(BaseModel):
    amount: float = Field(..., gt=0)
    merchant_category: str
    hour_of_day: int = Field(..., ge=0, le=23)
    day_of_week: int = Field(..., ge=0, le=6)
    transaction_count_1h: int = Field(..., ge=0)
    avg_amount_7d: float = Field(..., gt=0)
    distance_from_home_km: float = Field(..., ge=0)

    @field_validator("merchant_category")
    @classmethod
    def validate_category(cls, v):
        valid = ["groceries","electronics","restaurant","fuel",
                 "travel","healthcare","entertainment","retail"]
        if v not in valid:
            raise ValueError(f"merchant_category must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
                "amount": 9850.00,
                "merchant_category": "electronics",
                "hour_of_day": 2,
                "day_of_week": 6,
                "transaction_count_1h": 8,
                "avg_amount_7d": 450.00,
                "distance_from_home_km": 1200.0
            }
        }
    }


class PredictionResponse(BaseModel):
    transaction_id: str
    is_anomaly: bool
    anomaly_score: float
    reconstruction_error: float
    ensemble_confidence: float
    risk_level: RiskLevel
    processing_time_ms: float


class BatchRequest(BaseModel):
    transactions: list[TransactionRequest]


class BatchResponse(BaseModel):
    total: int
    anomalies_detected: int
    anomaly_rate: float
    results: list[PredictionResponse]
    processing_time_ms: float


class HealthResponse(BaseModel):
    status: str
    models_loaded: bool
    isolation_forest: bool
    autoencoder: bool
    version: str = "1.0.0"


class MetricsResponse(BaseModel):
    isolation_forest: dict
    autoencoder: dict
    ensemble: dict
