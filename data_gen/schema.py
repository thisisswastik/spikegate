"""
data_gen/schema.py — Pydantic models for the SpikeGate transaction stream.

All monetary amounts are in Indian Rupees (INR).
"""
from __future__ import annotations

from enum import Enum
from typing import Optional
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field


class PaymentMethod(str, Enum):
    UPI = "upi"
    CARD = "card"
    NETBANKING = "netbanking"
    WALLET = "wallet"
    EMI = "emi"


class MerchantRiskTier(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class TransactionStatus(str, Enum):
    SUCCESS = "success"
    FAILED = "failed"
    PENDING = "pending"


class Transaction(BaseModel):
    """A single Razorpay-style payment transaction."""

    # Identity
    payment_id: str = Field(description="Unique payment identifier, e.g. pay_XXXXX")
    merchant_id: str = Field(description="Merchant identifier, e.g. mid_XXXXX")

    # Entity keys used for velocity tracking
    card_bin: Optional[str] = Field(
        default=None,
        description="First 6 digits of card number (BIN). Null for non-card payments.",
    )
    device_id: str = Field(description="Hashed device fingerprint")
    ip_address: str = Field(description="IPv4 address of the payer")

    # Transaction details
    amount_inr: float = Field(description="Transaction amount in INR", gt=0)
    payment_method: PaymentMethod
    status: TransactionStatus

    # Merchant context
    merchant_risk_tier: MerchantRiskTier

    # Timing
    timestamp: datetime = Field(description="UTC timestamp of the transaction event")

    # Ground-truth label (set by spike injector; absent in production)
    is_spike: bool = Field(
        default=False,
        description="True if this transaction was injected as part of a spike burst.",
    )
    spike_id: Optional[str] = Field(
        default=None,
        description="Identifier of the spike burst this transaction belongs to.",
    )

    model_config = ConfigDict(use_enum_values=True)


class SpikeBurst(BaseModel):
    """Metadata for a single injected spike event."""

    spike_id: str = Field(description="Unique identifier for this spike burst")
    entity_type: str = Field(
        description="Which entity dimension was spiked: merchant_id, card_bin, device_id, ip_address"
    )
    entity_id: str = Field(description="The specific entity value that was spiked")
    start_time: datetime
    end_time: datetime
    multiplier: float = Field(
        description="Velocity multiplier applied during this spike (e.g. 10.0 = 10x normal rate)"
    )
    n_transactions: int = Field(description="Number of transactions injected in this burst")


class DetectorOutput(BaseModel):
    """Output contract from the detector pipeline."""

    entity_type: str
    entity_id: str
    window_seconds: int = Field(description="Rolling window size in seconds (60, 300, or 3600)")
    spike_score: float = Field(ge=0.0, le=1.0, description="Ensemble spike probability score")
    top_features: list[dict] = Field(
        description="Top-5 features by |contribution|: [{name, value, contribution}]"
    )
    timestamp: datetime
    # Carry the triggering transaction through for the agent
    trigger_transaction: Optional[Transaction] = None
