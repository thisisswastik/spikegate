"""
agent/state.py — LangGraph state schema for the SpikeGate agent graph.

The AgentState TypedDict is passed between all nodes. Each node reads what it
needs and adds/updates the fields it owns.
"""
from __future__ import annotations

from datetime import datetime
from typing import Optional, TypedDict

from data_gen.schema import DetectorOutput


class MerchantContext(TypedDict):
    """Merchant risk context fetched from the SQLite lookup table."""
    merchant_id: str
    risk_tier: str                  # "low" | "medium" | "high"
    avg_transaction_value_inr: float
    historical_fp_rate: float       # fraction of past alerts that were false positives
    chargeback_rate: float          # fraction of past transactions that were chargebacks
    monthly_volume_inr: float


class AgentState(TypedDict):
    """Full state flowing through the LangGraph agent graph."""

    # --- Input (from detector) ---
    detector_output: DetectorOutput

    # --- context_fetch outputs ---
    merchant_context: Optional[MerchantContext]
    context_available: bool         # False if DB lookup timed out / failed

    # --- policy_reasoner outputs ---
    llm_action: Optional[str]       # raw action from LLM (before bounds gate)
    llm_confidence: Optional[float]
    llm_reasoning: Optional[str]

    # --- bounds_gate outputs ---
    final_action: Optional[str]     # one of: auto_block, soft_challenge, flag_for_review, allow
    gate_override: bool             # True if bounds_gate overrode the LLM
    gate_override_reason: Optional[str]

    # --- explainer outputs ---
    explanation: Optional[str]      # human-readable rationale

    # --- audit_writer outputs ---
    audit_id: Optional[int]         # row ID in audit_log.db
    audit_written: bool

    # --- error tracking ---
    errors: list[str]               # accumulated non-fatal errors
