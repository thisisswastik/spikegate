"""
agent/nodes/context_fetch.py — Fetch merchant risk context from SQLite lookup table.

Reads the merchant_context.db (pre-seeded at startup) and returns:
  - merchant risk tier
  - historical false-positive rate for this alert pattern
  - chargeback rate
  - average transaction value
  - monthly volume

On any failure (timeout, DB missing, key not found) → sets context_available=False
and logs the error. Downstream nodes must handle context_available=False gracefully
(policy_reasoner defaults to flag_for_review).

DEFENSE-ONLY: This node is purely read-only (SELECT queries only).
"""
from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from agent.state import AgentState, MerchantContext


# Default context used when the DB lookup fails
_DEFAULT_CONTEXT: MerchantContext = {
    "merchant_id": "UNKNOWN",
    "risk_tier": "medium",
    "avg_transaction_value_inr": 1000.0,
    "historical_fp_rate": 0.10,
    "chargeback_rate": 0.02,
    "monthly_volume_inr": 5_000_00.0,
}


def context_fetch(state: AgentState) -> AgentState:
    """
    LangGraph node: fetch merchant context from SQLite.

    Mutates state with:
    - merchant_context: MerchantContext | None
    - context_available: bool
    """
    detector_output = state["detector_output"]
    merchant_id = detector_output.entity_id if detector_output.entity_type == "merchant_id" \
        else (detector_output.trigger_transaction.merchant_id
              if detector_output.trigger_transaction else None)

    db_path = os.environ.get("CONTEXT_DB_PATH", "data/merchant_context.db")

    try:
        if not Path(db_path).exists():
            raise FileNotFoundError(f"Context DB not found: {db_path}")

        conn = sqlite3.connect(db_path, timeout=2.0)  # 2-second timeout
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT merchant_id, risk_tier, avg_transaction_value_inr,
                   historical_fp_rate, chargeback_rate, monthly_volume_inr
            FROM merchant_context
            WHERE merchant_id = ?
            """,
            (merchant_id,),
        )
        row = cursor.fetchone()
        conn.close()

        if row is None:
            # Merchant not in DB → use defaults but still mark available
            context: MerchantContext = {
                **_DEFAULT_CONTEXT,
                "merchant_id": merchant_id or "UNKNOWN",
            }
        else:
            context = MerchantContext(
                merchant_id=row["merchant_id"],
                risk_tier=row["risk_tier"],
                avg_transaction_value_inr=float(row["avg_transaction_value_inr"]),
                historical_fp_rate=float(row["historical_fp_rate"]),
                chargeback_rate=float(row["chargeback_rate"]),
                monthly_volume_inr=float(row["monthly_volume_inr"]),
            )

        return {
            **state,
            "merchant_context": context,
            "context_available": True,
        }

    except Exception as e:
        # Any failure → gracefully degrade; policy_reasoner will default to flag_for_review
        error_msg = f"context_fetch failed: {type(e).__name__}: {e}"
        return {
            **state,
            "merchant_context": None,
            "context_available": False,
            "errors": state.get("errors", []) + [error_msg],
        }
