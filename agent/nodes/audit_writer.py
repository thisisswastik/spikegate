"""
agent/nodes/audit_writer.py — Append-only audit log writer.

Every agent decision is written to audit_log.db (SQLite) BEFORE or AS it fires.
The table is append-only by design: no UPDATE or DELETE statements are issued.

Schema:
  id                   INTEGER PRIMARY KEY AUTOINCREMENT
  timestamp            TEXT     (ISO-8601 UTC)
  entity_id            TEXT
  entity_type          TEXT
  spike_score          REAL
  final_action         TEXT     (one of: auto_block, soft_challenge, flag_for_review, allow)
  llm_action           TEXT     (what the LLM originally proposed)
  gate_override        INTEGER  (0 or 1)
  gate_override_reason TEXT
  explanation          TEXT
  top_features_json    TEXT     (JSON array)
  context_available    INTEGER  (0 or 1)
  merchant_risk_tier   TEXT
  transaction_amount_inr REAL
  payment_id           TEXT
  errors_json          TEXT     (JSON array)

DEFENSE-ONLY: Only INSERT statements. No UPDATE, DELETE, or DROP.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from agent.state import AgentState

_CREATE_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS audit_log (
    id                      INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp               TEXT    NOT NULL,
    entity_id               TEXT    NOT NULL,
    entity_type             TEXT    NOT NULL,
    spike_score             REAL    NOT NULL,
    final_action            TEXT    NOT NULL,
    llm_action              TEXT,
    gate_override           INTEGER NOT NULL DEFAULT 0,
    gate_override_reason    TEXT,
    explanation             TEXT,
    top_features_json       TEXT,
    context_available       INTEGER NOT NULL DEFAULT 1,
    merchant_risk_tier      TEXT,
    transaction_amount_inr  REAL,
    payment_id              TEXT,
    errors_json             TEXT
);
"""

_INSERT_SQL = """
INSERT INTO audit_log (
    timestamp, entity_id, entity_type, spike_score,
    final_action, llm_action, gate_override, gate_override_reason,
    explanation, top_features_json, context_available,
    merchant_risk_tier, transaction_amount_inr, payment_id, errors_json
) VALUES (
    :timestamp, :entity_id, :entity_type, :spike_score,
    :final_action, :llm_action, :gate_override, :gate_override_reason,
    :explanation, :top_features_json, :context_available,
    :merchant_risk_tier, :transaction_amount_inr, :payment_id, :errors_json
)
"""


def _get_db_path() -> str:
    return os.environ.get("AUDIT_DB_PATH", "audit_log.db")


def _ensure_table(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_TABLE_SQL)
    conn.commit()


def audit_writer(state: AgentState) -> AgentState:
    """
    LangGraph node: write the decision to the append-only audit log.

    Reads all relevant fields from state and writes a single row.
    Sets state["audit_id"] and state["audit_written"] = True on success.
    On failure, logs the error but does NOT raise (the decision already fired).
    """
    det = state["detector_output"]
    trigger = det.trigger_transaction
    ctx = state.get("merchant_context")

    row = {
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
        "entity_id": det.entity_id,
        "entity_type": det.entity_type,
        "spike_score": det.spike_score,
        "final_action": state.get("final_action", "flag_for_review"),
        "llm_action": state.get("llm_action"),
        "gate_override": 1 if state.get("gate_override") else 0,
        "gate_override_reason": state.get("gate_override_reason"),
        "explanation": state.get("explanation"),
        "top_features_json": json.dumps(det.top_features),
        "context_available": 1 if state.get("context_available") else 0,
        "merchant_risk_tier": ctx["risk_tier"] if ctx else None,
        "transaction_amount_inr": trigger.amount_inr if trigger else None,
        "payment_id": trigger.payment_id if trigger else None,
        "errors_json": json.dumps(state.get("errors", [])),
    }

    try:
        db_path = _get_db_path()
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(db_path)
        _ensure_table(conn)
        cursor = conn.execute(_INSERT_SQL, row)
        audit_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return {
            **state,
            "audit_id": audit_id,
            "audit_written": True,
        }

    except Exception as e:
        error_msg = f"audit_writer failed: {type(e).__name__}: {e}"
        return {
            **state,
            "audit_id": None,
            "audit_written": False,
            "errors": state.get("errors", []) + [error_msg],
        }
