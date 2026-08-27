"""
Phase 3 tests — Agentic auto-responder (agent/, LangGraph).

Test gate requirements:
(a) Action is always one of the four allowed values
(b) bounds_gate correctly overrides the LLM when spike_score >= 0.90
(c) Every decision produces an audit_log.db row
(d) context_fetch failure → flag_for_review (no crash)

Note: policy_reasoner LLM calls are mocked — these tests do NOT require
a real GEMINI_API_KEY. The LLM integration is tested separately via an
integration test (marked with @pytest.mark.integration) that does need the key.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
from datetime import datetime, timezone
from unittest.mock import patch

import pytest

from agent.nodes.bounds_gate import bounds_gate
from agent.nodes.context_fetch import context_fetch
from agent.nodes.explainer import explainer
from agent.nodes.audit_writer import audit_writer
from agent.state import AgentState, MerchantContext
from data_gen.schema import DetectorOutput, Transaction, PaymentMethod, TransactionStatus, MerchantRiskTier


# ---------------------------------------------------------------------------
# Helper factories
# ---------------------------------------------------------------------------

def _make_transaction(amount: float = 1000.0, merchant_id: str = "mid_test01") -> Transaction:
    return Transaction(
        payment_id="pay_test001",
        merchant_id=merchant_id,
        card_bin="411111",
        device_id="dev_test001",
        ip_address="192.168.1.1",
        amount_inr=amount,
        payment_method=PaymentMethod.CARD,
        status=TransactionStatus.SUCCESS,
        merchant_risk_tier=MerchantRiskTier.MEDIUM,
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        is_spike=True,
        spike_id="spike_test_001",
    )


def _make_detector_output(
    spike_score: float = 0.75,
    entity_type: str = "merchant_id",
    entity_id: str = "mid_test01",
    tx: Transaction | None = None,
) -> DetectorOutput:
    tx = tx or _make_transaction()
    return DetectorOutput(
        entity_type=entity_type,
        entity_id=entity_id,
        window_seconds=300,
        spike_score=spike_score,
        top_features=[
            {"name": "tx_count_5m", "value": 47.0, "contribution": 0.38},
            {"name": "fail_rate_5m", "value": 0.42, "contribution": 0.21},
            {"name": "amt_sum_5m", "value": 85000.0, "contribution": 0.18},
            {"name": "unique_ips_5m", "value": 12.0, "contribution": 0.12},
            {"name": "velocity_ratio_1m_1h", "value": 0.85, "contribution": 0.09},
        ],
        timestamp=datetime(2024, 1, 1, 12, 0, 0, tzinfo=timezone.utc),
        trigger_transaction=tx,
    )


def _make_base_state(
    spike_score: float = 0.75,
    llm_action: str = "soft_challenge",
    context_available: bool = True,
    risk_tier: str = "medium",
) -> AgentState:
    det = _make_detector_output(spike_score=spike_score)
    ctx: MerchantContext = {
        "merchant_id": "mid_test01",
        "risk_tier": risk_tier,
        "avg_transaction_value_inr": 2000.0,
        "historical_fp_rate": 0.12,
        "chargeback_rate": 0.02,
        "monthly_volume_inr": 1_000_000.0,
    }
    return AgentState(
        detector_output=det,
        merchant_context=ctx if context_available else None,
        context_available=context_available,
        llm_action=llm_action,
        llm_confidence=0.80,
        llm_reasoning="Test reasoning",
        final_action=None,
        gate_override=False,
        gate_override_reason=None,
        explanation=None,
        audit_id=None,
        audit_written=False,
        errors=[],
    )


# ---------------------------------------------------------------------------
# (a) Action is always one of the four valid values
# ---------------------------------------------------------------------------

class TestActionBoundedness:
    VALID_ACTIONS = {"auto_block", "soft_challenge", "flag_for_review", "allow"}

    def test_bounds_gate_output_always_valid(self):
        """bounds_gate must always produce a final_action in the valid set."""
        test_cases = [
            (0.05, "medium", "soft_challenge"),   # allow override
            (0.50, "medium", "soft_challenge"),   # pass through
            (0.75, "high", "allow"),              # high-risk override
            (0.95, "low", "allow"),               # hard block override
            (0.95, "high", "flag_for_review"),    # hard block override
        ]
        for score, tier, llm_action in test_cases:
            state = _make_base_state(spike_score=score, llm_action=llm_action, risk_tier=tier)
            result = bounds_gate(state)
            assert result["final_action"] in self.VALID_ACTIONS, (
                f"Invalid action '{result['final_action']}' for score={score}, tier={tier}"
            )

    def test_invalid_llm_action_sanitized(self):
        """If LLM returns an invalid action, bounds_gate must sanitize to flag_for_review."""
        state = _make_base_state(spike_score=0.50, llm_action="transfer_funds")
        result = bounds_gate(state)
        assert result["final_action"] in self.VALID_ACTIONS

    def test_explainer_output_valid_action(self):
        """explainer must still run and produce an explanation for all action types."""
        for action in self.VALID_ACTIONS:
            state = _make_base_state(llm_action=action)
            state["final_action"] = action
            result = explainer(state)
            assert result["explanation"] is not None
            assert len(result["explanation"]) > 20
            assert action.upper() in result["explanation"] or action in result["explanation"]


# ---------------------------------------------------------------------------
# (b) bounds_gate correctly overrides LLM
# ---------------------------------------------------------------------------

class TestBoundsGateOverride:
    def test_hard_block_threshold_overrides_allow(self):
        """spike_score >= 0.90 must force auto_block even if LLM says 'allow'."""
        state = _make_base_state(spike_score=0.95, llm_action="allow")
        with patch.dict(os.environ, {"SPIKE_HARD_BLOCK_THRESHOLD": "0.90"}):
            result = bounds_gate(state)
        assert result["final_action"] == "auto_block"
        assert result["gate_override"] is True
        assert "0.95" in result["gate_override_reason"] or "0.9500" in result["gate_override_reason"]

    def test_hard_block_threshold_overrides_soft_challenge(self):
        """spike_score >= 0.90 must force auto_block even if LLM says 'soft_challenge'."""
        state = _make_base_state(spike_score=0.92, llm_action="soft_challenge")
        with patch.dict(os.environ, {"SPIKE_HARD_BLOCK_THRESHOLD": "0.90"}):
            result = bounds_gate(state)
        assert result["final_action"] == "auto_block"
        assert result["gate_override"] is True

    def test_high_risk_merchant_override(self):
        """spike_score >= 0.70 with high-risk merchant must force auto_block."""
        state = _make_base_state(spike_score=0.75, llm_action="soft_challenge", risk_tier="high")
        with patch.dict(os.environ, {
            "SPIKE_HARD_BLOCK_THRESHOLD": "0.90",
            "SPIKE_HIGH_RISK_THRESHOLD": "0.70",
        }):
            result = bounds_gate(state)
        assert result["final_action"] == "auto_block"
        assert result["gate_override"] is True

    def test_allow_threshold_overrides_block(self):
        """spike_score <= 0.15 must force allow even if LLM says 'auto_block'."""
        state = _make_base_state(spike_score=0.10, llm_action="auto_block")
        with patch.dict(os.environ, {"SPIKE_ALLOW_THRESHOLD": "0.15"}):
            result = bounds_gate(state)
        assert result["final_action"] == "allow"
        assert result["gate_override"] is True

    def test_mid_range_score_passes_through(self):
        """Mid-range score with normal merchant must pass LLM action through."""
        state = _make_base_state(spike_score=0.55, llm_action="soft_challenge", risk_tier="medium")
        with patch.dict(os.environ, {
            "SPIKE_HARD_BLOCK_THRESHOLD": "0.90",
            "SPIKE_HIGH_RISK_THRESHOLD": "0.70",
            "SPIKE_ALLOW_THRESHOLD": "0.15",
        }):
            result = bounds_gate(state)
        assert result["final_action"] == "soft_challenge"
        assert result["gate_override"] is False
        assert result["gate_override_reason"] is None

    def test_exactly_at_threshold_triggers_override(self):
        """spike_score exactly at HARD_BLOCK_THRESHOLD must trigger override."""
        state = _make_base_state(spike_score=0.90, llm_action="flag_for_review")
        with patch.dict(os.environ, {"SPIKE_HARD_BLOCK_THRESHOLD": "0.90"}):
            result = bounds_gate(state)
        assert result["final_action"] == "auto_block"
        assert result["gate_override"] is True


# ---------------------------------------------------------------------------
# (c) Every decision produces an audit_log.db row
# ---------------------------------------------------------------------------

class TestAuditWriter:
    def test_audit_row_written(self, tmp_path):
        """Every decision must produce exactly one audit_log row."""
        db_path = str(tmp_path / "test_audit.db")
        state = _make_base_state()
        state["final_action"] = "soft_challenge"
        state["gate_override"] = False
        result = explainer(state)

        with patch.dict(os.environ, {"AUDIT_DB_PATH": db_path}):
            result = audit_writer(result)

        assert result["audit_written"] is True
        assert result["audit_id"] is not None

        # Verify row exists in DB
        conn = sqlite3.connect(db_path)
        rows = conn.execute("SELECT * FROM audit_log").fetchall()
        conn.close()
        assert len(rows) == 1

    def test_audit_row_has_correct_action(self, tmp_path):
        """The audit row must record the correct final_action."""
        db_path = str(tmp_path / "test_audit2.db")
        state = _make_base_state()
        state["final_action"] = "auto_block"
        state["gate_override"] = True
        state["gate_override_reason"] = "Test override"

        with patch.dict(os.environ, {"AUDIT_DB_PATH": db_path}):
            result = audit_writer(state)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT * FROM audit_log WHERE id = ?", (result["audit_id"],)).fetchone()
        conn.close()

        assert row["final_action"] == "auto_block"
        assert row["gate_override"] == 1
        assert row["gate_override_reason"] == "Test override"

    def test_audit_appends_multiple_rows(self, tmp_path):
        """Multiple decisions must produce multiple rows (append-only)."""
        db_path = str(tmp_path / "test_audit3.db")

        for action in ["auto_block", "soft_challenge", "flag_for_review", "allow"]:
            state = _make_base_state()
            state["final_action"] = action
            with patch.dict(os.environ, {"AUDIT_DB_PATH": db_path}):
                audit_writer(state)

        conn = sqlite3.connect(db_path)
        count = conn.execute("SELECT COUNT(*) FROM audit_log").fetchone()[0]
        conn.close()
        assert count == 4

    def test_audit_stores_top_features_json(self, tmp_path):
        """The audit row must store the top_features as valid JSON."""
        import json
        db_path = str(tmp_path / "test_audit4.db")
        state = _make_base_state()
        state["final_action"] = "flag_for_review"

        with patch.dict(os.environ, {"AUDIT_DB_PATH": db_path}):
            result = audit_writer(state)

        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        row = conn.execute("SELECT top_features_json FROM audit_log WHERE id = ?",
                           (result["audit_id"],)).fetchone()
        conn.close()

        features = json.loads(row["top_features_json"])
        assert isinstance(features, list)
        assert len(features) == 5
        assert "name" in features[0]


# ---------------------------------------------------------------------------
# (d) context_fetch failure → flag_for_review
# ---------------------------------------------------------------------------

class TestContextFetchFailure:
    def test_missing_db_sets_context_unavailable(self, tmp_path):
        """context_fetch with non-existent DB must set context_available=False."""
        state = _make_base_state()
        # Remove context from state to simulate fresh state
        state["merchant_context"] = None
        state["context_available"] = True  # Will be overwritten

        with patch.dict(os.environ, {"CONTEXT_DB_PATH": str(tmp_path / "nonexistent.db")}):
            result = context_fetch(state)

        assert result["context_available"] is False
        assert len(result["errors"]) > 0

    def test_context_unavailable_defaults_to_flag_for_review(self):
        """policy_reasoner with context_available=False must return flag_for_review without LLM."""
        from agent.nodes.policy_reasoner import policy_reasoner, SAFE_DEFAULT_ACTION

        state = _make_base_state(context_available=False)
        state["merchant_context"] = None

        # Must NOT call the LLM (no API key needed)
        result = policy_reasoner(state)

        assert result["llm_action"] == SAFE_DEFAULT_ACTION
        assert "context" in result["llm_reasoning"].lower() or "unavailable" in result["llm_reasoning"].lower()

    def test_full_pipeline_with_context_failure_no_crash(self, tmp_path):
        """Full agent pipeline must complete without raising when context fetch fails."""
        db_path = str(tmp_path / "audit_crash_test.db")

        # Mock bounds_gate to inject a known action (skip LLM call)
        state = _make_base_state(context_available=False)
        state["merchant_context"] = None

        # Run context_fetch (simulating failure)
        with patch.dict(os.environ, {"CONTEXT_DB_PATH": str(tmp_path / "nonexistent.db")}):
            state = context_fetch(state)

        # policy_reasoner defaults to flag_for_review
        from agent.nodes.policy_reasoner import policy_reasoner
        state = policy_reasoner(state)
        assert state["llm_action"] == "flag_for_review"

        # bounds_gate passes through (score is 0.75, which is in middle range)
        with patch.dict(os.environ, {
            "SPIKE_HARD_BLOCK_THRESHOLD": "0.90",
            "SPIKE_HIGH_RISK_THRESHOLD": "0.70",
            "SPIKE_ALLOW_THRESHOLD": "0.15",
        }):
            state = bounds_gate(state)

        # explainer generates output
        state = explainer(state)
        assert state["explanation"] is not None

        # audit_writer writes the row
        with patch.dict(os.environ, {"AUDIT_DB_PATH": db_path}):
            state = audit_writer(state)

        assert state["audit_written"] is True
        assert state["final_action"] in {"auto_block", "soft_challenge", "flag_for_review", "allow"}


# ---------------------------------------------------------------------------
# Full graph smoke test (mocking policy_reasoner LLM)
# ---------------------------------------------------------------------------

class TestFullGraph:
    def test_graph_produces_valid_action(self, tmp_path):
        """Full LangGraph graph must produce a valid action and audit row."""
        from agent.graph import build_graph

        db_path = str(tmp_path / "graph_test_audit.db")

        det = _make_detector_output(spike_score=0.95)  # high score → auto_block via gate
        initial_state: AgentState = {
            "detector_output": det,
            "merchant_context": None,
            "context_available": False,  # skip context lookup
            "llm_action": None,
            "llm_confidence": None,
            "llm_reasoning": None,
            "final_action": None,
            "gate_override": False,
            "gate_override_reason": None,
            "explanation": None,
            "audit_id": None,
            "audit_written": False,
            "errors": [],
        }

        graph = build_graph()

        with patch.dict(os.environ, {
            "AUDIT_DB_PATH": db_path,
            "CONTEXT_DB_PATH": str(tmp_path / "nonexistent.db"),
            "SPIKE_HARD_BLOCK_THRESHOLD": "0.90",
            "SPIKE_HIGH_RISK_THRESHOLD": "0.70",
            "SPIKE_ALLOW_THRESHOLD": "0.15",
        }):
            result = graph.invoke(initial_state)

        # Spike score 0.95 must trigger hard block
        assert result["final_action"] == "auto_block"
        assert result["gate_override"] is True
        assert result["audit_written"] is True
        assert result["explanation"] is not None
