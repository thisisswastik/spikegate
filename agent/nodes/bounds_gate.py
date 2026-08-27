"""
agent/nodes/bounds_gate.py — Deterministic, non-LLM guardrail node.

Hard override rules that fire REGARDLESS of what the LLM decided.
This ensures that no LLM hallucination or misclassification can bypass
critical safety thresholds.

Override rules (applied in priority order):
1. spike_score >= HARD_BLOCK_THRESHOLD          → force auto_block
2. spike_score >= HIGH_RISK_THRESHOLD
   AND merchant risk_tier == "high"             → force auto_block
3. spike_score <= ALLOW_THRESHOLD               → force allow

If none of the hard rules fire, the LLM's action passes through unchanged.

All overrides are logged with a reason string in the agent state.

DEFENSE-ONLY: This node is purely deterministic Python — no LLM, no external calls.
"""
from __future__ import annotations

import os

from agent.state import AgentState

# Hard threshold constants — can be overridden via environment variables
_HARD_BLOCK_THRESHOLD = float(os.environ.get("SPIKE_HARD_BLOCK_THRESHOLD", "0.90"))
_HIGH_RISK_THRESHOLD = float(os.environ.get("SPIKE_HIGH_RISK_THRESHOLD", "0.70"))
_ALLOW_THRESHOLD = float(os.environ.get("SPIKE_ALLOW_THRESHOLD", "0.15"))

VALID_ACTIONS = frozenset({"auto_block", "soft_challenge", "flag_for_review", "allow"})


def bounds_gate(state: AgentState) -> AgentState:
    """
    LangGraph node: deterministic hard-override guardrail.

    Reads:
    - detector_output.spike_score
    - merchant_context.risk_tier (if available)
    - llm_action

    Writes:
    - final_action: the (possibly overridden) action
    - gate_override: True if the LLM was overridden
    - gate_override_reason: human-readable reason for any override
    """
    spike_score = state["detector_output"].spike_score
    llm_action = state.get("llm_action", "flag_for_review")

    # Validate LLM action is in the allowed enum (last safety check)
    if llm_action not in VALID_ACTIONS:
        llm_action = "flag_for_review"

    ctx = state.get("merchant_context")
    risk_tier = ctx["risk_tier"] if ctx else "medium"

    # ------------------------------------------------------------------
    # Rule 1: Hard block at very high confidence
    # ------------------------------------------------------------------
    if spike_score >= _HARD_BLOCK_THRESHOLD:
        return {
            **state,
            "final_action": "auto_block",
            "gate_override": True,
            "gate_override_reason": (
                f"spike_score={spike_score:.4f} >= hard block threshold "
                f"{_HARD_BLOCK_THRESHOLD:.2f} → auto_block (deterministic override)"
            ),
        }

    # ------------------------------------------------------------------
    # Rule 2: High-risk merchant at elevated confidence
    # ------------------------------------------------------------------
    if spike_score >= _HIGH_RISK_THRESHOLD and risk_tier == "high":
        return {
            **state,
            "final_action": "auto_block",
            "gate_override": True,
            "gate_override_reason": (
                f"spike_score={spike_score:.4f} >= high-risk threshold "
                f"{_HIGH_RISK_THRESHOLD:.2f} with risk_tier='high' → auto_block (deterministic override)"
            ),
        }

    # ------------------------------------------------------------------
    # Rule 3: Very low score → force allow (suppress low-confidence alerts)
    # ------------------------------------------------------------------
    if spike_score <= _ALLOW_THRESHOLD:
        return {
            **state,
            "final_action": "allow",
            "gate_override": True,
            "gate_override_reason": (
                f"spike_score={spike_score:.4f} <= allow threshold "
                f"{_ALLOW_THRESHOLD:.2f} → allow (deterministic override)"
            ),
        }

    # ------------------------------------------------------------------
    # No override — pass LLM action through
    # ------------------------------------------------------------------
    return {
        **state,
        "final_action": llm_action,
        "gate_override": False,
        "gate_override_reason": None,
    }
