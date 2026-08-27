"""
agent/nodes/explainer.py — Human-readable explanation generator.

Generates a plain-English explanation of the agent's decision by referencing
the actual top_features that drove the spike score.  Does NOT call an LLM —
the explanation is generated deterministically from structured data so it is
always faithful to the actual features and never hallucinates.

Output format:
  "Action: auto_block | Score: 0.94
   Reason: Blocked because tx_count_5m=47 (contribution: +0.38) and
   fail_rate_5m=0.42 (contribution: +0.21) for merchant mid_XXXX on
   2024-01-01T12:00:00Z. [Gate override: ...]"
"""
from __future__ import annotations

from agent.state import AgentState

# Action → human-readable description
ACTION_DESCRIPTIONS = {
    "auto_block": "Automatically blocked all transactions from this entity",
    "soft_challenge": "Required additional authentication (OTP/3DS) for this entity",
    "flag_for_review": "Flagged for human analyst review; transactions continue",
    "allow": "Allowed; alert assessed as a false positive",
}


def explainer(state: AgentState) -> AgentState:
    """
    LangGraph node: generate a human-readable, feature-grounded explanation.

    Reads:
    - final_action
    - detector_output (spike_score, top_features, entity_type, entity_id)
    - gate_override + gate_override_reason
    - merchant_context (optional)
    - llm_reasoning (optional)

    Writes:
    - explanation: str
    """
    det = state["detector_output"]
    final_action = state.get("final_action", "flag_for_review")
    gate_override = state.get("gate_override", False)
    gate_reason = state.get("gate_override_reason")
    llm_reasoning = state.get("llm_reasoning", "")
    ctx = state.get("merchant_context")

    action_desc = ACTION_DESCRIPTIONS.get(final_action, final_action)

    # Build feature evidence string
    feature_lines = []
    for feat in det.top_features:
        name = feat["name"]
        value = feat["value"]
        contrib = feat["contribution"]
        direction = "↑ spike signal" if contrib > 0 else "↓ anti-spike"
        feature_lines.append(
            f"    • {name} = {value:.4f}  [SHAP: {contrib:+.4f}, {direction}]"
        )
    features_text = "\n".join(feature_lines)

    # Context summary
    if ctx:
        ctx_summary = (
            f"  Merchant risk tier: {ctx['risk_tier']} | "
            f"Historical FP rate: {ctx['historical_fp_rate']:.1%} | "
            f"Chargeback rate: {ctx['chargeback_rate']:.1%}"
        )
    else:
        ctx_summary = "  Merchant context: unavailable"

    # Transaction summary
    trigger = det.trigger_transaction
    tx_summary = ""
    if trigger:
        tx_summary = (
            f"  Transaction: ₹{trigger.amount_inr:,.2f} via {trigger.payment_method} "
            f"(status: {trigger.status})"
        )

    # Override notice
    override_text = ""
    if gate_override:
        override_text = f"\n  ⚠️  DETERMINISTIC OVERRIDE: {gate_reason}"
    elif llm_reasoning:
        override_text = f"\n  LLM reasoning: {llm_reasoning}"

    explanation = (
        f"{'='*60}\n"
        f"DECISION: {final_action.upper()}\n"
        f"  {action_desc}\n"
        f"  Entity: {det.entity_type} = {det.entity_id}\n"
        f"  Spike score: {det.spike_score:.4f}  |  Window: {det.window_seconds//60}min\n"
        f"  Timestamp: {det.timestamp.isoformat()}\n"
        f"{ctx_summary}\n"
        f"{tx_summary}\n"
        f"\nTOP SPIKE FEATURES:\n{features_text}"
        f"{override_text}\n"
        f"{'='*60}"
    )

    return {
        **state,
        "explanation": explanation,
    }
