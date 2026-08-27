"""
agent/nodes/policy_reasoner.py — LLM-based policy decision node.

Uses Google Gemini with structured JSON output to select from the fixed
action enum: {auto_block, soft_challenge, flag_for_review, allow}.

STRICT CONSTRAINTS (enforced in code, not just in the prompt):
1. Action is forced through a JSON schema — the LLM cannot emit free-text actions.
2. If context_available=False → default to "flag_for_review" WITHOUT calling the LLM.
3. If the LLM call fails → default to "flag_for_review" (safe degradation).
4. The model is DEFENSE-ONLY: the system prompt explicitly forbids offense-capable output.

Environment variables required:
  GEMINI_API_KEY — Google AI Studio key
  GEMINI_MODEL   — model name (default: gemini-1.5-flash)
"""
from __future__ import annotations

import json
import os
from typing import Any

from agent.state import AgentState

# Valid actions (the ONLY values the agent may return)
VALID_ACTIONS = frozenset({"auto_block", "soft_challenge", "flag_for_review", "allow"})
SAFE_DEFAULT_ACTION = "flag_for_review"

_SYSTEM_PROMPT = """
You are a fraud risk assessor for a payment gateway. Your role is STRICTLY DEFENSE-ONLY.
You are analyzing a potential fraud spike alert.

YOUR ONLY VALID ACTIONS ARE:
- auto_block: Immediately block all transactions from this entity
- soft_challenge: Require additional authentication (OTP, 3DS) for this entity
- flag_for_review: Queue for human analyst review; allow transactions to continue
- allow: No action required; the alert appears to be a false positive

IMPORTANT RULES:
1. You MUST choose exactly one action from the list above. No other actions are permitted.
2. You MUST NOT suggest offensive, retaliatory, or data-exfiltration actions of any kind.
3. Explain your reasoning by referencing the specific feature values provided.
4. Your output MUST be valid JSON matching the schema: {"action": str, "confidence": float, "reasoning": str}
5. "confidence" must be between 0.0 and 1.0.
6. "reasoning" must reference the actual top feature names and values.
"""


def _build_user_prompt(state: AgentState) -> str:
    """Construct the per-alert user prompt from agent state."""
    det = state["detector_output"]
    ctx = state.get("merchant_context")

    features_text = "\n".join(
        f"  - {f['name']}: {f['value']:.4f} (SHAP contribution: {f['contribution']:+.4f})"
        for f in det.top_features
    )

    ctx_text = "UNAVAILABLE (use flag_for_review as default)" if not state.get("context_available") \
        else f"""
  Risk tier            : {ctx['risk_tier']}
  Avg transaction value: ₹{ctx['avg_transaction_value_inr']:,.2f}
  Historical FP rate   : {ctx['historical_fp_rate']:.1%}
  Chargeback rate      : {ctx['chargeback_rate']:.1%}
  Monthly volume       : ₹{ctx['monthly_volume_inr']:,.0f}"""

    trigger = det.trigger_transaction
    tx_text = ""
    if trigger:
        tx_text = f"""
  Transaction amount   : ₹{trigger.amount_inr:,.2f}
  Payment method       : {trigger.payment_method}
  Transaction status   : {trigger.status}"""

    return f"""
SPIKE ALERT
===========
Entity type       : {det.entity_type}
Entity ID         : {det.entity_id}
Spike score       : {det.spike_score:.4f}
Window            : {det.window_seconds // 60} minutes
Timestamp         : {det.timestamp.isoformat()}

TOP CONTRIBUTING FEATURES:
{features_text}

MERCHANT CONTEXT:
{ctx_text}
{tx_text}

Based on the above spike alert, choose the appropriate defense action.
Remember: ONLY auto_block, soft_challenge, flag_for_review, or allow.
Return ONLY valid JSON: {{"action": "...", "confidence": 0.0-1.0, "reasoning": "..."}}
"""


def _call_gemini(prompt: str) -> dict[str, Any]:
    """Call Gemini API and return parsed JSON response."""
    try:
        import google.generativeai as genai
    except ImportError:
        raise RuntimeError("google-generativeai not installed. Run: pip install google-generativeai")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        raise ValueError("GEMINI_API_KEY is not set. Please set it in .env")

    model_name = os.environ.get("GEMINI_MODEL", "gemini-1.5-flash")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(
        model_name=model_name,
        system_instruction=_SYSTEM_PROMPT,
        generation_config=genai.types.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,   # low temperature for consistent structured output
            max_output_tokens=512,
        ),
    )

    response = model.generate_content(prompt)
    raw_text = response.text.strip()

    # Parse JSON
    parsed = json.loads(raw_text)
    return parsed


def policy_reasoner(state: AgentState) -> AgentState:
    """
    LangGraph node: call the LLM to decide a bounded action.

    If context is unavailable OR the LLM call fails → default to flag_for_review.
    Action is validated against the VALID_ACTIONS enum before being accepted.
    """

    # Graceful degradation: no context → safe default, no LLM call
    if not state.get("context_available", True):
        return {
            **state,
            "llm_action": SAFE_DEFAULT_ACTION,
            "llm_confidence": 0.5,
            "llm_reasoning": "Context unavailable — defaulting to flag_for_review for human review.",
        }

    try:
        user_prompt = _build_user_prompt(state)
        parsed = _call_gemini(user_prompt)

        # Validate action — reject anything not in the enum
        raw_action = str(parsed.get("action", "")).strip().lower()
        if raw_action not in VALID_ACTIONS:
            raise ValueError(
                f"LLM returned invalid action '{raw_action}'. "
                f"Must be one of: {sorted(VALID_ACTIONS)}"
            )

        confidence = float(parsed.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))  # clamp to [0, 1]
        reasoning = str(parsed.get("reasoning", "No reasoning provided."))

        return {
            **state,
            "llm_action": raw_action,
            "llm_confidence": confidence,
            "llm_reasoning": reasoning,
        }

    except Exception as e:
        # Any LLM failure → safe default
        error_msg = f"policy_reasoner failed: {type(e).__name__}: {e}"
        return {
            **state,
            "llm_action": SAFE_DEFAULT_ACTION,
            "llm_confidence": 0.5,
            "llm_reasoning": f"LLM call failed; defaulting to {SAFE_DEFAULT_ACTION}. Error: {e}",
            "errors": state.get("errors", []) + [error_msg],
        }
