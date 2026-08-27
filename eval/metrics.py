"""
eval/metrics.py — Evaluation metrics and rupee-cost computation.

Computes:
1. Per-action precision/recall/F1 (multi-class, one-vs-rest)
2. False-positive cost in rupees: sum of legitimate transaction amounts wrongly
   actioned (auto_block or soft_challenge)
3. True-positive value in rupees: sum of fraud amounts correctly blocked/challenged
4. Net value = TP value − FP cost

Design notes:
- We map the 4-action output to a binary fraud/not-fraud decision for
  standard precision/recall:
    auto_block + soft_challenge → "actioned" (treated as positive prediction)
    flag_for_review + allow     → "not actioned" (treated as negative prediction)
- Per-action breakdown also shown separately for interpretability.
- All monetary values are in Indian Rupees (INR).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np


VALID_ACTIONS = ["auto_block", "soft_challenge", "flag_for_review", "allow"]

# Which actions constitute a "positive" prediction (i.e., treating as fraud)
ACTIONED_LABELS = {"auto_block", "soft_challenge"}
# Friction cost multiplier for soft_challenge (merchant pays partial friction even if legit)
SOFT_CHALLENGE_FRICTION_RATE = 0.10   # 10% of blocked amount as friction cost


@dataclass
class EvalResult:
    """Container for all evaluation metrics."""

    # Basic counts
    n_total: int = 0
    n_spike: int = 0     # ground-truth spike transactions
    n_normal: int = 0    # ground-truth normal transactions

    # Per-action counts
    action_counts: dict[str, int] = field(default_factory=dict)

    # Binary classification metrics (actioned vs not-actioned)
    true_positives: int = 0    # spike correctly actioned
    false_positives: int = 0   # normal wrongly actioned
    false_negatives: int = 0   # spike not actioned
    true_negatives: int = 0    # normal correctly not actioned

    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0

    # Per-action precision/recall
    per_action_metrics: dict[str, dict] = field(default_factory=dict)

    # Rupee cost/value
    fp_cost_inr: float = 0.0        # legitimate transaction amounts wrongly blocked/challenged
    tp_value_inr: float = 0.0       # fraud amounts correctly caught
    net_value_inr: float = 0.0      # tp_value - fp_cost

    # Failure case demonstration
    failure_cases: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            "n_total": self.n_total,
            "n_spike": self.n_spike,
            "n_normal": self.n_normal,
            "spike_rate_pct": round(100 * self.n_spike / max(self.n_total, 1), 2),
            "action_counts": self.action_counts,
            "binary_metrics": {
                "true_positives": self.true_positives,
                "false_positives": self.false_positives,
                "false_negatives": self.false_negatives,
                "true_negatives": self.true_negatives,
                "precision": round(self.precision, 4),
                "recall": round(self.recall, 4),
                "f1": round(self.f1, 4),
            },
            "per_action_metrics": {
                k: {kk: round(vv, 4) if isinstance(vv, float) else vv
                    for kk, vv in v.items()}
                for k, v in self.per_action_metrics.items()
            },
            "rupee_costs": {
                "fp_cost_inr": round(self.fp_cost_inr, 2),
                "tp_value_inr": round(self.tp_value_inr, 2),
                "net_value_inr": round(self.net_value_inr, 2),
            },
            "failure_cases_count": len(self.failure_cases),
        }


def compute_metrics(
    predictions: list[dict],   # [{action, is_spike, amount_inr, payment_id, ...}]
) -> EvalResult:
    """
    Compute all evaluation metrics from a list of prediction records.

    Each record must have:
    - action: str — one of the 4 valid actions
    - is_spike: bool — ground truth
    - amount_inr: float — transaction amount
    - spike_score: float — detector output score
    """
    result = EvalResult()
    result.n_total = len(predictions)

    if not predictions:
        return result

    # Initialize action counts
    result.action_counts = {a: 0 for a in VALID_ACTIONS}

    for pred in predictions:
        action = pred["action"]
        is_spike = pred["is_spike"]
        amount = pred.get("amount_inr", 0.0)

        if is_spike:
            result.n_spike += 1
        else:
            result.n_normal += 1

        result.action_counts[action] = result.action_counts.get(action, 0) + 1

        # Binary confusion matrix
        actioned = action in ACTIONED_LABELS
        if actioned and is_spike:
            result.true_positives += 1
            result.tp_value_inr += amount
        elif actioned and not is_spike:
            result.false_positives += 1
            # FP cost: blocked amount (conservative — merchant loses the sale)
            if action == "auto_block":
                result.fp_cost_inr += amount
            else:  # soft_challenge
                result.fp_cost_inr += amount * SOFT_CHALLENGE_FRICTION_RATE
        elif not actioned and is_spike:
            result.false_negatives += 1
        else:
            result.true_negatives += 1

    # Binary precision/recall/F1
    tp = result.true_positives
    fp = result.false_positives
    fn = result.false_negatives

    result.precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    result.recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    if result.precision + result.recall > 0:
        result.f1 = (2 * result.precision * result.recall) / (result.precision + result.recall)

    result.net_value_inr = result.tp_value_inr - result.fp_cost_inr

    # Per-action metrics (one-vs-rest)
    for action in VALID_ACTIONS:
        preds_this = [p["action"] == action for p in predictions]
        truths_is_spike = [p["is_spike"] for p in predictions]

        # For auto_block and soft_challenge: TP = spike predicted as this action
        # For flag_for_review/allow: TP = normal predicted as this action (we want FP suppression)
        if action in ACTIONED_LABELS:
            tp_a = sum(1 for p, t in zip(preds_this, truths_is_spike) if p and t)
            fp_a = sum(1 for p, t in zip(preds_this, truths_is_spike) if p and not t)
            fn_a = sum(1 for p, t in zip(preds_this, truths_is_spike) if not p and t)
        else:
            # For non-actioned: TP = normal not actioned
            tp_a = sum(1 for p, t in zip(preds_this, truths_is_spike) if p and not t)
            fp_a = sum(1 for p, t in zip(preds_this, truths_is_spike) if p and t)
            fn_a = sum(1 for p, t in zip(preds_this, truths_is_spike) if not p and not t)

        prec_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        rec_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        f1_a = (2 * prec_a * rec_a) / (prec_a + rec_a) if (prec_a + rec_a) > 0 else 0.0
        count = result.action_counts.get(action, 0)

        result.per_action_metrics[action] = {
            "count": count,
            "precision": prec_a,
            "recall": rec_a,
            "f1": f1_a,
        }

    return result


def format_report(result: EvalResult, run_timestamp: Optional[str] = None) -> str:
    """Format the evaluation result as a Markdown report."""
    ts = run_timestamp or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    spike_rate = 100 * result.n_spike / max(result.n_total, 1)

    lines = [
        "# SpikeGate Evaluation Report",
        f"> Generated: {ts}",
        "",
        "## Dataset",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total transactions | {result.n_total:,} |",
        f"| Spike transactions | {result.n_spike:,} ({spike_rate:.1f}%) |",
        f"| Normal transactions | {result.n_normal:,} |",
        "",
        "## Action Distribution",
        "",
        "| Action | Count | % of Total |",
        "|--------|-------|-----------|",
    ]
    for action, count in result.action_counts.items():
        pct = 100 * count / max(result.n_total, 1)
        lines.append(f"| {action} | {count:,} | {pct:.1f}% |")

    lines += [
        "",
        "## Binary Classification Metrics",
        "> (auto_block + soft_challenge = 'actioned' positive prediction)",
        "",
        "| Metric | Value |",
        "|--------|-------|",
        f"| True Positives | {result.true_positives:,} |",
        f"| False Positives | {result.false_positives:,} |",
        f"| False Negatives | {result.false_negatives:,} |",
        f"| True Negatives | {result.true_negatives:,} |",
        f"| **Precision** | **{result.precision:.4f}** |",
        f"| **Recall** | **{result.recall:.4f}** |",
        f"| **F1 Score** | **{result.f1:.4f}** |",
        "",
        "## Per-Action Metrics",
        "",
        "| Action | Count | Precision | Recall | F1 |",
        "|--------|-------|-----------|--------|-----|",
    ]
    for action, m in result.per_action_metrics.items():
        lines.append(
            f"| {action} | {m['count']:,} | {m['precision']:.4f} | "
            f"{m['recall']:.4f} | {m['f1']:.4f} |"
        )

    lines += [
        "",
        "## False-Positive Cost Analysis (INR)",
        "",
        "> FP cost = blocked legitimate transaction amount (conservative model)",
        "> soft_challenge FP cost = 10% friction of transaction amount",
        "",
        "| Metric | Amount (₹) |",
        "|--------|-----------|",
        f"| False-positive cost (wrongly blocked/challenged) | ₹{result.fp_cost_inr:>12,.2f} |",
        f"| True-positive value (fraud correctly caught) | ₹{result.tp_value_inr:>12,.2f} |",
        f"| **Net value (TP value − FP cost)** | **₹{result.net_value_inr:>12,.2f}** |",
        "",
        "## Defense-Only Statement",
        "",
        "> SpikeGate is strictly defense-only. The agent action space is limited to",
        "> `{auto_block, soft_challenge, flag_for_review, allow}`. No offensive,",
        "> retaliatory, or data-exfiltration capabilities are present anywhere in the codebase.",
        "> All decisions are explainable (SHAP feature attribution) and gated (deterministic",
        "> bounds_gate overrides before any action fires). Every decision is append-logged",
        "> to `audit_log.db` before it takes effect.",
    ]

    if result.failure_cases:
        lines += [
            "",
            "## Failure Case Demonstration",
            "",
            "The following cases demonstrate graceful degradation:",
            "",
        ]
        for i, case in enumerate(result.failure_cases[:3], 1):
            lines.append(f"### Case {i}: {case.get('scenario', 'Unknown')}")
            lines.append(f"- **Input**: {case.get('input', '')}")
            lines.append(f"- **Expected**: {case.get('expected', '')}")
            lines.append(f"- **Actual**: {case.get('actual', '')}")
            lines.append(f"- **Outcome**: {case.get('outcome', '')}")
            lines.append("")

    return "\n".join(lines)
