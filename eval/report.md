# SpikeGate Evaluation Report
> Generated: 2026-08-27T16:57:17Z

## Dataset

| Metric | Value |
|--------|-------|
| Total transactions | 18,107 |
| Spike transactions | 860 (4.7%) |
| Normal transactions | 17,247 |

## Action Distribution

| Action | Count | % of Total |
|--------|-------|-----------|
| auto_block | 0 | 0.0% |
| soft_challenge | 1,573 | 8.7% |
| flag_for_review | 10,487 | 57.9% |
| allow | 6,047 | 33.4% |

## Binary Classification Metrics
> (auto_block + soft_challenge = 'actioned' positive prediction)

| Metric | Value |
|--------|-------|
| True Positives | 698 |
| False Positives | 875 |
| False Negatives | 162 |
| True Negatives | 16,372 |
| **Precision** | **0.4437** |
| **Recall** | **0.8116** |
| **F1 Score** | **0.5738** |

## Per-Action Metrics

| Action | Count | Precision | Recall | F1 |
|--------|-------|-----------|--------|-----|
| auto_block | 0 | 0.0000 | 0.0000 | 0.0000 |
| soft_challenge | 1,573 | 0.4437 | 0.8116 | 0.5738 |
| flag_for_review | 10,487 | 0.9858 | 0.5994 | 0.7455 |
| allow | 6,047 | 0.9979 | 0.3499 | 0.5181 |

## False-Positive Cost Analysis (INR)

> FP cost = blocked legitimate transaction amount (conservative model)
> soft_challenge FP cost = 10% friction of transaction amount

| Metric | Amount (₹) |
|--------|-----------|
| False-positive cost (wrongly blocked/challenged) | ₹  241,492.28 |
| True-positive value (fraud correctly caught) | ₹3,530,294.94 |
| **Net value (TP value − FP cost)** | **₹3,288,802.66** |

## Defense-Only Statement

> SpikeGate is strictly defense-only. The agent action space is limited to
> `{auto_block, soft_challenge, flag_for_review, allow}`. No offensive,
> retaliatory, or data-exfiltration capabilities are present anywhere in the codebase.
> All decisions are explainable (SHAP feature attribution) and gated (deterministic
> bounds_gate overrides before any action fires). Every decision is append-logged
> to `audit_log.db` before it takes effect.