# SpikeGate Evaluation Report
> Generated: 2026-08-27T14:33:38Z

## Dataset

| Metric | Value |
|--------|-------|
| Total transactions | 100 |
| Spike transactions | 97 (97.0%) |
| Normal transactions | 3 |

## Action Distribution

| Action | Count | % of Total |
|--------|-------|-----------|
| auto_block | 0 | 0.0% |
| soft_challenge | 71 | 71.0% |
| flag_for_review | 29 | 29.0% |
| allow | 0 | 0.0% |

## Binary Classification Metrics
> (auto_block + soft_challenge = 'actioned' positive prediction)

| Metric | Value |
|--------|-------|
| True Positives | 70 |
| False Positives | 1 |
| False Negatives | 27 |
| True Negatives | 2 |
| **Precision** | **0.9859** |
| **Recall** | **0.7216** |
| **F1 Score** | **0.8333** |

## Per-Action Metrics

| Action | Count | Precision | Recall | F1 |
|--------|-------|-----------|--------|-----|
| auto_block | 0 | 0.0000 | 0.0000 | 0.0000 |
| soft_challenge | 71 | 0.9859 | 0.7216 | 0.8333 |
| flag_for_review | 29 | 0.0690 | 0.6667 | 0.1250 |
| allow | 0 | 0.0000 | 0.0000 | 0.0000 |

## False-Positive Cost Analysis (INR)

> FP cost = blocked legitimate transaction amount (conservative model)
> soft_challenge FP cost = 10% friction of transaction amount

| Metric | Amount (₹) |
|--------|-----------|
| False-positive cost (wrongly blocked/challenged) | ₹      115.13 |
| True-positive value (fraud correctly caught) | ₹  746,669.54 |
| **Net value (TP value − FP cost)** | **₹  746,554.41** |

## Defense-Only Statement

> SpikeGate is strictly defense-only. The agent action space is limited to
> `{auto_block, soft_challenge, flag_for_review, allow}`. No offensive,
> retaliatory, or data-exfiltration capabilities are present anywhere in the codebase.
> All decisions are explainable (SHAP feature attribution) and gated (deterministic
> bounds_gate overrides before any action fires). Every decision is append-logged
> to `audit_log.db` before it takes effect.