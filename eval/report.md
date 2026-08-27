# SpikeGate Evaluation Report
> Generated: 2026-08-27T12:19:45Z

## Dataset

| Metric | Value |
|--------|-------|
| Total transactions | 12,636 |
| Spike transactions | 9,114 (72.1%) |
| Normal transactions | 3,522 |

## Action Distribution

| Action | Count | % of Total |
|--------|-------|-----------|
| auto_block | 0 | 0.0% |
| soft_challenge | 11,271 | 89.2% |
| flag_for_review | 1,342 | 10.6% |
| allow | 23 | 0.2% |

## Binary Classification Metrics
> (auto_block + soft_challenge = 'actioned' positive prediction)

| Metric | Value |
|--------|-------|
| True Positives | 9,014 |
| False Positives | 2,257 |
| False Negatives | 100 |
| True Negatives | 1,265 |
| **Precision** | **0.7998** |
| **Recall** | **0.9890** |
| **F1 Score** | **0.8844** |

## Per-Action Metrics

| Action | Count | Precision | Recall | F1 |
|--------|-------|-----------|--------|-----|
| auto_block | 0 | 0.0000 | 0.0000 | 0.0000 |
| soft_challenge | 11,271 | 0.7998 | 0.9890 | 0.8844 |
| flag_for_review | 1,342 | 0.9255 | 0.3526 | 0.5107 |
| allow | 23 | 1.0000 | 0.0065 | 0.0130 |

## False-Positive Cost Analysis (INR)

> FP cost = blocked legitimate transaction amount (conservative model)
> soft_challenge FP cost = 10% friction of transaction amount

| Metric | Amount (₹) |
|--------|-----------|
| False-positive cost (wrongly blocked/challenged) | ₹  947,025.35 |
| True-positive value (fraud correctly caught) | ₹30,994,706.38 |
| **Net value (TP value − FP cost)** | **₹30,047,681.03** |

## Defense-Only Statement

> SpikeGate is strictly defense-only. The agent action space is limited to
> `{auto_block, soft_challenge, flag_for_review, allow}`. No offensive,
> retaliatory, or data-exfiltration capabilities are present anywhere in the codebase.
> All decisions are explainable (SHAP feature attribution) and gated (deterministic
> bounds_gate overrides before any action fires). Every decision is append-logged
> to `audit_log.db` before it takes effect.
