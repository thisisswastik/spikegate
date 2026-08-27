# SpikeGate Evaluation Report
> Generated: 2026-08-27T15:51:30Z

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
| soft_challenge | 996 | 5.5% |
| flag_for_review | 8,771 | 48.4% |
| allow | 8,340 | 46.1% |

## Binary Classification Metrics
> (auto_block + soft_challenge = 'actioned' positive prediction)

| Metric | Value |
|--------|-------|
| True Positives | 703 |
| False Positives | 293 |
| False Negatives | 157 |
| True Negatives | 16,954 |
| **Precision** | **0.7058** |
| **Recall** | **0.8174** |
| **F1 Score** | **0.7575** |

## Per-Action Metrics

| Action | Count | Precision | Recall | F1 |
|--------|-------|-----------|--------|-----|
| auto_block | 0 | 0.0000 | 0.0000 | 0.0000 |
| soft_challenge | 996 | 0.7058 | 0.8174 | 0.7575 |
| flag_for_review | 8,771 | 0.9847 | 0.5008 | 0.6639 |
| allow | 8,340 | 0.9972 | 0.4822 | 0.6501 |

## False-Positive Cost Analysis (INR)

> FP cost = blocked legitimate transaction amount (conservative model)
> soft_challenge FP cost = 10% friction of transaction amount

| Metric | Amount (₹) |
|--------|-----------|
| False-positive cost (wrongly blocked/challenged) | ₹  137,876.28 |
| True-positive value (fraud correctly caught) | ₹3,603,136.98 |
| **Net value (TP value − FP cost)** | **₹3,465,260.70** |

## Defense-Only Statement

> SpikeGate is strictly defense-only. The agent action space is limited to
> `{auto_block, soft_challenge, flag_for_review, allow}`. No offensive,
> retaliatory, or data-exfiltration capabilities are present anywhere in the codebase.
> All decisions are explainable (SHAP feature attribution) and gated (deterministic
> bounds_gate overrides before any action fires). Every decision is append-logged
> to `audit_log.db` before it takes effect.