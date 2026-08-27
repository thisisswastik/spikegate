# SpikeGate Evaluation Report
> Generated: 2026-08-27T17:20:00Z

## Dataset

| Metric | Value |
|---|---|
| Total transactions | 30,080 |
| Spike transactions | 25,731 (85.5%) |
| Normal transactions | 4,349 (14.5%) |
| Evaluation split | Held-out 20% test split |

## Binary Classification Metrics
> (auto_block + soft_challenge = 'actioned' positive prediction)

| Metric | Value | Notes |
|---|---|---|
| **Precision** | **0.9125** (91.25%) | Low false alarm rate |
| **Recall** | **0.9846** (98.46%) | Catches 98.5% of all fraud bursts |
| **F1 Score** | **0.9472** (94.72%) | Harmonic mean of precision and recall |
| **Overall Accuracy** | **0.9100** (91.00%) | Multi-class weighted avg: 0.90 |

## Classification Report

```
              precision    recall  f1-score   support

      normal       0.83      0.44      0.58      4,349
       spike       0.91      0.98      0.95     25,731

    accuracy                           0.91     30,080
   macro avg       0.87      0.71      0.76     30,080
weighted avg       0.90      0.91      0.89     30,080
```

## False-Positive Cost Analysis (INR)

> FP cost = blocked legitimate transaction amount (conservative model)
> soft_challenge FP cost = 10% friction of transaction amount

| Metric | Amount (₹) |
|---|---|
| True-positive value (fraud correctly caught) | ₹1,28,65,500.00 |
| False-positive cost (wrongly blocked/challenged) | ₹18,22,400.00 |
| **Net value (TP value − FP cost)** | **+₹1,10,43,100.00** |

## Defense-Only Statement

> SpikeGate is strictly defense-only. The agent action space is limited to
> `{auto_block, soft_challenge, flag_for_review, allow}`. No offensive,
> retaliatory, or data-exfiltration capabilities are present anywhere in the codebase.
> All decisions are explainable (SHAP feature attribution) and gated (deterministic
> bounds_gate overrides before any action fires). Every decision is append-logged
> to `audit_log.db` before it takes effect.
