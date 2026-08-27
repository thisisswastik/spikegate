# SpikeGate ⚡

> **Razorpay Hackathon · Track 02: AI Risk Manager**
> Real-time fraud-spike detection and bounded agentic response system

---

## Problem Statement

Payment fraud doesn't spike uniformly — it arrives in bursts: a compromised card BIN used across 40 merchants in 5 minutes, a device fingerprint firing transactions across a dozen cities in seconds, or a merchant under a fraudulent account batch-triggering refunds. Standard per-transaction ML models miss these *velocity anomalies* because they score in isolation.

SpikeGate detects these spike patterns using **rolling-window entity features** and routes each alert through a **LangGraph agent** that takes a strictly bounded, explainable, and audit-logged action.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────┐
  │                    TRANSACTION STREAM                        │
  │          (Razorpay-style synthetic JSONL / live queue)       │
  └───────────────────────┬─────────────────────────────────────┘
                          │
                          ▼
  ┌─────────────────────────────────────────────────────────────┐
  │                   DETECTOR PIPELINE                          │
  │  ┌──────────────┐  ┌─────────────────┐  ┌───────────────┐  │
  │  │ Rolling-     │  │ Feature         │  │ SpikeDetector │  │
  │  │ Window       │→ │ Extractor       │→ │ XGB + IsoFst  │  │
  │  │ Engine       │  │ (54 features)   │  │ + SHAP        │  │
  │  │ 1m/5m/1h     │  │ per entity type │  │               │  │
  │  └──────────────┘  └─────────────────┘  └───────┬───────┘  │
  │                                                  │          │
  │                              spike_score + top_features     │
  └──────────────────────────────────────────────────┼──────────┘
                                                     │
                          ┌──────────────────────────▼───────────┐
                          │         LANGGRAPH AGENT               │
                          │                                       │
                          │  context_fetch                        │
                          │    ↓ (merchant_context.db SQLite)     │
                          │  policy_reasoner                      │
                          │    ↓ (Gemini API, JSON-mode, enum)    │
                          │  bounds_gate  ← DETERMINISTIC GUARDRAIL│
                          │    ↓ (overrides LLM at hard thresholds)│
                          │  explainer                            │
                          │    ↓ (SHAP-feature-grounded text)     │
                          │  audit_writer                         │
                          │    ↓ (append-only SQLite log)         │
                          └──────────────────────────────────────┘
                                           │
                    ┌──────────────────────▼───────────┐
                    │  ACTION: one of:                  │
                    │    auto_block                     │
                    │    soft_challenge                 │
                    │    flag_for_review                │
                    │    allow                          │
                    └──────────────────────────────────┘
```

---

## Defense-Only Statement

> **SpikeGate is strictly defense-only.**
>
> The agent's action space is limited to `{auto_block, soft_challenge, flag_for_review, allow}`.
> No offensive, retaliatory, or data-exfiltration capabilities exist anywhere in this codebase.
> All decisions are:
> - **Explainable** — every action references the actual SHAP feature values that drove it
> - **Bounded** — a deterministic `bounds_gate` node overrides the LLM at hard thresholds
> - **Gated** — every decision is append-logged to `audit_log.db` before/as it fires

---

## Project Structure

```
spikegate/
├── data_gen/           # Synthetic Razorpay-style stream + spike injection
│   ├── schema.py       # Pydantic models (Transaction, SpikeBurst, DetectorOutput)
│   ├── generator.py    # TransactionGenerator with Poisson sampling + spike injection
│   └── stream.py       # JSONL serialization, async replay, train/test split
│
├── detector/           # Rolling-window feature engine + ensemble model
│   ├── windows.py      # Epoch-bucket rolling windows (1m/5m/1h per entity)
│   ├── features.py     # FeatureExtractor — 54 features across 9 groups
│   ├── model.py        # SpikeDetector: XGBoost + IsolationForest + SHAP
│   └── pipeline.py     # DetectorPipeline: fit + process_one + process_batch
│
├── agent/              # LangGraph agent
│   ├── state.py        # AgentState TypedDict
│   ├── graph.py        # LangGraph graph builder
│   ├── seed_context_db.py  # Seeds merchant_context.db
│   └── nodes/
│       ├── context_fetch.py    # SQLite merchant context lookup
│       ├── policy_reasoner.py  # Gemini LLM → bounded enum action
│       ├── bounds_gate.py      # Deterministic hard-override guardrail
│       ├── explainer.py        # SHAP-grounded human-readable explanation
│       └── audit_writer.py     # Append-only audit_log.db writer
│
├── eval/               # Evaluation harness
│   ├── metrics.py      # Precision/recall/F1 + rupee cost/value
│   ├── run_eval.py     # Full pipeline eval → eval/report.md + report.json
│   └── report.md       # ← Generated by `python -m eval.run_eval` (checked in after run)
│
├── dashboard/          # Live demo UI
│   ├── app.py          # Streamlit dashboard
│   └── stream_adapter.py  # Thread-safe stream queue adapter
│
├── tests/              # Mirrored by phase
│   ├── test_phase0_imports.py
│   ├── test_phase1_datagen.py
│   ├── test_phase2_detector.py
│   ├── test_phase3_agent.py
│   └── test_phase4_eval.py
│
├── audit_log.db        # Append-only decision log (generated at runtime)
├── .env.example        # All required environment variables documented
├── pyproject.toml
└── README.md
```

---

## Features (54 total, all interpretable)

| Group | Features | Count |
|-------|----------|-------|
| Velocity | `tx_count_1m/5m/1h` | 3 |
| Amount stats | `amt_sum/mean/std/max` × 3 windows | 12 |
| Counterparty diversity | unique merchants/cards/devices/IPs × 3 windows | 12 |
| Failure rate | `fail_count/fail_rate` × 3 windows | 6 |
| Payment method mix | fraction UPI/card/netbanking/wallet/EMI × 3 windows | 15 |
| Geo dispersion | unique /24 subnets × 3 windows | 3 |
| Temporal | hour_of_day, day_of_week, is_weekend, seconds_since_last_tx | 4 |
| Merchant tier | risk tier (ordinal 0/1/2) | 1 |
| Recency ratios | velocity_ratio_1m_1h, 5m_1h; amt_ratio_1m_1h, 5m_1h | 4 |
| **Total** | | **60** |

---

## How to Run

### 1. Setup

```bash
git clone <repo>
cd spikegate
uv venv
.venv/Scripts/activate        # Windows
# or: source .venv/bin/activate  # Linux/Mac

uv pip install -e ".[dev]"

# Copy and fill in your API key
cp .env.example .env
# Edit .env: set GEMINI_API_KEY=your_key_here
```

### 2. Run Tests (Phase by Phase)

```bash
pytest tests/test_phase0_imports.py -v   # Phase 0: scaffolding
pytest tests/test_phase1_datagen.py -v   # Phase 1: data generator
pytest tests/test_phase2_detector.py -v  # Phase 2: detector (prints real metrics)
pytest tests/test_phase3_agent.py -v     # Phase 3: agent (mocked LLM)
pytest tests/test_phase4_eval.py -v      # Phase 4: eval harness
pytest tests/ -v                         # All phases
```

### 3. Run Detector-Only Evaluation (no LLM key needed)

```bash
python -m eval.run_eval --no-agent --hours 12 --merchants 50
# Output: eval/report.md, eval/report.json
```

### 4. Run Full Evaluation (requires GEMINI_API_KEY in .env)

```bash
python -m eval.run_eval --hours 12 --merchants 50
# Output: eval/report.md with honest precision/recall + rupee cost table
```

### 5. Launch Dashboard

```bash
streamlit run dashboard/app.py --server.port 8501
# Open: http://localhost:8501
```

---

## Honest Metrics (Held-Out Test Set: 18,107 Transactions)

> Every number below is from an actual evaluation on held-out test data (`simulation_hours=12`, `n_merchants=50`, `seed=42`).
> No numbers have been rounded up, simulated, or cherry-picked.

| Metric | Value | Notes |
|---|---|---|
| **Test Set Size** | **18,107** | Held-out 20% split |
| **Spike Transactions** | **860 (4.75%)** | Ground-truth spike bursts (realistic fraud prevalence) |
| **Normal Transactions** | **17,247 (95.25%)** | Poisson baseline traffic |
| **Decision Threshold** | **0.50** | Action boundary threshold |
| **Precision** | **70.58% (0.7058)** | Fraction of predicted spikes that were real |
| **Recall** | **81.74% (0.8174)** | Fraction of actual spikes caught |
| **F1 Score** | **75.75% (0.7575)** | Harmonic mean of precision & recall |
| **Overall Accuracy** | **97.51%** | True Positives (703) + True Negatives (16,954) |

### False-Positive Cost Analysis (INR)

> **FP cost model**: 100% of transaction amount for wrong blocks, 10% friction cost for soft challenges.

| Metric | Amount (₹) | Details |
|---|---|---|
| **False-Positive Cost** | **₹137,876.28** | Friction on 293 legitimate transactions challenged |
| **True-Positive Value** | **₹3,603,136.98** | Fraud prevented on 703 real spike attacks caught |
| **Net Platform Value** | **₹3,465,260.70** | **TP Value − FP Cost (25.1× ROI)** |

*See [`eval/report.md`](eval/report.md) for full per-action breakdown.*

---

## Graceful Failure Demonstration

When `context_fetch` fails (DB timeout, missing file):
1. `context_available` is set to `False`
2. `policy_reasoner` skips the LLM call and defaults to `flag_for_review`
3. `bounds_gate` still applies its hard-threshold overrides
4. `audit_writer` logs the decision with `context_available=0`
5. **No crash, no blind block** — the system degrades gracefully

---

## Environment Variables

See [`.env.example`](.env.example) for the full list. Key variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `GEMINI_API_KEY` | Google AI Studio API key | _(required for Phase 3)_ |
| `GEMINI_MODEL` | Gemini model name | `gemini-1.5-flash` |
| `SPIKE_HARD_BLOCK_THRESHOLD` | Score ≥ this → force `auto_block` | `0.90` |
| `SPIKE_HIGH_RISK_THRESHOLD` | Score ≥ this + high-risk merchant → `auto_block` | `0.70` |
| `SPIKE_ALLOW_THRESHOLD` | Score ≤ this → force `allow` | `0.15` |
| `AUDIT_DB_PATH` | Audit log SQLite path | `audit_log.db` |
| `DATAGEN_SEED` | Random seed for reproducibility | `42` |
| `STREAM_REPLAY_SPEED` | Dashboard stream speed multiplier | `10.0` |

---

## Submission Checklist

- ✅ Working detector with measured precision/recall on a held-out test set
- ✅ Auto-responder with strictly bounded action space (4-value enum only)
- ✅ Every decision explainable (SHAP feature attribution in explanation)
- ✅ Every decision gated (deterministic bounds_gate before action fires)
- ✅ Every decision audit-logged (append-only SQLite before/as it fires)
- ✅ Honest metrics including false-positive cost in rupee terms
- ✅ One failure case handled gracefully (context_fetch timeout → flag_for_review)
- ✅ Strictly defense-only — no offense-capable code anywhere in the repo
- ✅ All environment variables documented in `.env.example`, none hardcoded