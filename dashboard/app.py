"""
dashboard/app.py — SpikeGate Live Demo Dashboard (Streamlit).

Layout:
- Left column   : Live transaction stream ticker
- Center column : Spike score gauge per entity (color-coded green/yellow/red)
- Right column  : Agent decision + explanation + recent audit trail

Run with:
  streamlit run dashboard/app.py --server.port 8501
"""
import os
import sys
from pathlib import Path
import sqlite3
import time
from datetime import datetime, timezone

# Ensure project root is on sys.path for Streamlit Cloud and subfolder execution
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# Sync Streamlit Cloud secrets to os.environ if present
try:
    if hasattr(st, "secrets"):
        for key, value in st.secrets.items():
            if isinstance(value, str) and key not in os.environ:
                os.environ[key] = value
except Exception:
    pass

# ─────────────────────────────────────────────────────────────
# Page config (must be first Streamlit call)
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="SpikeGate — Fraud Spike Detector",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────
# CSS styling
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Dark card */
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 10px;
        border: 1px solid #2d3561;
    }

    /* Score badge */
    .score-high   { color: #ff4e4e; font-weight: 700; font-size: 1.4em; }
    .score-medium { color: #ffa64d; font-weight: 700; font-size: 1.4em; }
    .score-low    { color: #4dff91; font-weight: 700; font-size: 1.4em; }

    /* Action badge */
    .action-auto_block      { background: #ff4e4e; color: white; }
    .action-soft_challenge  { background: #ffa64d; color: black; }
    .action-flag_for_review { background: #4d9fff; color: white; }
    .action-allow           { background: #4dff91; color: black; }

    .action-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9em;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Transaction row */
    .tx-row {
        font-family: 'Courier New', monospace;
        font-size: 0.78em;
        padding: 4px 0;
        border-bottom: 1px solid #2d3561;
    }

    .spike-tx { color: #ff4e4e; }
    .normal-tx { color: #aaa; }

    /* Header */
    .header-title {
        font-size: 2.2em;
        font-weight: 700;
        background: linear-gradient(90deg, #4d9fff, #a64dff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Lazy imports (after page_config to avoid blocking startup)
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_pipeline_and_adapter():
    """Initialize detector pipeline and stream adapter (cached across reruns)."""
    from dashboard.stream_adapter import StreamAdapter
    from detector.pipeline import DetectorPipeline
    from data_gen.generator import TransactionGenerator
    from data_gen.stream import split_train_test
    from agent.seed_context_db import seed_from_generator

    SEED = int(os.environ.get("DATAGEN_SEED", "42"))
    N_MERCHANTS = int(os.environ.get("DATAGEN_N_MERCHANTS", "30"))
    CONTEXT_DB = os.environ.get("CONTEXT_DB_PATH", "data/merchant_context.db")

    # Seed context DB
    Path("data").mkdir(parents=True, exist_ok=True)
    if not Path(CONTEXT_DB).exists():
        seed_from_generator(db_path=CONTEXT_DB, n_merchants=N_MERCHANTS, seed=SEED)

    # Train detector on a short warm-up run
    with st.spinner("Training detector model (one-time)..."):
        gen = TransactionGenerator(n_merchants=N_MERCHANTS, seed=SEED, simulation_hours=3.0)
        all_txns, _ = gen.generate_batch()
        train, _ = split_train_test(all_txns, test_fraction=0.20)

        pipeline = DetectorPipeline(score_threshold=0.10)
        pipeline.fit(train)

    # Create live stream adapter
    adapter = StreamAdapter(
        n_merchants=N_MERCHANTS,
        seed=SEED + 1,
        simulation_hours=2.0,
        speed=float(os.environ.get("STREAM_REPLAY_SPEED", "10.0")),
    )
    adapter.start()

    return pipeline, adapter


@st.cache_resource
def get_agent_graph():
    """Build agent graph (cached)."""
    from agent.graph import build_graph
    return build_graph()


# ─────────────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────────────

if "tx_log" not in st.session_state:
    st.session_state.tx_log = []          # list of {tx, spike_score, action}
if "alert_log" not in st.session_state:
    st.session_state.alert_log = []       # high-score alerts only
if "total_processed" not in st.session_state:
    st.session_state.total_processed = 0
if "total_alerts" not in st.session_state:
    st.session_state.total_alerts = 0


# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────

st.markdown('<div class="header-title">⚡ SpikeGate</div>', unsafe_allow_html=True)
st.markdown("**Real-Time Fraud Spike Detector** — Razorpay AI Risk Manager")
st.divider()

# ─────────────────────────────────────────────────────────────
# Main layout: 3 columns
# ─────────────────────────────────────────────────────────────

col_stream, col_gauge, col_decision = st.columns([1.2, 1.5, 1.5])

with col_stream:
    st.subheader("📡 Live Stream")
    stream_placeholder = st.empty()

with col_gauge:
    st.subheader("🎯 Spike Scores")
    gauge_placeholder = st.empty()

with col_decision:
    st.subheader("🛡️ Agent Decisions")
    decision_placeholder = st.empty()

# Stats bar
st.divider()
stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
stat_placeholder1 = stat_col1.empty()
stat_placeholder2 = stat_col2.empty()
stat_placeholder3 = stat_col3.empty()
stat_placeholder4 = stat_col4.empty()

# ─────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────

pipeline, adapter = get_pipeline_and_adapter()

AUDIT_DB = os.environ.get("AUDIT_DB_PATH", "audit_log.db")

# Counters for action distribution
action_counts = {"auto_block": 0, "soft_challenge": 0, "flag_for_review": 0, "allow": 0}

BATCH_SIZE = 5  # Process N transactions per UI refresh

while True:
    new_txs = []
    for _ in range(BATCH_SIZE):
        tx = adapter.poll(timeout=0.02)
        if tx:
            new_txs.append(tx)

    for tx in new_txs:
        st.session_state.total_processed += 1

        # Detector
        det_output = pipeline.process_one(tx)

        if det_output is None:
            spike_score = 0.0
            action = "allow"
            top_features = []
            explanation = None
        else:
            spike_score = det_output.spike_score

            # Bounds gate (deterministic, no LLM)
            from agent.nodes.bounds_gate import bounds_gate
            from agent.state import AgentState

            dummy_state: AgentState = {
                "detector_output": det_output,
                "merchant_context": None,
                "context_available": False,
                "llm_action": "flag_for_review",
                "llm_confidence": 0.5,
                "llm_reasoning": "Dashboard mode — LLM bypassed",
                "final_action": None,
                "gate_override": False,
                "gate_override_reason": None,
                "explanation": None,
                "audit_id": None,
                "audit_written": False,
                "errors": [],
            }

            gate_result = bounds_gate(dummy_state)
            action = gate_result["final_action"]
            top_features = det_output.top_features

            # Generate explanation
            from agent.nodes.explainer import explainer
            gate_result = explainer(gate_result)
            explanation = gate_result.get("explanation", "")

        action_counts[action] = action_counts.get(action, 0) + 1

        # Log
        entry = {
            "ts": tx.timestamp.strftime("%H:%M:%S"),
            "merchant": tx.merchant_id[:10],
            "amount": tx.amount_inr,
            "method": tx.payment_method,
            "spike_score": spike_score,
            "action": action,
            "is_spike": tx.is_spike,
            "top_features": top_features,
            "explanation": explanation,
        }
        st.session_state.tx_log.insert(0, entry)
        st.session_state.tx_log = st.session_state.tx_log[:50]  # Keep last 50

        if spike_score >= 0.40:
            st.session_state.alert_log.insert(0, entry)
            st.session_state.alert_log = st.session_state.alert_log[:20]
            st.session_state.total_alerts += 1

    # ── Render stream ticker ──
    with stream_placeholder.container():
        for entry in st.session_state.tx_log[:20]:
            color = "🔴" if entry["is_spike"] else "⚪"
            score_str = f"{entry['spike_score']:.2f}"
            st.markdown(
                f"`{entry['ts']}` {color} **{entry['merchant']}**  "
                f"₹{entry['amount']:,.0f} | score={score_str}",
            )

    # ── Render gauge ──
    with gauge_placeholder.container():
        recent_alerts = st.session_state.alert_log[:8]
        if not recent_alerts:
            st.info("No significant spikes detected yet...")
        else:
            for entry in recent_alerts:
                score = entry["spike_score"]
                if score >= 0.80:
                    color_class = "score-high"
                    bar_color = "#ff4e4e"
                elif score >= 0.50:
                    color_class = "score-medium"
                    bar_color = "#ffa64d"
                else:
                    color_class = "score-low"
                    bar_color = "#4dff91"

                st.markdown(
                    f'<div class="metric-card">'
                    f'<span style="color:#aaa;font-size:0.8em;">{entry["ts"]} | {entry["merchant"]}</span><br>'
                    f'<span class="{color_class}">{score:.3f}</span>'
                    f'<progress value="{score}" max="1.0" style="width:100%;accent-color:{bar_color}"></progress>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Render decisions ──
    with decision_placeholder.container():
        recent_decisions = st.session_state.alert_log[:5]
        if not recent_decisions:
            st.info("Waiting for alerts...")
        else:
            for entry in recent_decisions:
                action = entry["action"]
                action_colors = {
                    "auto_block": "#ff4e4e",
                    "soft_challenge": "#ffa64d",
                    "flag_for_review": "#4d9fff",
                    "allow": "#4dff91",
                }
                color = action_colors.get(action, "#aaa")

                feat_lines = ""
                for f in entry["top_features"][:3]:
                    feat_lines += f"• `{f['name']}`={f['value']:.2f} (SHAP:{f['contribution']:+.3f})<br>"

                st.markdown(
                    f'<div class="metric-card">'
                    f'<span style="color:#aaa;font-size:0.8em;">{entry["ts"]} | {entry["merchant"]}</span><br>'
                    f'<span style="color:{color};font-weight:700;font-size:1.1em;">{action.upper()}</span><br>'
                    f'<span style="color:#ccc;font-size:0.8em;">{feat_lines}</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )

    # ── Stats bar ──
    stat_placeholder1.metric("Transactions Processed", f"{st.session_state.total_processed:,}")
    stat_placeholder2.metric("Alerts Fired (score≥0.40)", f"{st.session_state.total_alerts:,}")
    stat_placeholder3.metric("Auto-Blocked", f"{action_counts.get('auto_block', 0):,}")
    stat_placeholder4.metric("Soft Challenged", f"{action_counts.get('soft_challenge', 0):,}")

    time.sleep(0.3)  # 3 FPS refresh
    st.rerun()


def main():
    """Entry point for pyproject.toml script."""
    import subprocess
    import sys
    subprocess.run([sys.executable, "-m", "streamlit", "run", __file__])
