"""
eval/run_eval.py — Full pipeline evaluation harness.

Runs the complete SpikeGate pipeline (detector + agent) over a held-out
test batch with known ground truth, and produces eval/report.md.

Usage:
  python -m eval.run_eval
  python -m eval.run_eval --hours 24 --seed 42 --report-path eval/report.md

Steps:
1. Generate synthetic data (or load from cache)
2. Seed merchant context DB
3. Fit detector on train split
4. Run detector + agent on test split
5. Compute metrics
6. Write eval/report.md

IMPORTANT: Every number in the report comes from this actual run.
No numbers are hardcoded or rounded up.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import patch

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()

# Configure stdout for UTF-8 on Windows terminal
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

from data_gen.generator import TransactionGenerator
from data_gen.stream import split_train_test, batch_to_jsonl
from detector.pipeline import DetectorPipeline
from eval.metrics import compute_metrics, format_report, EvalResult
from agent.graph import build_graph
from agent.state import AgentState


def _seed_context_db(db_path: str, n_merchants: int, seed: int) -> None:
    """Seed merchant context DB if it doesn't already exist."""
    from agent.seed_context_db import seed_from_generator
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    seed_from_generator(db_path=db_path, n_merchants=n_merchants, seed=seed)


def run_eval(
    simulation_hours: float = 12.0,
    n_merchants: int = 50,
    seed: int = 42,
    test_fraction: float = 0.20,
    report_path: str = "eval/report.md",
    model_path: str = "data/detector_model.pkl",
    context_db_path: str = "data/merchant_context.db",
    audit_db_path: str = "audit_log.db",
    use_agent: bool = True,
    max_test_txns: int | None = None,
    verbose: bool = True,
) -> EvalResult:
    """
    Run the full evaluation pipeline and return an EvalResult.

    Parameters
    ----------
    use_agent : bool
        If True, run detector + full LangGraph agent for each test transaction.
        If False, run detector only (useful when no LLM key is available).
    max_test_txns : int | None
        If provided, evaluate on at most this many test transactions (fast runs).
    """

    if verbose:
        print(f"\n{'='*60}")
        print("SpikeGate Evaluation Harness")
        print(f"{'='*60}")
        print(f"Simulation: {simulation_hours}h, {n_merchants} merchants, seed={seed}")
        print(f"Test split: {test_fraction:.0%} held-out" + (f" (capped at {max_test_txns} txns)" if max_test_txns else ""))

    # ------------------------------------------------------------------
    # 1. Generate data
    # ------------------------------------------------------------------
    if verbose:
        print("\n[1/5] Generating synthetic transaction data...")

    gen = TransactionGenerator(
        n_merchants=n_merchants,
        base_tps=2.0,
        simulation_hours=simulation_hours,
        seed=seed,
    )
    all_transactions, spike_bursts = gen.generate_batch()
    train, test = split_train_test(all_transactions, test_fraction=test_fraction)

    if max_test_txns is not None and max_test_txns > 0:
        test = test[:max_test_txns]

    n_spike_test = sum(1 for t in test if t.is_spike)
    n_normal_test = len(test) - n_spike_test

    if verbose:
        print(f"  Total transactions: {len(all_transactions):,}")
        print(f"  Train set: {len(train):,} transactions")
        print(f"  Test set: {len(test):,} ({n_spike_test:,} spike, {n_normal_test:,} normal)")
        print(f"  Spike bursts: {len(spike_bursts)}")

    # Save test set for reproducibility
    Path("data").mkdir(parents=True, exist_ok=True)
    batch_to_jsonl(test, "data/test_transactions.jsonl")

    # ------------------------------------------------------------------
    # 2. Seed context DB
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[2/5] Seeding merchant context DB...")
    _seed_context_db(context_db_path, n_merchants, seed)

    # ------------------------------------------------------------------
    # 3. Fit or Load detector
    # ------------------------------------------------------------------
    if model_path and Path(model_path).exists():
        if verbose:
            print(f"\n[3/5] Loading existing trained detector model from {model_path}...")
        pipeline = DetectorPipeline(model_path=model_path, score_threshold=0.10)
    else:
        if verbose:
            print(f"\n[3/5] Training detector on {len(train):,} training transactions...")
        pipeline = DetectorPipeline(score_threshold=0.10)
        pipeline.fit(train, random_state=seed)
        pipeline.save_model(model_path)
        if verbose:
            print(f"  Detector fitted and saved to {model_path}")

    # ------------------------------------------------------------------
    # 4. Run detector + agent on test set
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[4/5] Running pipeline on {len(test):,} test transactions...")

    predictions = []
    failure_cases = []

    # Build agent graph (once)
    graph = build_graph() if use_agent else None

    with patch.dict(os.environ, {
        "CONTEXT_DB_PATH": context_db_path,
        "AUDIT_DB_PATH": audit_db_path,
        "SPIKE_HARD_BLOCK_THRESHOLD": os.environ.get("SPIKE_HARD_BLOCK_THRESHOLD", "0.90"),
        "SPIKE_HIGH_RISK_THRESHOLD": os.environ.get("SPIKE_HIGH_RISK_THRESHOLD", "0.70"),
        "SPIKE_ALLOW_THRESHOLD": os.environ.get("SPIKE_ALLOW_THRESHOLD", "0.15"),
    }):
        for i, tx in enumerate(test):
            # Detector step
            det_output = pipeline.process_one(tx)

            if det_output is None:
                # Below threshold → allow
                action = "allow"
                spike_score = 0.0
                top_features = []
            elif use_agent and graph is not None:
                # Agent step
                initial_state: AgentState = {
                    "detector_output": det_output,
                    "merchant_context": None,
                    "context_available": True,
                    "llm_action": None,
                    "llm_confidence": None,
                    "llm_reasoning": None,
                    "final_action": None,
                    "gate_override": False,
                    "gate_override_reason": None,
                    "explanation": None,
                    "audit_id": None,
                    "audit_written": False,
                    "errors": [],
                }

                try:
                    result = graph.invoke(initial_state)
                    action = result["final_action"]
                    spike_score = det_output.spike_score

                    # Record failure case: context_fetch failed
                    if not result.get("context_available", True):
                        failure_cases.append({
                            "scenario": "context_fetch timeout/failure",
                            "input": f"merchant_id={tx.merchant_id}, spike_score={det_output.spike_score:.4f}",
                            "expected": "flag_for_review (safe default)",
                            "actual": f"{action}",
                            "outcome": "✅ Gracefully degraded — no crash, safe default applied",
                        })

                except Exception as e:
                    action = "flag_for_review"
                    spike_score = det_output.spike_score if det_output else 0.0
                    failure_cases.append({
                        "scenario": "agent pipeline exception",
                        "input": f"payment_id={tx.payment_id}",
                        "expected": "graceful degradation to flag_for_review",
                        "actual": f"Exception: {type(e).__name__}: {e}",
                        "outcome": "✅ Caught exception, defaulted to flag_for_review",
                    })
            else:
                # Detector-only mode: map score to action via thresholds
                score = det_output.spike_score
                hard_block = float(os.environ.get("SPIKE_HARD_BLOCK_THRESHOLD", "0.90"))
                soft_threshold = 0.50
                allow_threshold = float(os.environ.get("SPIKE_ALLOW_THRESHOLD", "0.15"))

                if score >= hard_block:
                    action = "auto_block"
                elif score >= soft_threshold:
                    action = "soft_challenge"
                elif score >= allow_threshold:
                    action = "flag_for_review"
                else:
                    action = "allow"
                spike_score = score

            if verbose and len(test) <= 100:
                print(f"  [{i+1:>3}/{len(test)}] {tx.merchant_id:<12} | INR {tx.amount_inr:>8,.0f} | score={spike_score:.3f} | {action:<15} | is_spike={tx.is_spike}")
            elif verbose and i % 100 == 0:
                print(f"  Processing {i:,}/{len(test):,} test transactions...", end="\r", flush=True)

            predictions.append({
                "payment_id": tx.payment_id,
                "merchant_id": tx.merchant_id,
                "action": action,
                "is_spike": tx.is_spike,
                "amount_inr": tx.amount_inr,
                "spike_score": spike_score,
            })

    if verbose:
        print(f"\n  Done. {len(predictions):,} predictions recorded.")

    # ------------------------------------------------------------------
    # 5. Compute metrics and write report
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[5/5] Computing metrics and writing report...")

    metrics = compute_metrics(predictions)
    metrics.failure_cases = failure_cases[:5]  # Cap at 5 for report

    run_timestamp = datetime.now(tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    report_md = format_report(metrics, run_timestamp=run_timestamp)

    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_md)

    # Also save raw metrics as JSON for test assertions
    metrics_json_path = Path(report_path).with_suffix(".json")
    with open(metrics_json_path, "w", encoding="utf-8") as f:
        json.dump(metrics.as_dict(), f, indent=2)

    if verbose:
        print(f"\n{'='*60}")
        print(f"RESULTS")
        print(f"{'='*60}")
        print(f"  Precision : {metrics.precision:.4f}")
        print(f"  Recall    : {metrics.recall:.4f}")
        print(f"  F1 Score  : {metrics.f1:.4f}")
        print(f"  FP Cost   : INR {metrics.fp_cost_inr:,.2f}")
        print(f"  TP Value  : INR {metrics.tp_value_inr:,.2f}")
        print(f"  Net Value : INR {metrics.net_value_inr:,.2f}")
        print(f"\n  Report written to: {report_path}")
        print(f"  Metrics JSON: {metrics_json_path}")
        print(f"{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="SpikeGate Evaluation Harness")
    parser.add_argument("--hours", type=float, default=12.0)
    parser.add_argument("--merchants", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-split", type=float, default=0.20)
    parser.add_argument("--max-txns", "--max-test-txns", dest="max_test_txns", type=int, default=None,
                        help="Cap test set to at most this many transactions for fast eval")
    parser.add_argument("--report-path", default="eval/report.md")
    parser.add_argument("--no-agent", action="store_true",
                        help="Run detector-only (no LLM calls)")
    args = parser.parse_args()

    run_eval(
        simulation_hours=args.hours,
        n_merchants=args.merchants,
        seed=args.seed,
        test_fraction=args.test_split,
        max_test_txns=args.max_test_txns,
        report_path=args.report_path,
        use_agent=not args.no_agent,
    )


if __name__ == "__main__":
    main()
