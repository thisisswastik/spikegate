"""
Phase 4 tests — Evaluation harness (eval/).

Test gate requirements:
- eval script runs end-to-end without error
- report.md is created and contains all required sections
- All metric values are finite numbers (not NaN/Inf)
- Failure case is demonstrated in the report
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from eval.metrics import compute_metrics, format_report, EvalResult


# ---------------------------------------------------------------------------
# Unit tests for metrics computation
# ---------------------------------------------------------------------------

class TestMetricsComputation:
    def _make_predictions(self, n_normal=100, n_spike=20) -> list[dict]:
        preds = []
        for i in range(n_spike):
            preds.append({
                "payment_id": f"pay_spike_{i}",
                "action": "auto_block",
                "is_spike": True,
                "amount_inr": 5000.0,
                "spike_score": 0.95,
            })
        for i in range(n_normal):
            preds.append({
                "payment_id": f"pay_normal_{i}",
                "action": "allow",
                "is_spike": False,
                "amount_inr": 1000.0,
                "spike_score": 0.05,
            })
        return preds

    def test_perfect_classifier(self):
        """Perfect classifier: precision=recall=F1=1.0, FP cost=0."""
        preds = self._make_predictions(n_normal=100, n_spike=20)
        result = compute_metrics(preds)
        assert result.precision == 1.0
        assert result.recall == 1.0
        assert result.f1 == 1.0
        assert result.fp_cost_inr == 0.0
        assert result.tp_value_inr == 20 * 5000.0

    def test_all_false_positives(self):
        """All predictions are auto_block on normal transactions → precision=0."""
        preds = []
        for i in range(50):
            preds.append({
                "payment_id": f"pay_{i}",
                "action": "auto_block",
                "is_spike": False,
                "amount_inr": 2000.0,
                "spike_score": 0.95,
            })
        result = compute_metrics(preds)
        assert result.precision == 0.0
        assert result.recall == 0.0
        assert result.fp_cost_inr == 50 * 2000.0

    def test_soft_challenge_fp_friction_cost(self):
        """soft_challenge FP cost is 10% of transaction amount (friction model)."""
        preds = [{
            "payment_id": "pay_001",
            "action": "soft_challenge",
            "is_spike": False,
            "amount_inr": 10000.0,
            "spike_score": 0.60,
        }]
        result = compute_metrics(preds)
        # 10% friction of 10000 = 1000
        assert abs(result.fp_cost_inr - 1000.0) < 0.01

    def test_metric_values_finite(self):
        """All metric values must be finite floats."""
        preds = self._make_predictions(n_normal=80, n_spike=20)
        result = compute_metrics(preds)
        d = result.as_dict()

        def check_finite(obj, path=""):
            if isinstance(obj, float):
                assert not (obj != obj), f"NaN at {path}"
                assert obj != float('inf') and obj != float('-inf'), f"Inf at {path}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_finite(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_finite(v, f"{path}[{i}]")

        check_finite(d)

    def test_empty_predictions(self):
        """Empty predictions must not crash and return zero metrics."""
        result = compute_metrics([])
        assert result.n_total == 0
        assert result.precision == 0.0
        assert result.recall == 0.0

    def test_action_counts_sum_to_total(self):
        """Sum of per-action counts must equal total transaction count."""
        preds = self._make_predictions(n_normal=80, n_spike=20)
        # Mix in some other actions
        for p in preds[:30]:
            p["action"] = "soft_challenge"
        for p in preds[30:50]:
            p["action"] = "flag_for_review"
        result = compute_metrics(preds)
        assert sum(result.action_counts.values()) == result.n_total

    def test_net_value_equals_tp_minus_fp(self):
        """Net value must equal TP value − FP cost."""
        preds = self._make_predictions(n_normal=50, n_spike=20)
        # Add some false positives
        for p in preds[:10]:
            p["is_spike"] = False  # Make these FPs
        result = compute_metrics(preds)
        assert abs(result.net_value_inr - (result.tp_value_inr - result.fp_cost_inr)) < 0.01


# ---------------------------------------------------------------------------
# Report format tests
# ---------------------------------------------------------------------------

class TestReportFormat:
    def _make_result(self) -> EvalResult:
        preds = [
            {"payment_id": "p1", "action": "auto_block", "is_spike": True, "amount_inr": 5000, "spike_score": 0.95},
            {"payment_id": "p2", "action": "allow", "is_spike": False, "amount_inr": 1000, "spike_score": 0.05},
            {"payment_id": "p3", "action": "soft_challenge", "is_spike": True, "amount_inr": 3000, "spike_score": 0.70},
            {"payment_id": "p4", "action": "flag_for_review", "is_spike": False, "amount_inr": 2000, "spike_score": 0.30},
        ]
        return compute_metrics(preds)

    def test_report_has_required_sections(self):
        """The Markdown report must contain all required sections."""
        result = self._make_result()
        report = format_report(result)

        required_sections = [
            "# SpikeGate Evaluation Report",
            "## Dataset",
            "## Action Distribution",
            "## Binary Classification Metrics",
            "## Per-Action Metrics",
            "## False-Positive Cost Analysis (INR)",
            "## Defense-Only Statement",
        ]
        for section in required_sections:
            assert section in report, f"Missing section: '{section}'"

    def test_report_contains_rupee_values(self):
        """The report must contain rupee cost/value figures."""
        result = self._make_result()
        report = format_report(result)
        # Should have rupee symbol or INR mention
        assert "₹" in report or "INR" in report

    def test_report_contains_precision_recall(self):
        """The report must contain precision and recall values."""
        result = self._make_result()
        report = format_report(result)
        assert "Precision" in report
        assert "Recall" in report
        assert "F1" in report

    def test_report_contains_defense_statement(self):
        """The report must contain the defense-only statement."""
        result = self._make_result()
        report = format_report(result)
        assert "defense-only" in report.lower() or "defense only" in report.lower()


# ---------------------------------------------------------------------------
# End-to-end eval run (detector-only mode to avoid LLM dependency)
# ---------------------------------------------------------------------------

class TestEvalEndToEnd:
    def test_eval_runs_without_error(self, tmp_path):
        """eval/run_eval.py must run end-to-end without raising exceptions."""
        from eval.run_eval import run_eval

        report_path = str(tmp_path / "report.md")
        context_db = str(tmp_path / "merchant_context.db")
        audit_db = str(tmp_path / "audit_log.db")
        model_path = str(tmp_path / "model.pkl")

        result = run_eval(
            simulation_hours=2.0,     # short sim for test speed
            n_merchants=10,
            seed=42,
            test_fraction=0.20,
            report_path=report_path,
            model_path=model_path,
            context_db_path=context_db,
            audit_db_path=audit_db,
            use_agent=False,          # detector-only (no LLM key needed)
            verbose=False,
        )

        assert result is not None
        assert result.n_total > 0

    def test_report_md_created(self, tmp_path):
        """eval/run_eval must create the report.md file."""
        from eval.run_eval import run_eval

        report_path = str(tmp_path / "report.md")
        run_eval(
            simulation_hours=1.0,
            n_merchants=5,
            seed=42,
            test_fraction=0.20,
            report_path=report_path,
            model_path=str(tmp_path / "model.pkl"),
            context_db_path=str(tmp_path / "ctx.db"),
            audit_db_path=str(tmp_path / "audit.db"),
            use_agent=False,
            verbose=False,
        )

        assert Path(report_path).exists(), "report.md was not created"

    def test_report_json_has_all_fields(self, tmp_path):
        """The report.json must have all required metric fields."""
        from eval.run_eval import run_eval

        report_path = str(tmp_path / "report.md")
        run_eval(
            simulation_hours=1.0,
            n_merchants=5,
            seed=42,
            test_fraction=0.20,
            report_path=report_path,
            model_path=str(tmp_path / "model.pkl"),
            context_db_path=str(tmp_path / "ctx.db"),
            audit_db_path=str(tmp_path / "audit.db"),
            use_agent=False,
            verbose=False,
        )

        json_path = Path(report_path).with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)

        required_keys = ["n_total", "n_spike", "n_normal", "binary_metrics", "rupee_costs"]
        for key in required_keys:
            assert key in data, f"Missing key '{key}' in report.json"

        binary = data["binary_metrics"]
        for metric in ["precision", "recall", "f1"]:
            val = binary[metric]
            assert isinstance(val, float)
            assert 0.0 <= val <= 1.0, f"Metric '{metric}' out of [0,1]: {val}"

    def test_metrics_all_finite(self, tmp_path):
        """All metric values in report.json must be finite numbers."""
        from eval.run_eval import run_eval
        import math

        report_path = str(tmp_path / "report.md")
        run_eval(
            simulation_hours=1.0,
            n_merchants=5,
            seed=42,
            test_fraction=0.20,
            report_path=report_path,
            model_path=str(tmp_path / "model.pkl"),
            context_db_path=str(tmp_path / "ctx.db"),
            audit_db_path=str(tmp_path / "audit.db"),
            use_agent=False,
            verbose=False,
        )

        json_path = Path(report_path).with_suffix(".json")
        with open(json_path) as f:
            data = json.load(f)

        def check_finite(obj, path=""):
            if isinstance(obj, float):
                assert math.isfinite(obj), f"Non-finite value at {path}: {obj}"
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    check_finite(v, f"{path}.{k}")
            elif isinstance(obj, list):
                for i, v in enumerate(obj):
                    check_finite(v, f"{path}[{i}]")

        check_finite(data)
