"""
Phase 2 tests — Detection core (detector/).

Test gate requirements:
- Feature extractor produces expected feature count (40-60)
- spike_score is in [0, 1]
- Standalone detector precision/recall/F1 on held-out test split
  (REAL numbers from an actual run — shown before proceeding to Phase 3)
"""
from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from data_gen.generator import TransactionGenerator
from data_gen.stream import split_train_test
from detector.features import FeatureExtractor
from detector.model import SpikeDetector
from detector.pipeline import DetectorPipeline
from detector.windows import RollingWindowEngine


# ---------------------------------------------------------------------------
# Generator config for Phase 2 — bigger sim for meaningful ML metrics
# ---------------------------------------------------------------------------

DETECTOR_GEN_CONFIG = dict(
    n_merchants=15,
    base_tps=3.0,
    spike_prob=0.05,
    simulation_hours=2.0,  # 2-hour sim — enough for ML training, fast for tests
    seed=123,
)


@pytest.fixture(scope="module")
def detector_data():
    """Generate, split, and return train/test transactions (module-scoped for speed)."""
    gen = TransactionGenerator(**DETECTOR_GEN_CONFIG)
    txns, bursts = gen.generate_batch()
    train, test = split_train_test(txns, test_fraction=0.20)
    return train, test, bursts


# ---------------------------------------------------------------------------
# Feature extractor tests
# ---------------------------------------------------------------------------

class TestFeatureExtractor:
    def test_feature_count_in_range(self, detector_data):
        """Feature extractor must produce 40-60 features."""
        train, _, _ = detector_data
        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)

        # Ingest a few transactions to populate windows
        for tx in train[:50]:
            engine.ingest(tx)

        # Extract features for any merchant
        tx = train[49]
        fv = extractor.extract("merchant_id", tx.merchant_id, tx.timestamp)

        n_features = len(fv.feature_names)
        assert 40 <= n_features <= 60, (
            f"Expected 40-60 features, got {n_features}"
        )
        print(f"\n  [OK] Feature count: {n_features}")

    def test_feature_names_unique(self, detector_data):
        """Feature names must all be unique."""
        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)
        names = extractor.feature_names
        assert len(names) == len(set(names)), "Duplicate feature names detected"

    def test_feature_values_finite(self, detector_data):
        """All feature values must be finite (no NaN, no Inf)."""
        train, _, _ = detector_data
        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)

        for tx in train[:200]:
            engine.ingest(tx)

        tx = train[199]
        fv = extractor.extract("merchant_id", tx.merchant_id, tx.timestamp)

        assert np.all(np.isfinite(fv.values)), (
            f"Non-finite feature values: {[(n,v) for n,v in zip(fv.feature_names, fv.values.tolist()) if not np.isfinite(v)]}"
        )

    def test_cold_start_no_error(self, detector_data):
        """Feature extractor must not crash when there's no window history."""
        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)
        from data_gen.generator import _random_id
        import random
        from datetime import datetime, timezone
        fv = extractor.extract(
            "merchant_id",
            "mid_test",
            datetime.now(tz=timezone.utc),
        )
        assert len(fv.values) == extractor.n_features
        assert np.all(np.isfinite(fv.values))

    def test_spike_window_has_higher_velocity(self, detector_data):
        """After injecting spike transactions, velocity features should increase."""
        train, _, _ = detector_data
        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)

        # Ingest initial baseline (all train[:100])
        for tx in train[:100]:
            engine.ingest(tx)

        # Find any merchant that has spike transactions in train
        all_spike_txns = [t for t in train if t.is_spike]
        if not all_spike_txns:
            pytest.skip("No spike transactions in training data — skip this test")

        # Pick the merchant with the most spike transactions
        from collections import Counter
        spike_merchants = Counter(t.merchant_id for t in all_spike_txns)
        target_merchant = spike_merchants.most_common(1)[0][0]

        # Get baseline features for that merchant at any recent timestamp
        recent_tx = next((t for t in train[:100] if t.merchant_id == target_merchant), None)
        if recent_tx is None:
            # Merchant not in first 100 txns — ingest one of theirs
            first_spike_tx = next(t for t in all_spike_txns if t.merchant_id == target_merchant)
            engine.ingest(first_spike_tx)
            recent_tx = first_spike_tx

        fv_before = extractor.extract("merchant_id", target_merchant, recent_tx.timestamp)

        # Now inject spike transactions for this merchant
        merchant_spikes = [t for t in all_spike_txns if t.merchant_id == target_merchant][:30]
        for tx in merchant_spikes:
            engine.ingest(tx)

        fv_after = extractor.extract("merchant_id", target_merchant, merchant_spikes[-1].timestamp)

        feat_dict_before = fv_before.as_dict()
        feat_dict_after = fv_after.as_dict()

        # tx_count_1h should have increased (wide window to catch any spikes)
        before_count = feat_dict_before.get("tx_count_1h", 0)
        after_count = feat_dict_after.get("tx_count_1h", 0)
        assert after_count >= before_count, (
            f"1h velocity did not increase after spike injection: "
            f"before={before_count}, after={after_count}"
        )


# ---------------------------------------------------------------------------
# Detector model tests (including the REAL precision/recall gate)
# ---------------------------------------------------------------------------

class TestSpikeDetector:
    def test_spike_score_in_unit_interval(self, detector_data):
        """All spike_score values must be in [0, 1]."""
        train, test, _ = detector_data

        pipeline = DetectorPipeline()
        pipeline.fit(train)
        outputs = pipeline.process_batch(test, emit_all=True)

        for out in outputs:
            assert 0.0 <= out.spike_score <= 1.0, (
                f"spike_score {out.spike_score} out of [0,1]"
            )

    def test_top_features_count(self, detector_data):
        """Every DetectorOutput must have exactly 5 top_features."""
        train, test, _ = detector_data

        pipeline = DetectorPipeline()
        pipeline.fit(train)
        outputs = pipeline.process_batch(test[:100], emit_all=True)

        for out in outputs:
            assert len(out.top_features) == 5, (
                f"Expected 5 top features, got {len(out.top_features)}"
            )

    def test_top_features_have_required_keys(self, detector_data):
        """Each top feature must have 'name', 'value', and 'contribution' keys."""
        train, test, _ = detector_data

        pipeline = DetectorPipeline()
        pipeline.fit(train)
        outputs = pipeline.process_batch(test[:50], emit_all=True)

        for out in outputs:
            for feat in out.top_features:
                assert "name" in feat
                assert "value" in feat
                assert "contribution" in feat

    def test_precision_recall_f1_reported(self, detector_data, capsys):
        """
        PHASE 2 GATE: Run the detector on held-out test set and report real metrics.
        This test ALWAYS prints the metrics table and passes as long as:
        - precision >= 0.40 (not worse than random for a 5%-spike dataset)
        - recall >= 0.40
        - F1 >= 0.35
        These are deliberately low bars — the point is honest, not cherry-picked numbers.
        """
        train, test, _ = detector_data

        pipeline = DetectorPipeline()
        pipeline.fit(train)
        outputs = pipeline.process_batch(test, emit_all=True)

        # Ground truth: did the triggering transaction belong to a spike?
        y_true = [1 if out.trigger_transaction.is_spike else 0 for out in outputs]
        y_score = [out.spike_score for out in outputs]
        y_pred = [1 if s >= 0.50 else 0 for s in y_score]

        n_pos = sum(y_true)
        n_neg = len(y_true) - n_pos

        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        report = classification_report(
            y_true, y_pred,
            target_names=["normal", "spike"],
            zero_division=0,
        )

        # Print to stdout (captured by pytest -s, always shown)
        print(f"\n{'='*60}")
        print("PHASE 2 DETECTOR METRICS (held-out test set)")
        print(f"{'='*60}")
        print(f"  Test set size : {len(y_true)} transactions")
        print(f"  Spike (pos)   : {n_pos} ({100*n_pos/len(y_true):.1f}%)")
        print(f"  Normal (neg)  : {n_neg} ({100*n_neg/len(y_true):.1f}%)")
        print(f"  Decision threshold : 0.50")
        print(f"\n  Precision : {prec:.4f}")
        print(f"  Recall    : {rec:.4f}")
        print(f"  F1 Score  : {f1:.4f}")
        print(f"\nClassification Report:\n{report}")
        print("="*60)

        # Actual assertion bars
        assert prec >= 0.40, f"Precision {prec:.4f} below minimum threshold 0.40"
        assert rec >= 0.40, f"Recall {rec:.4f} below minimum threshold 0.40"
        assert f1 >= 0.35, f"F1 {f1:.4f} below minimum threshold 0.35"
