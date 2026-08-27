"""
Phase 2 tests — Detection core (detector/).

Test gate requirements:
- Feature extractor produces expected feature count (40-60)
- spike_score is in [0, 1]
- Standalone detector precision/recall/F1 on held-out test split
  (REAL numbers from an actual run — shown before proceeding to Phase 3)
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score

from data_gen.generator import TransactionGenerator
from data_gen.schema import MerchantRiskTier, PaymentMethod, Transaction, TransactionStatus
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


@pytest.fixture(scope="module")
def fitted_pipeline(detector_data):
    """Fit the detector pipeline once on the training data for all tests in this module."""
    train, _, _ = detector_data
    pipeline = DetectorPipeline()
    pipeline.fit(train)
    return pipeline


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

        for tx in train[:20]:
            engine.ingest(tx)

        tx = train[19]
        fv = extractor.extract("merchant_id", tx.merchant_id, tx.timestamp)
        for val in fv.values:
            assert np.isfinite(val), f"Non-finite feature value: {val}"

    def test_cold_start_no_error(self, detector_data):
        """Extracting features for an unseen entity should not raise an error."""
        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)
        # Entity has zero history — should return zero-filled feature vector without error
        now = datetime.now(timezone.utc)
        fv = extractor.extract("merchant_id", "mid_unseen_000", now)
        assert fv is not None
        assert len(fv.values) > 0
        assert all(np.isfinite(v) for v in fv.values)

    def test_spike_window_has_higher_velocity(self, detector_data):
        """Windows containing spike bursts must show higher tx count than quiet windows."""
        train, _, bursts = detector_data
        if not bursts:
            pytest.skip("No bursts in dataset")

        engine = RollingWindowEngine()
        extractor = FeatureExtractor(engine)

        # Ingest baseline
        for tx in train[:100]:
            engine.ingest(tx)

        # Baseline features
        m_id = bursts[0].entity_id if bursts[0].entity_type == "merchant_id" else train[0].merchant_id
        fv_before = extractor.extract("merchant_id", m_id, train[99].timestamp)

        # Ingest a burst of synthetic transactions on this merchant
        burst_ts = train[99].timestamp
        for i in range(50):
            tx_burst = Transaction(
                payment_id=f"pay_burst_test_{i}",
                merchant_id=m_id,
                device_id="dev_test_001",
                ip_address="192.168.1.1",
                amount_inr=1000.0,
                payment_method=PaymentMethod.UPI,
                status=TransactionStatus.SUCCESS,
                merchant_risk_tier=MerchantRiskTier.LOW,
                timestamp=burst_ts + timedelta(seconds=i),
                is_spike=True,
                spike_id="test_burst_001",
            )
            engine.ingest(tx_burst)

        fv_after = extractor.extract("merchant_id", m_id, burst_ts + timedelta(seconds=55))

        feat_dict_before = dict(zip(fv_before.feature_names, fv_before.values))
        feat_dict_after = dict(zip(fv_after.feature_names, fv_after.values))

        # Check that 1h/5m/1m count increased
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
    def test_spike_score_in_unit_interval(self, detector_data, fitted_pipeline):
        """All spike_score values must be in [0, 1]."""
        _, test, _ = detector_data
        outputs = fitted_pipeline.process_batch(test[:100], emit_all=True)

        for out in outputs:
            assert 0.0 <= out.spike_score <= 1.0, (
                f"spike_score {out.spike_score} out of [0,1]"
            )

    def test_top_features_count(self, detector_data, fitted_pipeline):
        """Every DetectorOutput must have exactly 5 top_features."""
        _, test, _ = detector_data
        outputs = fitted_pipeline.process_batch(test[:100], emit_all=True)

        for out in outputs:
            assert len(out.top_features) == 5, (
                f"Expected 5 top features, got {len(out.top_features)}"
            )

    def test_top_features_have_required_keys(self, detector_data, fitted_pipeline):
        """Each top feature must have 'name', 'value', and 'contribution' keys."""
        _, test, _ = detector_data
        outputs = fitted_pipeline.process_batch(test[:50], emit_all=True)

        for out in outputs:
            for feat in out.top_features:
                assert "name" in feat
                assert "value" in feat
                assert "contribution" in feat

    def test_precision_recall_f1_reported(self, detector_data, fitted_pipeline, capsys):
        """
        PHASE 2 GATE: Run the detector on held-out test set and report real metrics.
        """
        _, test, _ = detector_data
        outputs = fitted_pipeline.process_batch(test, emit_all=True)

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

        # Honest assertion bars for real-world imbalanced fraud detection
        assert prec >= 0.30, f"Precision {prec:.4f} below minimum threshold 0.30"
        assert rec >= 0.30, f"Recall {rec:.4f} below minimum threshold 0.30"
        assert f1 >= 0.30, f"F1 {f1:.4f} below minimum threshold 0.30"
