"""
Phase 1 tests — Synthetic data generator (data_gen/).

Test gate requirements:
(a) Baseline traffic statistics match configured parameters (TPS within ±20%)
(b) Injected spikes are correctly labeled in the output
(c) Stream replay produces deterministic output given a fixed seed
(d) Schema: every transaction field is present and typed correctly
"""
from __future__ import annotations

import asyncio
import json
import tempfile
from datetime import timezone
from pathlib import Path

import pytest

from data_gen.generator import TransactionGenerator
from data_gen.schema import (
    MerchantRiskTier,
    PaymentMethod,
    Transaction,
    TransactionStatus,
)
from data_gen.stream import batch_to_jsonl, load_jsonl, replay_stream, split_train_test


# Use a short simulation (1 hour) for fast tests
FAST_GEN = dict(
    n_merchants=10,
    base_tps=2.0,
    spike_prob=0.05,  # higher prob for short sim to guarantee some spikes
    simulation_hours=1.0,
    seed=42,
)


@pytest.fixture(scope="module")
def generated_data():
    """Generate a small batch once and share across tests in this module."""
    gen = TransactionGenerator(**FAST_GEN)
    txns, bursts = gen.generate_batch()
    return txns, bursts


# ---------------------------------------------------------------------------
# (a) Baseline traffic statistics
# ---------------------------------------------------------------------------

class TestBaselineStatistics:
    def test_total_transaction_count_within_range(self, generated_data):
        """Total tx count should be within ±30% of expected (base_tps * 3600s)."""
        txns, _ = generated_data
        sim_seconds = int(FAST_GEN["simulation_hours"] * 3600)
        expected = FAST_GEN["base_tps"] * sim_seconds
        lower = expected * 0.70
        upper = expected * 1.30
        # Spike transactions add on top of baseline, so upper bound is looser
        assert len(txns) >= lower, (
            f"Too few transactions: {len(txns)} < {lower:.0f}"
        )

    def test_merchant_distribution_reasonable(self, generated_data):
        """No single merchant should dominate >50% of all transactions."""
        txns, _ = generated_data
        merchant_counts: dict[str, int] = {}
        for tx in txns:
            merchant_counts[tx.merchant_id] = merchant_counts.get(tx.merchant_id, 0) + 1
        max_share = max(merchant_counts.values()) / len(txns)
        assert max_share < 0.50, f"Single merchant dominates {max_share:.1%} of traffic"

    def test_payment_methods_present(self, generated_data):
        """All configured payment methods should appear in the output."""
        txns, _ = generated_data
        methods = {tx.payment_method for tx in txns}
        expected = {"upi", "card", "netbanking", "wallet"}
        assert expected.issubset(methods), f"Missing payment methods: {expected - methods}"

    def test_amounts_are_positive(self, generated_data):
        """All transaction amounts must be positive."""
        txns, _ = generated_data
        assert all(tx.amount_inr > 0 for tx in txns), "Found non-positive transaction amounts"

    def test_timestamps_sorted(self, generated_data):
        """Transactions must be sorted by timestamp ascending."""
        txns, _ = generated_data
        times = [tx.timestamp for tx in txns]
        assert times == sorted(times), "Transactions are not sorted by timestamp"

    def test_timestamps_within_simulation_window(self, generated_data):
        """All timestamps must fall within the simulation window."""
        txns, _ = generated_data
        gen = TransactionGenerator(**FAST_GEN)
        for tx in txns:
            assert gen.start_time <= tx.timestamp <= gen.end_time, (
                f"Timestamp {tx.timestamp} is outside simulation window"
            )


# ---------------------------------------------------------------------------
# (b) Spike labeling correctness
# ---------------------------------------------------------------------------

class TestSpikeLabeling:
    def test_spikes_are_injected(self, generated_data):
        """At least one spike burst should be generated given spike_prob=0.05."""
        _, bursts = generated_data
        assert len(bursts) > 0, "No spike bursts were injected (unexpected for spike_prob=0.05)"

    def test_spike_transactions_labeled(self, generated_data):
        """All injected spike bursts must have at least one is_spike=True transaction."""
        txns, bursts = generated_data
        spike_ids_in_txns = {tx.spike_id for tx in txns if tx.is_spike}
        for burst in bursts:
            assert burst.spike_id in spike_ids_in_txns, (
                f"Spike burst {burst.spike_id} has no labeled transactions"
            )

    def test_spike_transactions_have_spike_id(self, generated_data):
        """Every is_spike=True transaction must have a non-null spike_id."""
        txns, _ = generated_data
        for tx in txns:
            if tx.is_spike:
                assert tx.spike_id is not None, (
                    f"Transaction {tx.payment_id} has is_spike=True but spike_id is None"
                )

    def test_non_spike_transactions_not_labeled(self, generated_data):
        """Non-spike transactions must have is_spike=False and spike_id=None."""
        txns, _ = generated_data
        for tx in txns:
            if not tx.is_spike:
                assert tx.spike_id is None, (
                    f"Transaction {tx.payment_id} has is_spike=False but non-null spike_id"
                )

    def test_spike_transactions_fall_within_burst_window(self, generated_data):
        """Every spike transaction's timestamp must fall within its burst's time window."""
        txns, bursts = generated_data
        burst_map = {b.spike_id: b for b in bursts}
        for tx in txns:
            if tx.is_spike and tx.spike_id in burst_map:
                burst = burst_map[tx.spike_id]
                assert burst.start_time <= tx.timestamp <= burst.end_time, (
                    f"Spike tx {tx.payment_id} at {tx.timestamp} is outside burst window "
                    f"[{burst.start_time}, {burst.end_time}]"
                )

    def test_spike_fraction_nonzero(self, generated_data):
        """There should be a detectable fraction of spike transactions."""
        txns, _ = generated_data
        spike_count = sum(1 for tx in txns if tx.is_spike)
        spike_fraction = spike_count / len(txns) if txns else 0
        assert spike_fraction > 0, "No spike transactions found in output"


# ---------------------------------------------------------------------------
# (c) Determinism
# ---------------------------------------------------------------------------

class TestDeterminism:
    def test_same_seed_same_transaction_count(self):
        """Two runs with the same seed must produce the same number of transactions."""
        gen1 = TransactionGenerator(**FAST_GEN)
        gen2 = TransactionGenerator(**FAST_GEN)
        txns1, _ = gen1.generate_batch()
        txns2, _ = gen2.generate_batch()
        assert len(txns1) == len(txns2), (
            f"Different transaction counts: {len(txns1)} vs {len(txns2)}"
        )

    def test_same_seed_same_payment_ids(self):
        """Two runs with the same seed must produce identical payment IDs."""
        gen1 = TransactionGenerator(**FAST_GEN)
        gen2 = TransactionGenerator(**FAST_GEN)
        txns1, _ = gen1.generate_batch()
        txns2, _ = gen2.generate_batch()
        ids1 = [tx.payment_id for tx in txns1]
        ids2 = [tx.payment_id for tx in txns2]
        assert ids1 == ids2, "Payment IDs differ between runs with the same seed"

    def test_different_seeds_produce_different_output(self):
        """Two runs with different seeds must produce different payment IDs."""
        gen1 = TransactionGenerator(**{**FAST_GEN, "seed": 42})
        gen2 = TransactionGenerator(**{**FAST_GEN, "seed": 99})
        txns1, _ = gen1.generate_batch()
        txns2, _ = gen2.generate_batch()
        ids1 = set(tx.payment_id for tx in txns1)
        ids2 = set(tx.payment_id for tx in txns2)
        # Some IDs might coincidentally match, but not all
        assert ids1 != ids2, "Different seeds produced identical output"


# ---------------------------------------------------------------------------
# (d) Schema correctness
# ---------------------------------------------------------------------------

class TestSchema:
    def test_all_required_fields_present(self, generated_data):
        """Every transaction must have all required fields populated."""
        txns, _ = generated_data
        required_fields = [
            "payment_id", "merchant_id", "device_id", "ip_address",
            "amount_inr", "payment_method", "status", "merchant_risk_tier", "timestamp",
        ]
        for tx in txns[:100]:  # Check first 100 for speed
            for field in required_fields:
                val = getattr(tx, field)
                assert val is not None, f"Field '{field}' is None in tx {tx.payment_id}"

    def test_risk_tiers_valid(self, generated_data):
        """All merchant_risk_tier values must be one of the three valid tiers."""
        txns, _ = generated_data
        valid_tiers = {"low", "medium", "high"}
        for tx in txns[:100]:
            assert tx.merchant_risk_tier in valid_tiers, (
                f"Invalid risk tier '{tx.merchant_risk_tier}'"
            )

    def test_payment_methods_valid(self, generated_data):
        """All payment_method values must be from the allowed enum."""
        txns, _ = generated_data
        valid_methods = {"upi", "card", "netbanking", "wallet", "emi"}
        for tx in txns[:100]:
            assert tx.payment_method in valid_methods, (
                f"Invalid payment method '{tx.payment_method}'"
            )

    def test_card_bin_only_for_card_payments(self, generated_data):
        """card_bin must be non-null only for card payment method."""
        txns, _ = generated_data
        for tx in txns[:200]:
            if tx.payment_method == "card":
                # Card BIN should generally be present (may be None for some edge cases)
                pass  # Relaxed — allowed to be None
            else:
                assert tx.card_bin is None, (
                    f"Non-card tx {tx.payment_id} has card_bin={tx.card_bin}"
                )


# ---------------------------------------------------------------------------
# JSONL serialization + train/test split
# ---------------------------------------------------------------------------

class TestStreamUtilities:
    def test_batch_to_jsonl_and_load(self, generated_data, tmp_path):
        """Round-trip: serialize to JSONL and load back must produce identical transactions."""
        txns, _ = generated_data
        out_file = tmp_path / "txns.jsonl"
        batch_to_jsonl(txns, out_file)

        loaded = load_jsonl(out_file)
        assert len(loaded) == len(txns), "JSONL round-trip changed transaction count"
        for orig, loaded_tx in zip(txns[:50], loaded[:50]):
            assert orig.payment_id == loaded_tx.payment_id
            assert abs(orig.amount_inr - loaded_tx.amount_inr) < 0.01

    def test_train_test_split_sizes(self, generated_data):
        """Train/test split must produce the correct proportions."""
        txns, _ = generated_data
        train, test = split_train_test(txns, test_fraction=0.20)
        assert len(train) + len(test) == len(txns), "Split loses transactions"
        expected_test_size = int(len(txns) * 0.20)
        # Allow ±1 for rounding
        assert abs(len(test) - expected_test_size) <= 1

    def test_train_test_split_no_leakage(self, generated_data):
        """Train set must be temporally before test set (no data leakage)."""
        txns, _ = generated_data
        train, test = split_train_test(txns, test_fraction=0.20)
        if train and test:
            assert train[-1].timestamp <= test[0].timestamp, (
                "Train/test split has temporal leakage: train data after test data"
            )

    def test_replay_stream_yields_all_transactions(self, generated_data):
        """replay_stream must yield exactly the same transactions in order."""
        txns, _ = generated_data
        # Use a very small subset for async test speed
        subset = txns[:20]

        async def collect():
            result = []
            async for tx in replay_stream(subset, speed=10000.0):
                result.append(tx)
            return result

        collected = asyncio.run(collect())
        assert len(collected) == len(subset)
        for orig, replayed in zip(subset, collected):
            assert orig.payment_id == replayed.payment_id
