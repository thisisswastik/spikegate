"""
data_gen/generator.py — Synthetic Razorpay-style transaction generator with spike injection.

Design:
- 50 merchants, each with a risk tier and typical transaction profile
- Card BINs, device IDs, IPs are drawn from per-merchant pools to simulate realistic entity affinity
- Spike injection: at random intervals, a SpikeBurst fires on a chosen entity dimension,
  multiplying velocity by 5–20x for a window of 30–300 seconds
- All spike transactions carry is_spike=True and spike_id for ground-truth evaluation
- Fully deterministic given a fixed seed
"""
from __future__ import annotations

import math
import random
import string
import uuid
from datetime import datetime, timedelta, timezone
from typing import Iterator

import numpy as np

from data_gen.schema import (
    MerchantRiskTier,
    PaymentMethod,
    SpikeBurst,
    Transaction,
    TransactionStatus,
)


# ---------------------------------------------------------------------------
# Merchant profile helpers
# ---------------------------------------------------------------------------

RISK_TIER_WEIGHTS = {
    MerchantRiskTier.LOW: 0.60,
    MerchantRiskTier.MEDIUM: 0.30,
    MerchantRiskTier.HIGH: 0.10,
}

# Amount ranges (min, max) in INR per risk tier
AMOUNT_RANGES = {
    MerchantRiskTier.LOW: (50, 5_000),
    MerchantRiskTier.MEDIUM: (200, 25_000),
    MerchantRiskTier.HIGH: (500, 1_00_000),
}

# Payment method probability distributions per risk tier
PAYMENT_METHOD_PROBS = {
    MerchantRiskTier.LOW: {
        PaymentMethod.UPI: 0.55,
        PaymentMethod.CARD: 0.25,
        PaymentMethod.NETBANKING: 0.10,
        PaymentMethod.WALLET: 0.08,
        PaymentMethod.EMI: 0.02,
    },
    MerchantRiskTier.MEDIUM: {
        PaymentMethod.UPI: 0.40,
        PaymentMethod.CARD: 0.35,
        PaymentMethod.NETBANKING: 0.12,
        PaymentMethod.WALLET: 0.08,
        PaymentMethod.EMI: 0.05,
    },
    MerchantRiskTier.HIGH: {
        PaymentMethod.UPI: 0.25,
        PaymentMethod.CARD: 0.50,
        PaymentMethod.NETBANKING: 0.10,
        PaymentMethod.WALLET: 0.05,
        PaymentMethod.EMI: 0.10,
    },
}

# Failure rates per risk tier (fraction of transactions that fail)
FAILURE_RATES = {
    MerchantRiskTier.LOW: 0.05,
    MerchantRiskTier.MEDIUM: 0.10,
    MerchantRiskTier.HIGH: 0.18,
}


def _random_id(prefix: str, n: int = 8, rng: random.Random = None) -> str:
    """Generate a random alphanumeric ID like 'pay_Xy3Kp9Qz'."""
    chars = string.ascii_letters + string.digits
    r = rng or random
    return prefix + "".join(r.choices(chars, k=n))


def _random_ip(rng: random.Random) -> str:
    """Generate a random IPv4 address."""
    return ".".join(str(rng.randint(1, 254)) for _ in range(4))


def _random_card_bin(rng: random.Random) -> str:
    """Generate a realistic-looking 6-digit card BIN."""
    # Common BIN prefixes: Visa (4xxx), Mastercard (5xxx), RuPay (6xxx)
    prefix = rng.choice(["4", "51", "52", "53", "54", "55", "60"])
    remaining = 6 - len(prefix)
    return prefix + "".join(str(rng.randint(0, 9)) for _ in range(remaining))


# ---------------------------------------------------------------------------
# MerchantProfile
# ---------------------------------------------------------------------------

class MerchantProfile:
    """Encapsulates per-merchant configuration and entity pools."""

    def __init__(self, merchant_id: str, risk_tier: MerchantRiskTier, rng: random.Random):
        self.merchant_id = merchant_id
        self.risk_tier = risk_tier

        # Each merchant has a pool of card BINs, device IDs, and IPs
        # that customers typically use — creating entity-merchant affinity
        pool_size_multiplier = {"low": 1.0, "medium": 1.5, "high": 2.0}[risk_tier]
        n_bins = int(rng.randint(10, 30) * pool_size_multiplier)
        n_devices = int(rng.randint(20, 60) * pool_size_multiplier)
        n_ips = int(rng.randint(15, 40) * pool_size_multiplier)

        self.card_bins: list[str] = [_random_card_bin(rng) for _ in range(n_bins)]
        self.device_ids: list[str] = [_random_id("dev_", 12, rng) for _ in range(n_devices)]
        self.ip_pool: list[str] = [_random_ip(rng) for _ in range(n_ips)]

        # Per-merchant TPS weight (some merchants are busier than others)
        self.tps_weight: float = rng.uniform(0.5, 3.0)

        # Amount distribution parameters (lognormal)
        amt_min, amt_max = AMOUNT_RANGES[risk_tier]
        self.amount_mean = rng.uniform(amt_min, amt_max / 2)
        self.amount_std = self.amount_mean * rng.uniform(0.3, 0.8)

    def sample_transaction(
        self,
        timestamp: datetime,
        rng: random.Random,
        np_rng: np.random.Generator,
        is_spike: bool = False,
        spike_id: str | None = None,
    ) -> Transaction:
        """Sample a single transaction from this merchant's profile."""
        method = rng.choices(
            list(PAYMENT_METHOD_PROBS[self.risk_tier].keys()),
            weights=list(PAYMENT_METHOD_PROBS[self.risk_tier].values()),
        )[0]

        # Card BIN only for card payments
        card_bin = rng.choice(self.card_bins) if method == PaymentMethod.CARD else None

        # Amount: lognormal clamped to risk tier range
        amt_min, amt_max = AMOUNT_RANGES[self.risk_tier]
        raw_amount = float(np_rng.lognormal(
            mean=np.log(max(self.amount_mean, 1)),
            sigma=0.5,
        ))
        amount = float(np.clip(raw_amount, amt_min, amt_max))
        amount = round(amount, 2)

        # Status
        fail_rate = FAILURE_RATES[self.risk_tier]
        status = (
            TransactionStatus.FAILED
            if rng.random() < fail_rate
            else TransactionStatus.SUCCESS
        )

        return Transaction(
            payment_id=_random_id("pay_", 14, rng),
            merchant_id=self.merchant_id,
            card_bin=card_bin,
            device_id=rng.choice(self.device_ids),
            ip_address=rng.choice(self.ip_pool),
            amount_inr=amount,
            payment_method=method,
            status=status,
            merchant_risk_tier=self.risk_tier,
            timestamp=timestamp,
            is_spike=is_spike,
            spike_id=spike_id,
        )


# ---------------------------------------------------------------------------
# TransactionGenerator
# ---------------------------------------------------------------------------

class TransactionGenerator:
    """
    Generates a stream of Razorpay-style transactions with injected spike bursts.

    Parameters
    ----------
    n_merchants : int
        Number of synthetic merchants.
    base_tps : float
        Total baseline transactions per second across all merchants.
    spike_prob : float
        Probability per second that a new spike burst starts on some entity.
    spike_multiplier_range : tuple[float, float]
        Min/max velocity multiplier for spike bursts.
    spike_duration_range : tuple[float, float]
        Min/max duration (seconds) for spike bursts.
    simulation_hours : float
        Total simulated duration to generate.
    seed : int
        Random seed for full reproducibility.
    """

    def __init__(
        self,
        n_merchants: int = 50,
        base_tps: float = 2.0,
        spike_prob: float = 0.02,
        spike_multiplier_range: tuple[float, float] = (5.0, 20.0),
        spike_duration_range: tuple[float, float] = (30.0, 300.0),
        simulation_hours: float = 24.0,
        seed: int = 42,
        start_time: datetime | None = None,
    ):
        self.n_merchants = n_merchants
        self.base_tps = base_tps
        self.spike_prob = spike_prob
        self.spike_multiplier_range = spike_multiplier_range
        self.spike_duration_range = spike_duration_range
        self.simulation_hours = simulation_hours
        self.seed = seed

        self.start_time = start_time or datetime(2024, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
        self.end_time = self.start_time + timedelta(hours=simulation_hours)

        # Reproducible RNGs
        self._rng = random.Random(seed)
        self._np_rng = np.random.default_rng(seed)

        # Build merchant profiles
        risk_tiers = self._rng.choices(
            list(RISK_TIER_WEIGHTS.keys()),
            weights=list(RISK_TIER_WEIGHTS.values()),
            k=n_merchants,
        )
        self.merchants: list[MerchantProfile] = [
            MerchantProfile(
                merchant_id=_random_id("mid_", 8, self._rng),
                risk_tier=tier,
                rng=random.Random(seed + i + 1),
            )
            for i, tier in enumerate(risk_tiers)
        ]

        # Normalise TPS weights
        total_weight = sum(m.tps_weight for m in self.merchants)
        for m in self.merchants:
            m.tps_weight /= total_weight

    def generate_batch(self) -> tuple[list[Transaction], list[SpikeBurst]]:
        """
        Generate all transactions for the simulation period as a sorted list.

        Returns
        -------
        transactions : list[Transaction]
            All transactions, sorted by timestamp.
        spike_bursts : list[SpikeBurst]
            All injected spike bursts (ground-truth).
        """
        # Rebuild RNGs from seed for determinism
        rng = random.Random(self.seed)
        np_rng = np.random.default_rng(self.seed)

        # Fresh merchant RNGs (same seed → same profiles)
        merchant_rngs = [random.Random(self.seed + i + 1) for i in range(self.n_merchants)]

        transactions: list[Transaction] = []
        spike_bursts: list[SpikeBurst] = []

        # Track active spikes: entity_key → (SpikeBurst, multiplier)
        active_spikes: dict[tuple[str, str], SpikeBurst] = {}

        # Simulate second-by-second
        sim_seconds = int(self.simulation_hours * 3600)
        current_time = self.start_time

        for sec in range(sim_seconds):
            current_time = self.start_time + timedelta(seconds=sec)

            # --- Expire old spikes ---
            expired = [
                k for k, burst in active_spikes.items()
                if burst.end_time <= current_time
            ]
            for k in expired:
                del active_spikes[k]

            # --- Maybe inject a new spike ---
            if rng.random() < self.spike_prob:
                spike_burst = self._inject_spike(rng, current_time)
                entity_key = (spike_burst.entity_type, spike_burst.entity_id)
                active_spikes[entity_key] = spike_burst
                spike_bursts.append(spike_burst)

            # --- Sample transactions for this second ---
            # Expected number this second = base_tps
            n_this_second = int(np_rng.poisson(self.base_tps))

            # Add spike transactions
            spike_extra: list[tuple[MerchantProfile, SpikeBurst, tuple[str, str]]] = []
            for (etype, eid), burst in active_spikes.items():
                extra_count = int(np_rng.poisson(
                    self.base_tps * (burst.multiplier - 1) / len(self.merchants)
                ))
                # Find a merchant whose pool contains this entity
                matched_merchants = self._find_merchants_for_entity(etype, eid)
                for _ in range(extra_count):
                    if matched_merchants:
                        m = rng.choice(matched_merchants)
                        spike_extra.append((m, burst, (etype, eid)))

            # Normal baseline transactions
            for _ in range(n_this_second):
                # Pick merchant by TPS weight
                merchant_idx = rng.choices(
                    range(self.n_merchants),
                    weights=[m.tps_weight for m in self.merchants],
                )[0]
                m = self.merchants[merchant_idx]
                m_rng = merchant_rngs[merchant_idx]

                # Sub-second jitter
                jitter_ms = rng.randint(0, 999)
                ts = current_time + timedelta(milliseconds=jitter_ms)

                tx = m.sample_transaction(ts, m_rng, np_rng)
                transactions.append(tx)

            # Spike extra transactions
            for m, burst, (etype, eid) in spike_extra:
                idx = self.merchants.index(m)
                m_rng = merchant_rngs[idx]
                jitter_ms = rng.randint(0, 999)
                ts = current_time + timedelta(milliseconds=jitter_ms)

                # Override the entity field that's spiking
                tx = m.sample_transaction(ts, m_rng, np_rng, is_spike=True, spike_id=burst.spike_id)

                # Force the spiking entity onto this transaction
                tx = self._override_entity(tx, etype, eid, rng)
                transactions.append(tx)

        # Sort by timestamp
        transactions.sort(key=lambda t: t.timestamp)
        return transactions, spike_bursts

    def _inject_spike(self, rng: random.Random, current_time: datetime) -> SpikeBurst:
        """Create a new SpikeBurst on a random entity dimension."""
        entity_type = rng.choice(["merchant_id", "card_bin", "device_id", "ip_address"])

        # Pick an entity from the merchant pool
        merchant = rng.choice(self.merchants)
        if entity_type == "merchant_id":
            entity_id = merchant.merchant_id
        elif entity_type == "card_bin":
            entity_id = rng.choice(merchant.card_bins)
        elif entity_type == "device_id":
            entity_id = rng.choice(merchant.device_ids)
        else:  # ip_address
            entity_id = rng.choice(merchant.ip_pool)

        duration = rng.uniform(*self.spike_duration_range)
        multiplier = rng.uniform(*self.spike_multiplier_range)

        # Round end_time up to the next whole second so the per-transaction
        # jitter (0–999 ms) can never push a spike tx past the burst boundary.
        raw_end = current_time + timedelta(seconds=duration)
        end_seconds_ceil = math.ceil(raw_end.timestamp())
        end_time = datetime.fromtimestamp(end_seconds_ceil, tz=current_time.tzinfo)

        return SpikeBurst(
            spike_id=str(uuid.uuid4()),
            entity_type=entity_type,
            entity_id=entity_id,
            start_time=current_time,
            end_time=end_time,
            multiplier=multiplier,
            n_transactions=0,  # updated post-generation
        )

    def _find_merchants_for_entity(
        self, entity_type: str, entity_id: str
    ) -> list[MerchantProfile]:
        """Find merchants whose entity pools contain the given entity."""
        result = []
        for m in self.merchants:
            if entity_type == "merchant_id" and m.merchant_id == entity_id:
                result.append(m)
            elif entity_type == "card_bin" and entity_id in m.card_bins:
                result.append(m)
            elif entity_type == "device_id" and entity_id in m.device_ids:
                result.append(m)
            elif entity_type == "ip_address" and entity_id in m.ip_pool:
                result.append(m)
        # If no exact match, fall back to any merchant (spike on a new entity)
        return result or [self.merchants[0]]

    @staticmethod
    def _override_entity(
        tx: Transaction, entity_type: str, entity_id: str, rng: random.Random
    ) -> Transaction:
        """Return a copy of tx with the spiking entity field set to entity_id."""
        data = tx.model_dump()
        if entity_type == "merchant_id":
            data["merchant_id"] = entity_id
        elif entity_type == "card_bin":
            data["card_bin"] = entity_id
            data["payment_method"] = PaymentMethod.CARD.value
        elif entity_type == "device_id":
            data["device_id"] = entity_id
        elif entity_type == "ip_address":
            data["ip_address"] = entity_id
        return Transaction(**data)
