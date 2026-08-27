"""
detector/features.py — Rolling-window feature extractor.

Computes 50+ interpretable features per (entity_type, entity_id) tuple using
records from the RollingWindowEngine.  Feature groups:

1. Velocity            — tx count per window
2. Amount              — sum / mean / std / max per window
3. Counterparty        — unique merchants/cards/devices/IPs per window
4. Failure             — fail count and fail rate per window
5. Payment method mix  — fraction UPI/card/netbanking/wallet per window
6. Geo dispersion      — unique /24 subnets per window
7. Temporal            — hour-of-day, is_weekend, seconds-since-last-tx
8. Merchant tier       — risk tier encoded
9. Recency ratios      — short-window metric / long-window metric

All features are numeric (float).  Categorical fields (payment_method,
merchant_risk_tier) are one-hot or ordinal encoded inline so the
downstream XGBoost model receives a pure float vector.
"""
from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np

from detector.windows import RollingWindowEngine, TxRecord, WINDOW_SIZES

# Ordinal encoding for merchant risk tier
TIER_ENCODING = {"low": 0.0, "medium": 1.0, "high": 2.0}

# Payment methods we track
PAYMENT_METHODS = ["upi", "card", "netbanking", "wallet", "emi"]


class FeatureVector(NamedTuple):
    """Named container for the feature vector and its metadata."""
    entity_type: str
    entity_id: str
    timestamp: datetime
    feature_names: list[str]
    values: np.ndarray          # shape (n_features,)

    def as_dict(self) -> dict[str, float]:
        return dict(zip(self.feature_names, self.values.tolist()))


def _safe_std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    return float(np.std(values, ddof=0))


def _safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else 0.0


def _unique_slash24(ips: list[str]) -> int:
    """Count unique /24 subnets in a list of IPv4 addresses."""
    return len({".".join(ip.split(".")[:3]) for ip in ips})


class FeatureExtractor:
    """
    Computes the full feature vector for a given entity at a given timestamp.

    Parameters
    ----------
    engine : RollingWindowEngine
        The window engine that has already ingested all transactions up to now.
    """

    def __init__(self, engine: RollingWindowEngine):
        self.engine = engine
        self._feature_names: list[str] | None = None

    @property
    def feature_names(self) -> list[str]:
        if self._feature_names is None:
            self._feature_names = self._build_feature_names()
        return self._feature_names

    @property
    def n_features(self) -> int:
        return len(self.feature_names)

    def _build_feature_names(self) -> list[str]:
        """Return the ordered list of feature names (must match extract() output)."""
        names: list[str] = []

        # 1. Velocity
        for wk in WINDOW_SIZES:
            names.append(f"tx_count_{wk}")

        # 2. Amount per window
        for wk in WINDOW_SIZES:
            names += [
                f"amt_sum_{wk}",
                f"amt_mean_{wk}",
                f"amt_std_{wk}",
                f"amt_max_{wk}",
            ]

        # 3. Counterparty diversity per window
        for wk in WINDOW_SIZES:
            names += [
                f"unique_merchants_{wk}",
                f"unique_cards_{wk}",
                f"unique_devices_{wk}",
                f"unique_ips_{wk}",
            ]

        # 4. Failure rate per window
        for wk in WINDOW_SIZES:
            names += [
                f"fail_count_{wk}",
                f"fail_rate_{wk}",
            ]

        # 5. Payment method fractions per window (5 methods × 3 windows = 15)
        for wk in WINDOW_SIZES:
            for method in PAYMENT_METHODS:
                names.append(f"pct_{method}_{wk}")

        # 6. Geo dispersion per window
        for wk in WINDOW_SIZES:
            names.append(f"geo_dispersion_{wk}")

        # 7. Temporal features (scalar, not per-window)
        names += [
            "hour_of_day",
            "day_of_week",       # 0=Monday … 6=Sunday
            "is_weekend",
            "seconds_since_last_tx",
        ]

        # 8. Merchant tier
        names.append("merchant_risk_tier_encoded")

        # 9. Recency ratios (1m/1h, 5m/1h)
        names += [
            "velocity_ratio_1m_1h",    # tx_count_1m / (tx_count_1h + 1)
            "velocity_ratio_5m_1h",    # tx_count_5m / (tx_count_1h + 1)
            "amt_ratio_1m_1h",         # amt_sum_1m  / (amt_sum_1h  + 1)
            "amt_ratio_5m_1h",         # amt_sum_5m  / (amt_sum_1h  + 1)
        ]

        return names

    def extract(
        self,
        entity_type: str,
        entity_id: str,
        timestamp: datetime,
    ) -> FeatureVector:
        """
        Compute and return the full feature vector for this entity at this timestamp.

        Parameters
        ----------
        entity_type : str
            One of "merchant_id", "card_bin", "device_id", "ip_address".
        entity_id : str
            The specific entity value.
        timestamp : datetime
            The point-in-time at which to compute features (used as as_of).
        """
        # Pull records for each window size
        windows: dict[str, list[TxRecord]] = {
            wk: self.engine.get_window(entity_type, entity_id, wk, as_of=timestamp)
            for wk in WINDOW_SIZES
        }

        values: list[float] = []

        # ------------------------------------------------------------------
        # 1. Velocity
        # ------------------------------------------------------------------
        tx_counts: dict[str, int] = {}
        for wk, recs in windows.items():
            c = len(recs)
            tx_counts[wk] = c
            values.append(float(c))

        # ------------------------------------------------------------------
        # 2. Amount stats
        # ------------------------------------------------------------------
        amt_sums: dict[str, float] = {}
        for wk, recs in windows.items():
            amounts = [r.amount_inr for r in recs]
            s = sum(amounts)
            amt_sums[wk] = s
            values += [
                s,
                _safe_mean(amounts),
                _safe_std(amounts),
                float(max(amounts)) if amounts else 0.0,
            ]

        # ------------------------------------------------------------------
        # 3. Counterparty diversity
        # ------------------------------------------------------------------
        for wk, recs in windows.items():
            values += [
                float(len({r.merchant_id for r in recs})),
                float(len({r.card_bin for r in recs if r.card_bin})),
                float(len({r.device_id for r in recs})),
                float(len({r.ip_address for r in recs})),
            ]

        # ------------------------------------------------------------------
        # 4. Failure rate
        # ------------------------------------------------------------------
        for wk, recs in windows.items():
            fail_count = sum(1 for r in recs if r.status == "failed")
            n = len(recs)
            values += [
                float(fail_count),
                float(fail_count / n) if n > 0 else 0.0,
            ]

        # ------------------------------------------------------------------
        # 5. Payment method fractions
        # ------------------------------------------------------------------
        for wk, recs in windows.items():
            n = len(recs)
            method_counts = {m: 0 for m in PAYMENT_METHODS}
            for r in recs:
                if r.payment_method in method_counts:
                    method_counts[r.payment_method] += 1
            for method in PAYMENT_METHODS:
                values.append(float(method_counts[method] / n) if n > 0 else 0.0)

        # ------------------------------------------------------------------
        # 6. Geo dispersion (unique /24 subnets)
        # ------------------------------------------------------------------
        for wk, recs in windows.items():
            values.append(float(_unique_slash24([r.ip_address for r in recs])))

        # ------------------------------------------------------------------
        # 7. Temporal features
        # ------------------------------------------------------------------
        values += [
            float(timestamp.hour),
            float(timestamp.weekday()),
            float(1 if timestamp.weekday() >= 5 else 0),
        ]
        # Seconds since last transaction in any window
        all_recs_1h = windows["1h"]
        if len(all_recs_1h) >= 2:
            sorted_recs = sorted(all_recs_1h, key=lambda r: r.timestamp)
            delta = (timestamp - sorted_recs[-1].timestamp).total_seconds()
            values.append(max(float(delta), 0.0))
        else:
            values.append(3600.0)  # No recent history → use window max

        # ------------------------------------------------------------------
        # 8. Merchant tier
        # ------------------------------------------------------------------
        # Use the most recent tier seen for this entity (or 0 if unknown)
        latest_rec = max(windows["1h"], key=lambda r: r.timestamp) if windows["1h"] else None
        tier = latest_rec.merchant_risk_tier if latest_rec else "low"
        values.append(TIER_ENCODING.get(tier, 0.0))

        # ------------------------------------------------------------------
        # 9. Recency ratios
        # ------------------------------------------------------------------
        tc_1m = float(tx_counts["1m"])
        tc_5m = float(tx_counts["5m"])
        tc_1h = float(tx_counts["1h"])
        as_1m = amt_sums["1m"]
        as_5m = amt_sums["5m"]
        as_1h = amt_sums["1h"]

        values += [
            tc_1m / (tc_1h + 1),
            tc_5m / (tc_1h + 1),
            as_1m / (as_1h + 1),
            as_5m / (as_1h + 1),
        ]

        assert len(values) == len(self.feature_names), (
            f"Feature count mismatch: {len(values)} values vs "
            f"{len(self.feature_names)} names"
        )

        return FeatureVector(
            entity_type=entity_type,
            entity_id=entity_id,
            timestamp=timestamp,
            feature_names=self.feature_names,
            values=np.array(values, dtype=np.float32),
        )
