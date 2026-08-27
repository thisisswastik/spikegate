"""
detector/windows.py — Epoch-bucket rolling-window engine.

Uses fixed-size time buckets (1-min, 5-min, 1-hr) rather than a true sliding window.
This is cheaper to compute and produces stable, reproducible features per entity.

For each (entity_type, entity_id, bucket_size) triplet we maintain a deque of
Transaction-like records for the last N complete + current buckets.  On each
ingest call we evict any record older than the window horizon.

Design decisions:
- We keep raw transaction records in memory, not pre-aggregated counters, so
  the feature extractor can compute any statistic it needs.
- For the 1-hr window we only retain up to 3 600 records per entity to bound
  memory (extremely high-velocity entities are capped; the feature extractor
  sees a saturated count, which is already a spike signal).
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Deque

from data_gen.schema import Transaction

# Window sizes in seconds
WINDOW_SIZES = {
    "1m": 60,
    "5m": 300,
    "1h": 3600,
}

# Entity dimension keys
ENTITY_TYPES = ["merchant_id", "card_bin", "device_id", "ip_address"]

# Cap on records stored per (entity, window) to bound memory
MAX_RECORDS_PER_WINDOW = 3_600


@dataclass
class TxRecord:
    """Minimal representation of a transaction stored inside the window."""
    timestamp: datetime
    amount_inr: float
    status: str          # "success" | "failed" | "pending"
    payment_method: str
    merchant_id: str
    card_bin: str | None
    device_id: str
    ip_address: str
    merchant_risk_tier: str


def _tx_to_record(tx: Transaction) -> TxRecord:
    return TxRecord(
        timestamp=tx.timestamp,
        amount_inr=tx.amount_inr,
        status=tx.status,
        payment_method=tx.payment_method,
        merchant_id=tx.merchant_id,
        card_bin=tx.card_bin,
        device_id=tx.device_id,
        ip_address=tx.ip_address,
        merchant_risk_tier=tx.merchant_risk_tier,
    )


class RollingWindowEngine:
    """
    Maintains per-entity rolling windows for all entity types and window sizes.

    Usage
    -----
    engine = RollingWindowEngine()
    engine.ingest(tx)                   # update windows
    records = engine.get_window(entity_type, entity_id, window_key)
    """

    def __init__(self):
        # {(entity_type, entity_id, window_key): deque[TxRecord]}
        self._windows: dict[tuple[str, str, str], Deque[TxRecord]] = defaultdict(deque)

    def ingest(self, tx: Transaction) -> None:
        """Ingest a transaction, updating all relevant entity windows."""
        record = _tx_to_record(tx)
        now = tx.timestamp

        # The entity values present in this transaction
        entity_values: dict[str, str | None] = {
            "merchant_id": tx.merchant_id,
            "card_bin": tx.card_bin,
            "device_id": tx.device_id,
            "ip_address": tx.ip_address,
        }

        for etype, eid in entity_values.items():
            if eid is None:
                continue
            for wkey, wsec in WINDOW_SIZES.items():
                key = (etype, eid, wkey)
                dq = self._windows[key]
                # Evict stale records
                horizon = now - timedelta(seconds=wsec)
                while dq and dq[0].timestamp < horizon:
                    dq.popleft()
                # Append (with memory cap)
                if len(dq) < MAX_RECORDS_PER_WINDOW:
                    dq.append(record)

    def get_window(
        self,
        entity_type: str,
        entity_id: str,
        window_key: str,
        as_of: datetime | None = None,
    ) -> list[TxRecord]:
        """
        Return all records in the rolling window for a given entity.

        Parameters
        ----------
        entity_type : str
            One of "merchant_id", "card_bin", "device_id", "ip_address".
        entity_id : str
            The specific entity value.
        window_key : str
            One of "1m", "5m", "1h".
        as_of : datetime | None
            If provided, further restrict to records with timestamp >= as_of - window_size.
            (Useful for offline replay where 'now' differs from the latest record time.)

        Returns
        -------
        list[TxRecord]
            Records in the window, oldest first.
        """
        key = (entity_type, entity_id, window_key)
        dq = self._windows.get(key, deque())
        records = list(dq)

        if as_of is not None:
            wsec = WINDOW_SIZES[window_key]
            horizon = as_of - timedelta(seconds=wsec)
            records = [r for r in records if r.timestamp >= horizon]

        return records

    def entity_count(self) -> int:
        """Return the number of distinct (entity_type, entity_id) pairs tracked."""
        pairs = {(k[0], k[1]) for k in self._windows}
        return len(pairs)

    def clear(self) -> None:
        """Reset all windows (useful between test runs)."""
        self._windows.clear()
