"""
dashboard/stream_adapter.py — Adapts the data_gen stream for Streamlit rendering.

Provides a thread-safe queue-based adapter so the Streamlit main loop can
consume transactions from the generator without blocking the UI.
"""
from __future__ import annotations

import queue
import threading
import time
from datetime import datetime, timezone

from data_gen.generator import TransactionGenerator
from data_gen.schema import Transaction


class StreamAdapter:
    """
    Thread-safe adapter that runs the transaction generator in a background
    thread and exposes transactions via a queue for the Streamlit UI.

    Usage
    -----
    adapter = StreamAdapter(speed=10.0)
    adapter.start()
    # In Streamlit loop:
    tx = adapter.poll()  # Returns Transaction or None
    """

    def __init__(
        self,
        n_merchants: int = 30,
        seed: int = 42,
        simulation_hours: float = 2.0,
        speed: float = 10.0,  # 10x faster than real-time
    ):
        self.speed = speed
        self._queue: queue.Queue[Transaction] = queue.Queue(maxsize=500)
        self._stop_event = threading.Event()

        self.gen = TransactionGenerator(
            n_merchants=n_merchants,
            base_tps=2.0,
            spike_prob=0.03,
            simulation_hours=simulation_hours,
            seed=seed,
        )

    def start(self):
        """Start the background generator thread."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        """Signal the background thread to stop."""
        self._stop_event.set()

    def poll(self, timeout: float = 0.05) -> Transaction | None:
        """Return the next transaction or None if queue is empty."""
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def _run(self):
        """Generate transactions and feed them to the queue at the configured speed."""
        transactions, _ = self.gen.generate_batch()
        if not transactions:
            return

        prev_sim_time = transactions[0].timestamp

        for tx in transactions:
            if self._stop_event.is_set():
                break

            # Real-time gap between transactions
            sim_gap = (tx.timestamp - prev_sim_time).total_seconds()
            real_gap = sim_gap / self.speed
            real_gap = min(real_gap, 1.0)  # cap to 1s max

            if real_gap > 0:
                time.sleep(real_gap)

            prev_sim_time = tx.timestamp

            try:
                self._queue.put(tx, timeout=0.5)
            except queue.Full:
                pass  # Drop transaction if consumer is too slow
