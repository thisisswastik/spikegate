"""
data_gen/stream.py — Stream replay utilities for the SpikeGate pipeline.

Provides:
- batch_to_jsonl(): serialize a list of Transactions to a JSONL file
- load_jsonl(): deserialize JSONL back to Transactions
- replay_stream(): async generator that replays transactions at controlled speed
"""
from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncIterator

from data_gen.schema import Transaction


def batch_to_jsonl(transactions: list[Transaction], output_path: str | Path) -> Path:
    """
    Serialize a list of Transaction objects to a JSONL file.

    Each line is a JSON object representing one transaction.
    Returns the path to the written file.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for tx in transactions:
            f.write(tx.model_dump_json() + "\n")

    return output_path


def load_jsonl(input_path: str | Path) -> list[Transaction]:
    """
    Load transactions from a JSONL file.

    Assumes the file was written by batch_to_jsonl().
    """
    input_path = Path(input_path)
    transactions = []
    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                transactions.append(Transaction(**data))
    return transactions


async def replay_stream(
    transactions: list[Transaction],
    speed: float = 1.0,
    max_gap_seconds: float = 5.0,
) -> AsyncIterator[Transaction]:
    """
    Async generator that replays a sorted list of transactions.

    Parameters
    ----------
    transactions : list[Transaction]
        Must be sorted by timestamp ascending.
    speed : float
        Replay speed multiplier. 1.0 = real-time, 10.0 = 10x faster.
    max_gap_seconds : float
        Maximum real-time wait between transactions (cap on slow periods).
        This prevents the generator from stalling for many seconds during quiet windows.

    Yields
    ------
    Transaction
        Each transaction at the appropriate simulated time.
    """
    if not transactions:
        return

    prev_sim_time = transactions[0].timestamp

    for tx in transactions:
        # Calculate simulated inter-event gap
        sim_gap = (tx.timestamp - prev_sim_time).total_seconds()
        real_gap = sim_gap / speed

        # Cap the wait to avoid stalls
        real_gap = min(real_gap, max_gap_seconds)

        if real_gap > 0:
            await asyncio.sleep(real_gap)

        prev_sim_time = tx.timestamp
        yield tx


def split_train_test(
    transactions: list[Transaction],
    test_fraction: float = 0.20,
) -> tuple[list[Transaction], list[Transaction]]:
    """
    Split transactions into train and test sets by time (not random shuffle).

    Uses a time-based split to prevent data leakage: train = first (1-test_fraction),
    test = last test_fraction of the simulation period.

    Returns
    -------
    train_transactions, test_transactions
    """
    if not transactions:
        return [], []

    # Transactions are sorted by timestamp
    split_idx = int(len(transactions) * (1 - test_fraction))
    return transactions[:split_idx], transactions[split_idx:]
