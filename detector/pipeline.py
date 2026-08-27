"""
detector/pipeline.py — Full detector pipeline: ingest → window → features → score.

Orchestrates RollingWindowEngine + FeatureExtractor + SpikeDetector into a single
callable that consumes a stream of Transactions and emits DetectorOutputs.

Two modes:
1. Offline training/evaluation: process_batch() — for fitting and held-out eval
2. Online inference: process_one() — for the live dashboard and agent integration
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

import numpy as np

from data_gen.schema import DetectorOutput, Transaction
from detector.features import FeatureExtractor
from detector.model import SpikeDetector, build_labels
from detector.windows import RollingWindowEngine, ENTITY_TYPES


class DetectorPipeline:
    """
    End-to-end detector pipeline.

    Parameters
    ----------
    model_path : str | None
        If provided and the file exists, load a pre-fitted SpikeDetector from disk.
        Otherwise, the pipeline starts unfitted and must be trained with fit().
    score_threshold : float
        Minimum spike_score to emit a DetectorOutput (suppress noise).
    """

    def __init__(
        self,
        model_path: str | Path | None = None,
        score_threshold: float = 0.10,
    ):
        self.engine = RollingWindowEngine()
        self.extractor: FeatureExtractor | None = None  # built after first ingest
        self.detector: SpikeDetector | None = None
        self.score_threshold = score_threshold

        if model_path and Path(model_path).exists():
            self.detector = SpikeDetector.load(model_path)
            # Re-create the extractor so feature_names are consistent
            # (extractor is stateless — just name generation)
            self.extractor = FeatureExtractor(self.engine)

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(
        self,
        train_transactions: list[Transaction],
        random_state: int = 42,
    ) -> "DetectorPipeline":
        """
        Fit the detector on a list of training transactions.

        Steps
        -----
        1. Ingest all transactions into the rolling windows
        2. Extract a feature vector per unique (entity_type, entity_id) at each
           transaction's timestamp — one sample per transaction × entity dimension
        3. Build binary labels from ground-truth spike flags
        4. Fit the SpikeDetector ensemble
        """
        # Initialise extractor
        self.extractor = FeatureExtractor(self.engine)

        feature_vectors = []
        labels = []
        spike_entity_set: set[tuple[str, str]] = set()

        # Collect which (entity_type, entity_id) pairs are spike entities
        for tx in train_transactions:
            if tx.is_spike:
                for etype in ENTITY_TYPES:
                    eid = self._get_entity_id(tx, etype)
                    if eid:
                        spike_entity_set.add((etype, eid))

        # Ingest and extract features transaction by transaction
        # We sample one feature vector per transaction (from the merchant_id dimension)
        # to keep training tractable and tied to natural sampling frequency.
        for tx in train_transactions:
            self.engine.ingest(tx)
            etype = "merchant_id"
            eid = tx.merchant_id
            fv = self.extractor.extract(etype, eid, tx.timestamp)
            feature_vectors.append(fv)
            label = 1 if (etype, eid) in spike_entity_set and tx.is_spike else 0
            labels.append(label)

        X = np.stack([fv.values for fv in feature_vectors])
        y = np.array(labels, dtype=np.int32)

        self.detector = SpikeDetector(random_state=random_state)
        self.detector.fit(X, y, feature_names=feature_vectors[0].feature_names)

        return self

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def process_one(self, tx: Transaction) -> DetectorOutput | None:
        """
        Ingest a single transaction, compute features, and score.

        Returns a DetectorOutput if the spike_score >= score_threshold,
        else None (below-threshold transactions are not escalated).
        """
        if self.detector is None or not self.detector.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling process_one()")

        self.engine.ingest(tx)

        # Score the primary entity dimension (merchant_id)
        eid = tx.merchant_id
        fv = self.extractor.extract("merchant_id", eid, tx.timestamp)
        result = self.detector.predict_one(fv)

        if result["spike_score"] < self.score_threshold:
            return None

        return DetectorOutput(
            entity_type=result["entity_type"],
            entity_id=result["entity_id"],
            window_seconds=result["window_seconds"],
            spike_score=result["spike_score"],
            top_features=result["top_features"],
            timestamp=result["timestamp"],
            trigger_transaction=tx,
        )

    def process_batch(
        self,
        transactions: list[Transaction],
        emit_all: bool = False,
    ) -> list[DetectorOutput]:
        """
        Process a batch of transactions (offline evaluation mode).

        Parameters
        ----------
        transactions : list[Transaction], sorted by timestamp.
        emit_all : bool
            If True, emit a DetectorOutput for every transaction regardless
            of score_threshold. Used for evaluation to get full precision/recall.

        Returns
        -------
        list[DetectorOutput], one per transaction (if emit_all) or only those
        above score_threshold.
        """
        if self.detector is None or not self.detector.is_fitted:
            raise RuntimeError("Pipeline must be fitted before calling process_batch()")

        outputs = []
        for tx in transactions:
            self.engine.ingest(tx)
            fv = self.extractor.extract("merchant_id", tx.merchant_id, tx.timestamp)
            result = self.detector.predict_one(fv)

            if emit_all or result["spike_score"] >= self.score_threshold:
                outputs.append(DetectorOutput(
                    entity_type=result["entity_type"],
                    entity_id=result["entity_id"],
                    window_seconds=result["window_seconds"],
                    spike_score=result["spike_score"],
                    top_features=result["top_features"],
                    timestamp=result["timestamp"],
                    trigger_transaction=tx,
                ))

        return outputs

    def save_model(self, path: str | Path) -> None:
        if self.detector:
            self.detector.save(path)

    @staticmethod
    def _get_entity_id(tx: Transaction, etype: str) -> str | None:
        return {
            "merchant_id": tx.merchant_id,
            "card_bin": tx.card_bin,
            "device_id": tx.device_id,
            "ip_address": tx.ip_address,
        }.get(etype)
