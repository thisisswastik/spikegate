"""
detector/model.py — SpikeDetector: XGBoost + Isolation Forest ensemble.

Architecture
------------
Primary  : XGBoostClassifier trained on windowed feature vectors
           Label = 1 if the entity's current 5-min window contains a spike tx
Secondary: IsolationForest for anomaly scoring on unseen patterns
Ensemble : spike_score = 0.7 * xgb_proba + 0.3 * iso_score

Explainability
--------------
Uses SHAP TreeExplainer on the XGBoost model to get per-feature contributions.
Top-5 features by |SHAP| value are returned with each prediction.

Output contract
---------------
{
  entity_type, entity_id, window_seconds, spike_score (float in [0,1]),
  top_features: [{name, value, contribution}],
  timestamp
}
"""
from __future__ import annotations

import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import shap
import xgboost as xgb
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from detector.features import FeatureVector


# ---------------------------------------------------------------------------
# Training label builder
# ---------------------------------------------------------------------------

def build_labels(
    feature_vectors: list[FeatureVector],
    spike_entity_windows: set[tuple[str, str, str]],  # (entity_type, entity_id, window_key)
) -> np.ndarray:
    """
    Build binary labels for a list of feature vectors.

    A feature vector is labeled 1 (spike) if its (entity_type, entity_id)
    is in the spike_entity_windows set for the "5m" window.

    Parameters
    ----------
    feature_vectors : list[FeatureVector]
    spike_entity_windows : set of (entity_type, entity_id, "5m") tuples
        Pre-built from ground-truth transaction labels.

    Returns
    -------
    np.ndarray of shape (n,) with 0/1 labels.
    """
    labels = []
    for fv in feature_vectors:
        key = (fv.entity_type, fv.entity_id, "5m")
        labels.append(1 if key in spike_entity_windows else 0)
    return np.array(labels, dtype=np.int32)


# ---------------------------------------------------------------------------
# SpikeDetector
# ---------------------------------------------------------------------------

class SpikeDetector:
    """
    Ensemble spike detector: XGBoost + IsolationForest.

    Usage
    -----
    detector = SpikeDetector()
    detector.fit(X_train, y_train)
    result = detector.predict_one(feature_vector)
    """

    XGB_WEIGHT = 0.70
    ISO_WEIGHT = 0.30

    def __init__(
        self,
        xgb_params: dict[str, Any] | None = None,
        iso_params: dict[str, Any] | None = None,
        random_state: int = 42,
    ):
        default_xgb = {
            "n_estimators": 200,
            "max_depth": 6,
            "learning_rate": 0.05,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "min_child_weight": 3,
            "eval_metric": "logloss",
            "random_state": random_state,
            "n_jobs": -1,
        }
        default_iso = {
            "n_estimators": 100,
            "contamination": 0.04,
            "random_state": random_state,
            "n_jobs": -1,
        }
        self.xgb_params = {**default_xgb, **(xgb_params or {})}
        self.iso_params = {**default_iso, **(iso_params or {})}

        self.xgb_model: xgb.XGBClassifier | None = None
        self.iso_model: IsolationForest | None = None
        self.scaler: StandardScaler | None = None
        self.explainer: shap.TreeExplainer | None = None
        self.feature_names: list[str] = []
        self.is_fitted = False

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray, feature_names: list[str]) -> "SpikeDetector":
        """
        Fit the ensemble on a training set.

        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
        y : np.ndarray, shape (n_samples,), binary labels
        feature_names : list[str], length n_features
        """
        self.feature_names = feature_names

        # Scale features for IsolationForest (XGBoost doesn't need scaling)
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Dynamic scale_pos_weight to handle class imbalance
        n_pos = int(np.sum(y))
        n_neg = len(y) - n_pos
        scale_pos = (n_neg / max(n_pos, 1)) if n_pos > 0 else 1.0

        xgb_params = {**self.xgb_params, "scale_pos_weight": scale_pos}
        if "use_label_encoder" in xgb_params:
            del xgb_params["use_label_encoder"]

        # XGBoost
        self.xgb_model = xgb.XGBClassifier(**xgb_params)
        self.xgb_model.fit(X, y)

        # IsolationForest (train only on normal samples for better anomaly detection)
        normal_mask = (y == 0)
        X_normal = X_scaled[normal_mask] if normal_mask.sum() > 10 else X_scaled
        self.iso_model = IsolationForest(**self.iso_params)
        self.iso_model.fit(X_normal)

        # SHAP explainer (TreeExplainer is exact and fast for XGBoost)
        self.explainer = shap.TreeExplainer(self.xgb_model)

        self.is_fitted = True
        return self

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _iso_score_to_proba(self, X_scaled: np.ndarray) -> np.ndarray:
        """
        Convert IsolationForest decision_function scores to [0, 1] anomaly probabilities.

        decision_function returns negative anomaly scores (more negative = more anomalous).
        We negate and sigmoid-normalise.
        """
        raw = self.iso_model.decision_function(X_scaled)
        # Negate so higher = more anomalous
        anomaly = -raw
        # Sigmoid normalisation
        return 1.0 / (1.0 + np.exp(-anomaly * 3.0))

    def predict_proba_batch(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble spike scores for a batch.

        Returns
        -------
        np.ndarray of shape (n,), values in [0, 1].
        """
        if not self.is_fitted:
            raise RuntimeError("SpikeDetector must be fitted before calling predict_proba_batch")

        X_scaled = self.scaler.transform(X)
        xgb_proba = self.xgb_model.predict_proba(X)[:, 1]
        iso_proba = self._iso_score_to_proba(X_scaled)
        return self.XGB_WEIGHT * xgb_proba + self.ISO_WEIGHT * iso_proba

    def predict_one(self, fv: FeatureVector, compute_explanation: bool = True) -> dict:
        """
        Score a single FeatureVector and return the full output contract.

        Returns
        -------
        dict with keys: entity_type, entity_id, window_seconds, spike_score,
                        top_features, timestamp
        """
        X = fv.values.reshape(1, -1)
        spike_score = float(self.predict_proba_batch(X)[0])
        # Clip to valid range
        spike_score = float(np.clip(spike_score, 0.0, 1.0))

        top_features = []
        if compute_explanation:
            # SHAP is computed when spike_score >= 0.10 (anomalous/spike)
            if spike_score >= 0.10 and self.explainer is not None:
                shap_values = self.explainer.shap_values(X)
                if isinstance(shap_values, list):
                    shap_vals = shap_values[1][0]
                else:
                    shap_vals = shap_values[0]

                feature_contributions = list(zip(self.feature_names, fv.values.tolist(), shap_vals.tolist()))
                top_features = [
                    {"name": name, "value": float(val), "contribution": float(contrib)}
                    for name, val, contrib in sorted(
                        feature_contributions,
                        key=lambda x: abs(x[2]),
                        reverse=True,
                    )[:5]
                ]
            else:
                top_features = [
                    {"name": self.feature_names[i], "value": float(fv.values[i]), "contribution": 0.0}
                    for i in range(min(5, len(self.feature_names)))
                ]

        return {
            "entity_type": fv.entity_type,
            "entity_id": fv.entity_id,
            "window_seconds": 300,   # primary window: 5-min
            "spike_score": spike_score,
            "top_features": top_features,
            "timestamp": fv.timestamp,
        }

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """Serialize the fitted model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "SpikeDetector":
        """Deserialize a fitted model from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)
