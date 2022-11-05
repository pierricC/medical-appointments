"""Fonctions related to metrics."""

from typing import Dict, Tuple

import numpy as np
from sklearn.metrics import matthews_corrcoef, roc_auc_score


def proba_to_binary(probs: np.ndarray, threshold: float) -> np.ndarray:
    """
    Convert positive probability to binary.

    Class 1 if proba > threshold.
    """
    binary = []
    for value in probs:
        if value > threshold:
            binary.append(1)
        else:
            binary.append(0)
    return np.array(binary)


def get_max_metrics(
    y_test: np.ndarray, probs: np.ndarray, thresholds: np.ndarray
) -> Tuple[Dict[float, float], Dict[float, float]]:
    """Compute the auc and mcc for each threshold between 0 and 1."""
    mccs = {}
    aucs = {}
    for threshold in thresholds:
        preds = proba_to_binary(probs, threshold)
        aucs[threshold] = roc_auc_score(y_test, preds)
        mccs[threshold] = matthews_corrcoef(y_test, preds)
    return aucs, mccs
