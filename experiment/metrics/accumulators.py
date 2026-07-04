"""
MetricAccumulator — collects predictions and labels across batches
and computes aggregate metrics at the end.

Designed to be used inside a training or validation loop:

    1.  Create an accumulator at the start of the epoch / eval pass.
    2.  Call ``update()`` after each batch.
    3.  Call ``compute()`` at the end to get final metrics.

Usage::

    acc = MetricAccumulator()
    for batch in dataloader:
        preds, labels = model(batch)
        acc.update(preds, labels)
    metrics = acc.compute()

The accumulator delegates final computation to
:class:`MetricsCalculator` so metrics are computed in exactly one
place.
"""

import numpy as np
from .metrics import MetricsCalculator


class MetricAccumulator:
    """
    Batch-wise accumulator that delegates final computation to
    :class:`MetricsCalculator`.

    Args:
        threshold (float): Decision threshold passed to
            ``MetricsCalculator``.  Defaults to 0.5.
    """

    def __init__(self, threshold=0.5):
        self._calculator = MetricsCalculator(threshold=threshold)
        self._predictions = []
        self._labels = []
        self._losses = []

    # ------------------------------------------------------------------
    # Collection API
    # ------------------------------------------------------------------

    def update(self, predictions, labels, loss=None):
        """
        Append one batch of predictions and labels.

        Args:
            predictions: Array-like of predicted probabilities.
            labels: Array-like of ground-truth labels.
            loss (float, optional): Batch loss value.
        """
        self._predictions.append(np.asarray(predictions, dtype=np.float64).ravel())
        self._labels.append(np.asarray(labels, dtype=np.float64).ravel())
        if loss is not None:
            self._losses.append(float(loss))

    # ------------------------------------------------------------------
    # Computation API
    # ------------------------------------------------------------------

    def compute(self):
        """
        Compute aggregate metrics across all accumulated batches.

        Returns:
            dict with keys:
                ``accuracy``, ``precision``, ``recall``, ``f1``,
                ``roc_auc``, ``confusion_matrix``, ``avg_loss``,
                ``num_samples``.
        """
        if not self._predictions:
            return self._empty_result()

        all_preds = np.concatenate(self._predictions)
        all_labels = np.concatenate(self._labels)

        result = self._calculator.compute(all_preds, all_labels)

        result['avg_loss'] = (
            float(np.mean(self._losses)) if self._losses else 0.0
        )
        result['num_samples'] = len(all_labels)

        return result

    def reset(self):
        """Clear all accumulated data."""
        self._predictions.clear()
        self._labels.clear()
        self._losses.clear()

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def predictions(self):
        """Concatenated predictions as a numpy array, or empty array."""
        if not self._predictions:
            return np.array([], dtype=np.float64)
        return np.concatenate(self._predictions)

    @property
    def labels(self):
        """Concatenated labels as a numpy array, or empty array."""
        if not self._labels:
            return np.array([], dtype=np.float64)
        return np.concatenate(self._labels)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    @staticmethod
    def _empty_result():
        return {
            'accuracy': 0.0,
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0,
            'roc_auc': 0.0,
            'confusion_matrix': [[0, 0], [0, 0]],
            'avg_loss': 0.0,
            'num_samples': 0,
        }
