"""
MetricsCalculator — stateless, reusable metrics computation.

Receives numpy arrays of predictions and labels and computes all
classification metrics in a single call.  Used by:

    - Validator (during training validation)
    - Post-training evaluation scripts
    - Any future testing script

No state is stored — every call is independent and side-effect-free.

Usage::

    from Refactored.experiment.metrics import MetricsCalculator

    calc = MetricsCalculator()
    result = calc.compute(predictions, labels)
    print(result['accuracy'], result['f1'])
"""

import numpy as np


class MetricsCalculator:
    """
    Stateless binary classification metrics calculator.

    All public methods accept numpy arrays and return plain Python
    scalars or dicts.  No PyTorch dependency — works anywhere.

    Args:
        threshold (float): Decision threshold for binarising
            probability predictions.  Defaults to 0.5.
    """

    def __init__(self, threshold=0.5):
        self.threshold = threshold

    # ------------------------------------------------------------------
    # Main API
    # ------------------------------------------------------------------

    def compute(self, predictions, labels):
        """
        Compute all classification metrics.

        Args:
            predictions (np.ndarray): Predicted probabilities, shape ``(N,)``.
            labels (np.ndarray): Ground-truth binary labels, shape ``(N,)``.

        Returns:
            dict with keys:
                ``accuracy``, ``precision``, ``recall``, ``f1``,
                ``roc_auc``, ``confusion_matrix``.
        """
        predictions = np.asarray(predictions, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()

        pred_binary = (predictions >= self.threshold).astype(np.float64)

        accuracy = self._accuracy(pred_binary, labels)
        precision = self._precision(pred_binary, labels)
        recall = self._recall(pred_binary, labels)
        f1 = self._f1(precision, recall)
        roc_auc = self._roc_auc(labels, predictions)
        cm = self._confusion_matrix(pred_binary, labels)

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'roc_auc': roc_auc,
            'confusion_matrix': cm,
        }

    # ------------------------------------------------------------------
    # Individual metrics
    # ------------------------------------------------------------------

    @staticmethod
    def _accuracy(pred_binary, labels):
        """Fraction of correct predictions."""
        if len(labels) == 0:
            return 0.0
        return float(np.mean(pred_binary == labels))

    @staticmethod
    def _precision(pred_binary, labels):
        """TP / (TP + FP)."""
        tp = float(np.sum((pred_binary == 1) & (labels == 1)))
        fp = float(np.sum((pred_binary == 1) & (labels == 0)))
        return tp / (tp + fp) if (tp + fp) > 0 else 0.0

    @staticmethod
    def _recall(pred_binary, labels):
        """TP / (TP + FN)."""
        tp = float(np.sum((pred_binary == 1) & (labels == 1)))
        fn = float(np.sum((pred_binary == 0) & (labels == 1)))
        return tp / (tp + fn) if (tp + fn) > 0 else 0.0

    @staticmethod
    def _f1(precision, recall):
        """Harmonic mean of precision and recall."""
        if (precision + recall) == 0:
            return 0.0
        return 2.0 * precision * recall / (precision + recall)

    @staticmethod
    def _roc_auc(labels, predictions):
        """
        Compute ROC AUC.

        Returns 0.0 if sklearn is unavailable, if only one class is
        present, or on any computation error.
        """
        try:
            from sklearn.metrics import roc_auc_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(roc_auc_score(labels, predictions))
        except ImportError:
            return 0.0
        except Exception:
            return 0.0

    @staticmethod
    def _confusion_matrix(pred_binary, labels):
        """
        Compute a 2x2 confusion matrix as a nested list.

        Returns:
            ``[[TN, FP], [FN, TP]]`` as a list of lists of ints.
        """
        tp = int(np.sum((pred_binary == 1) & (labels == 1)))
        tn = int(np.sum((pred_binary == 0) & (labels == 0)))
        fp = int(np.sum((pred_binary == 1) & (labels == 0)))
        fn = int(np.sum((pred_binary == 0) & (labels == 1)))
        return [[tn, fp], [fn, tp]]
