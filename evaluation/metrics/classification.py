"""
Classification Metrics.

Extends the existing ``MetricsCalculator`` with the additional metrics
required by the evaluation pipeline (EER, FAR, FRR, Specificity,
Sensitivity, PR AUC, per-class metrics).

Reuses ``experiment.metrics.MetricsCalculator`` for the
core metrics and adds the evaluation-specific ones on top.

Usage::

    from evaluation.metrics import ClassificationMetrics

    metrics = ClassificationMetrics()
    result = metrics.compute_all(probabilities, labels)
"""

import numpy as np


class ClassificationMetrics:
    """
    Comprehensive binary classification metrics.

    Reuses the core MetricsCalculator for accuracy, precision, recall,
    F1, ROC AUC, and confusion matrix.  Adds EER, FAR, FRR,
    specificity, sensitivity, PR AUC, and per-class breakdowns.

    Args:
        threshold (float): Decision threshold for binarisation.
    """

    def __init__(self, threshold=0.5):
        from experiment.metrics import MetricsCalculator
        self._core = MetricsCalculator(threshold=threshold)
        self.threshold = threshold

    def compute_all(self, probabilities, labels):
        """
        Compute all classification metrics.

        Args:
            probabilities (np.ndarray): Predicted probabilities (N,).
            labels (np.ndarray): Ground-truth binary labels (N,).

        Returns:
            dict with all metric keys.
        """
        probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        pred_binary = (probabilities >= self.threshold).astype(np.float64)

        # Core metrics via existing MetricsCalculator
        core = self._core.compute(probabilities, labels)

        # Confusion matrix components
        cm = np.array(core['confusion_matrix'])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        # Specificity and Sensitivity
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # EER
        eer, eer_threshold = self._compute_eer(labels, probabilities)

        # FAR and FRR
        far, frr = self._compute_far_frr(labels, pred_binary)

        # PR AUC
        pr_auc = self._compute_pr_auc(labels, probabilities)

        # Per-class metrics
        per_class = self._compute_per_class(labels, pred_binary,
                                            probabilities)

        result = {
            # Core (from MetricsCalculator)
            'accuracy': float(core['accuracy']),
            'precision': float(core['precision']),
            'recall': float(core['recall']),
            'f1_score': float(core['f1']),
            'roc_auc': float(core['roc_auc']),
            'confusion_matrix': [[int(x) for x in row] for row in core['confusion_matrix']],
            # Extended
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'far': float(far),
            'frr': float(frr),
            'pr_auc': float(pr_auc),
            'decision_threshold': float(self.threshold),
            # Confusion matrix raw
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            # Per-class
            **per_class,
            # Dataset stats
            'total_samples': int(len(labels)),
            'num_positive': int((labels == 1).sum()),
            'num_negative': int((labels == 0).sum()),
        }

        return result

    # ------------------------------------------------------------------
    # Individual metric implementations
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_eer(labels, probabilities):
        """Compute Equal Error Rate."""
        try:
            from sklearn.metrics import roc_curve
            fpr, tpr, thresholds = roc_curve(labels, probabilities)
            fnr = 1 - tpr
            eer_idx = np.nanargmin(np.absolute(fnr - fpr))
            eer = float(fpr[eer_idx])
            eer_threshold = float(thresholds[eer_idx])
            return eer, eer_threshold
        except Exception:
            return 0.0, 0.5

    @staticmethod
    def _compute_far_frr(labels, pred_binary):
        """Compute False Accept Rate and False Reject Rate."""
        negatives = labels == 0
        positives = labels == 1

        far = (float(((pred_binary == 1) & (labels == 0)).sum()
                     / negatives.sum())
               if negatives.sum() > 0 else 0.0)

        frr = (float(((pred_binary == 0) & (labels == 1)).sum()
                     / positives.sum())
               if positives.sum() > 0 else 0.0)

        return far, frr

    @staticmethod
    def _compute_pr_auc(labels, probabilities):
        """Compute Precision-Recall AUC (Average Precision)."""
        try:
            from sklearn.metrics import average_precision_score
            if len(np.unique(labels)) < 2:
                return 0.0
            return float(average_precision_score(labels, probabilities))
        except Exception:
            return 0.0

    @staticmethod
    def _compute_per_class(labels, pred_binary, probabilities):
        """Compute per-class precision, recall, F1."""
        try:
            from sklearn.metrics import precision_recall_fscore_support
            prec, rec, f1, support = precision_recall_fscore_support(
                labels, pred_binary, zero_division=0
            )
            return {
                'precision_class_0': float(prec[0]) if len(prec) > 0 else 0.0,
                'precision_class_1': float(prec[1]) if len(prec) > 1 else 0.0,
                'recall_class_0': float(rec[0]) if len(rec) > 0 else 0.0,
                'recall_class_1': float(rec[1]) if len(rec) > 1 else 0.0,
                'f1_class_0': float(f1[0]) if len(f1) > 0 else 0.0,
                'f1_class_1': float(f1[1]) if len(f1) > 1 else 0.0,
                'support_class_0': int(support[0]) if len(support) > 0 else 0,
                'support_class_1': int(support[1]) if len(support) > 1 else 0,
            }
        except Exception:
            return {
                'precision_class_0': 0.0, 'precision_class_1': 0.0,
                'recall_class_0': 0.0, 'recall_class_1': 0.0,
                'f1_class_0': 0.0, 'f1_class_1': 0.0,
                'support_class_0': 0, 'support_class_1': 0,
            }
