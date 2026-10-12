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
        score_space (str): 'probability' (default) or 'logit'.
    """

    def __init__(self, threshold=0.5, score_space='probability'):
        from experiment.metrics import MetricsCalculator
        self.threshold = float(threshold)
        self.score_space = score_space or 'probability'
        self._core = MetricsCalculator(threshold=threshold)

    def compute_all(self, probabilities, labels, logits=None):
        """
        Compute all classification metrics.

        Args:
            probabilities (np.ndarray): Predicted probabilities (N,).
            labels (np.ndarray): Ground-truth binary labels (N,).
            logits (np.ndarray, optional): Unscaled logit predictions (N,).

        Returns:
            dict with all metric keys.
        """
        from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score
        from evaluation.generalization import compute_interpolated_eer, compute_binary_predictions

        probabilities = np.asarray(probabilities, dtype=np.float64).ravel()
        labels = np.asarray(labels, dtype=np.float64).ravel()
        if logits is not None:
            logits = np.asarray(logits, dtype=np.float64).ravel()

        pred_binary, _ = compute_binary_predictions(
            probabilities=probabilities,
            logits=logits,
            threshold=self.threshold,
            score_space=self.score_space
        )
        pred_binary = pred_binary.astype(np.float64)

        # Ranking score assignment
        if logits is not None:
            ranking_scores = logits
            ranking_score_space = 'logit'
        else:
            ranking_scores = probabilities
            ranking_score_space = 'probability'

        both = (len(np.unique(labels)) == 2)

        # Confusion matrix components
        cm = confusion_matrix(labels, pred_binary, labels=[0, 1])
        tn, fp, fn, tp = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]

        accuracy = float((tn + tp) / len(labels)) if len(labels) > 0 else 0.0
        precision = float(precision_score(labels, pred_binary, zero_division=0))
        recall = float(recall_score(labels, pred_binary, zero_division=0))
        f1 = float(f1_score(labels, pred_binary, zero_division=0))

        # Specificity and Sensitivity
        specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
        sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0

        # EER using linear interpolation
        eer = None
        if both:
            eer, _, _, _ = compute_interpolated_eer(labels, ranking_scores)

        # FAR and FRR
        far, frr = self._compute_far_frr(labels, pred_binary)

        # PR AUC & ROC AUC
        pr_auc = float(average_precision_score(labels, ranking_scores)) if both else None
        roc_auc = float(roc_auc_score(labels, ranking_scores)) if both else None
        roc_auc_logit = float(roc_auc_score(labels, logits)) if (both and logits is not None) else None
        roc_auc_probability = float(roc_auc_score(labels, probabilities)) if both else None

        # Per-class metrics
        per_class = self._compute_per_class(labels, pred_binary, probabilities)

        result = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'roc_auc_logit': roc_auc_logit,
            'roc_auc_probability': roc_auc_probability,
            'confusion_matrix': [[int(x) for x in row] for row in cm],
            'specificity': float(specificity),
            'sensitivity': float(sensitivity),
            'eer': eer,
            'far': float(far),
            'frr': float(frr),
            'pr_auc': pr_auc,
            'decision_threshold': float(self.threshold),
            'score_space': self.score_space,
            'ranking_score_space': ranking_score_space,
            'true_negatives': int(tn),
            'false_positives': int(fp),
            'false_negatives': int(fn),
            'true_positives': int(tp),
            **per_class,
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
