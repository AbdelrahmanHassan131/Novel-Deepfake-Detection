"""
Validator.

Runs model evaluation on a validation dataloader and returns a
structured ``ValidationResult``.

The Validator is completely model-agnostic: it delegates to the model's
``set_input`` / ``forward`` / ``get_loss`` interface that every
``BaseModel`` subclass already exposes.

Usage:
    validator = Validator()
    result = validator.validate(model, val_loader)
    print(result.accuracy, result.auc)
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ValidationResult:
    """Structured container for validation metrics."""
    loss: float = 0.0
    accuracy: float = 0.0
    auc: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    num_samples: int = 0
    predictions: Optional[np.ndarray] = field(default=None, repr=False)
    labels: Optional[np.ndarray] = field(default=None, repr=False)


class Validator:
    """
    Model-agnostic validation runner.

    Args:
        device: torch device to evaluate on.  If ``None``, uses the
                model's own ``device`` attribute.
    """

    def __init__(self, device=None):
        self.device = device

    @torch.no_grad()
    def validate(self, model, dataloader):
        """
        Run validation over the entire dataloader.

        This method:
            1. Puts the model in eval mode.
            2. Iterates the dataloader, calling ``model.set_input`` and
               ``model.forward`` on every batch.
            3. Collects losses, predictions, and labels.
            4. Computes aggregate metrics (accuracy, AUC, precision, recall).
            5. Returns a ``ValidationResult``.

        Args:
            model: A ``BaseModel`` subclass with ``set_input``,
                   ``forward``, ``get_loss``, ``output``, ``label``.
            dataloader: A PyTorch DataLoader.

        Returns:
            A ``ValidationResult`` instance.
        """
        model.eval()

        all_losses = []
        all_preds = []
        all_labels = []

        for batch in dataloader:
            model.set_input(batch)
            model.forward()

            # --- loss ---
            loss = model.get_loss()
            all_losses.append(loss.item())

            # --- predictions ---
            output = model.output
            if output.dim() > 1:
                output = output.squeeze(1)
            probs = torch.sigmoid(output)
            preds = (probs > 0.5).float()

            all_preds.append(probs.cpu().numpy())
            all_labels.append(model.label.cpu().numpy())

        # Aggregate
        all_preds = np.concatenate(all_preds)
        all_labels = np.concatenate(all_labels)
        num_samples = len(all_labels)

        avg_loss = float(np.mean(all_losses)) if all_losses else 0.0

        # Accuracy
        pred_binary = (all_preds > 0.5).astype(float)
        accuracy = float(np.mean(pred_binary == all_labels)) if num_samples > 0 else 0.0

        # AUC
        auc = _safe_auc(all_labels, all_preds)

        # Precision / Recall
        precision, recall = _precision_recall(all_labels, pred_binary)

        return ValidationResult(
            loss=avg_loss,
            accuracy=accuracy,
            auc=auc,
            precision=precision,
            recall=recall,
            num_samples=num_samples,
            predictions=all_preds,
            labels=all_labels,
        )


def _safe_auc(labels, preds):
    """Compute AUC, returning 0.0 if sklearn is unavailable or data is degenerate."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) < 2:
            return 0.0
        return float(roc_auc_score(labels, preds))
    except ImportError:
        return 0.0
    except Exception:
        return 0.0


def _precision_recall(labels, pred_binary):
    """Compute precision and recall without hard sklearn dependency."""
    tp = float(np.sum((pred_binary == 1) & (labels == 1)))
    fp = float(np.sum((pred_binary == 1) & (labels == 0)))
    fn = float(np.sum((pred_binary == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
