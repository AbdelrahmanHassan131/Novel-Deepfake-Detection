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
    auc: Optional[float] = None
    balanced_accuracy: Optional[float] = None
    best_threshold: float = 0.5
    precision: float = 0.0
    recall: float = 0.0
    f1: float = 0.0
    num_samples: int = 0
    predictions: Optional[np.ndarray] = field(default=None, repr=False)
    labels: Optional[np.ndarray] = field(default=None, repr=False)
    indices: Optional[np.ndarray] = field(default=None, repr=False)


def sync_module_buffers(module):
    """Synchronize buffers (e.g. BatchNorm running stats) from rank 0 across all ranks."""
    import torch.distributed as dist
    if not dist.is_available() or not dist.is_initialized():
        return
    for buffer in module.buffers():
        dist.broadcast(buffer.data, src=0)


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
            2. Safely unwraps DDP models to evaluate locally without collective forward hooks,
               synchronizing buffers beforehand.
            3. Iterates the dataloader, calling ``model.set_input`` and
               ``model.forward`` on every batch.
            4. Collects losses, predictions, and labels.
            5. Aggregates sample-weighted loss and gathers predictions across ranks.
            6. Computes aggregate metrics: AUC, balanced accuracy, accuracy, F1,
               and development Youden J threshold.
            7. Restores model wrapper exception-safely and returns a ``ValidationResult``.

        Args:
            model: A ``BaseModel`` subclass with ``set_input``,
                   ``forward``, ``get_loss``, ``output``, ``label``.
            dataloader: A PyTorch DataLoader.

        Returns:
            A ``ValidationResult`` instance.
        """
        import torch.distributed as dist
        is_dist = dist.is_initialized() and dist.get_world_size() > 1

        # Check if model.model is wrapped in DDP
        ddp_wrapped = False
        orig_model_module = None
        if hasattr(model, 'model') and hasattr(torch.nn.parallel, 'DistributedDataParallel'):
            if isinstance(model.model, torch.nn.parallel.DistributedDataParallel):
                ddp_wrapped = True
                orig_model_module = model.model
                # Explicit buffer synchronization from rank 0 before unpadding/unwrapping
                sync_module_buffers(orig_model_module.module)
                model.model = orig_model_module.module

        orig_training_mode = getattr(model, 'is_train', getattr(model, 'isTrain', True))
        model.eval()

        # Track dataset evaluation indices for this dataloader / rank
        sampler = getattr(dataloader, 'sampler', None)
        if sampler is not None and hasattr(sampler, 'indices'):
            rank_indices = list(sampler.indices)
        else:
            dataset_len = len(dataloader.dataset) if hasattr(dataloader, 'dataset') else 0
            rank_indices = list(range(dataset_len))

        all_preds = []
        all_labels = []
        all_indices = []
        local_loss_sum = 0.0
        local_sample_count = 0
        cursor = 0

        try:
            for batch in dataloader:
                model.set_input(batch)
                model.forward()

                batch_size = model.label.shape[0] if hasattr(model, 'label') and model.label is not None else 0
                if batch_size > 0:
                    loss = model.get_loss()
                    local_loss_sum += float(loss.item()) * batch_size
                    local_sample_count += batch_size

                    output = model.output
                    if output.dim() > 1:
                        output = output.squeeze(1)
                    probs = torch.sigmoid(output)

                    all_preds.append(probs.detach().cpu().numpy())
                    all_labels.append(model.label.detach().cpu().numpy())
                    batch_idx = rank_indices[cursor : cursor + batch_size]
                    all_indices.extend(batch_idx)
                    cursor += batch_size
        finally:
            # Exception-safe restoration of DDP wrapper and original training mode
            if ddp_wrapped:
                model.model = orig_model_module
            if orig_training_mode:
                model.train()

        # Aggregate local arrays
        local_preds = np.concatenate(all_preds) if len(all_preds) > 0 else np.array([], dtype=np.float32)
        local_labels = np.concatenate(all_labels) if len(all_labels) > 0 else np.array([], dtype=np.float32)
        local_indices = np.array(all_indices, dtype=np.int64) if len(all_indices) > 0 else np.array([], dtype=np.int64)

        if is_dist:
            world_size = dist.get_world_size()
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

            # Gather predictions, labels, and dataset indices across all ranks
            gathered_preds = [None for _ in range(world_size)]
            gathered_labels = [None for _ in range(world_size)]
            gathered_indices = [None for _ in range(world_size)]
            dist.all_gather_object(gathered_preds, local_preds)
            dist.all_gather_object(gathered_labels, local_labels)
            dist.all_gather_object(gathered_indices, local_indices)

            valid_preds = [p for p in gathered_preds if len(p) > 0]
            all_preds = np.concatenate(valid_preds) if valid_preds else np.array([], dtype=np.float32)
            valid_labels = [l for l in gathered_labels if len(l) > 0]
            all_labels = np.concatenate(valid_labels) if valid_labels else np.array([], dtype=np.float32)
            valid_indices = [idx for idx in gathered_indices if len(idx) > 0]
            all_indices = np.concatenate(valid_indices) if valid_indices else np.array([], dtype=np.int64)

            # Re-sort predictions, labels, and indices by canonical dataset index
            if len(all_indices) > 0:
                sort_order = np.argsort(all_indices)
                all_preds = all_preds[sort_order]
                all_labels = all_labels[sort_order]
                all_indices = all_indices[sort_order]

            # Sample-weighted loss reduction: sum(loss * samples) / sum(samples)
            stats = torch.tensor([local_loss_sum, float(local_sample_count)], dtype=torch.float64, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            global_loss_sum = stats[0].item()
            global_sample_count = int(stats[1].item())
            avg_loss = float(global_loss_sum / global_sample_count) if global_sample_count > 0 else 0.0
            num_samples = global_sample_count
        else:
            all_preds = local_preds
            all_labels = local_labels
            all_indices = local_indices
            avg_loss = float(local_loss_sum / local_sample_count) if local_sample_count > 0 else 0.0
            num_samples = local_sample_count

        num_samples = len(all_labels)
        unique_labels = np.unique(all_labels) if num_samples > 0 else np.array([])
        both_classes = len(unique_labels) == 2

        # Compute development threshold via Youden's J if both classes exist
        best_threshold = 0.5
        auc = None
        balanced_accuracy = None

        if both_classes and num_samples > 0:
            try:
                from sklearn.metrics import roc_curve, roc_auc_score
                fpr, tpr, thresholds = roc_curve(all_labels, all_preds)
                eligible = np.flatnonzero(np.isfinite(thresholds) & (thresholds <= 1.0) & (thresholds >= 0.0))
                if len(eligible) > 0:
                    best_idx = eligible[np.argmax((tpr - fpr)[eligible])]
                    best_threshold = float(thresholds[best_idx])
                auc = float(roc_auc_score(all_labels, all_preds))
            except Exception:
                auc = _safe_auc(all_labels, all_preds)
        else:
            auc = None

        # Try using MetricsCalculator from experiment package or fallback
        pred_binary = (all_preds >= 0.5).astype(float) if num_samples > 0 else np.array([])
        accuracy = float(np.mean(pred_binary == all_labels)) if num_samples > 0 else 0.0
        precision, recall = _precision_recall(all_labels, pred_binary)
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        if both_classes:
            r0 = float(np.sum((pred_binary == 0) & (all_labels == 0))) / max(float(np.sum(all_labels == 0)), 1e-8)
            r1 = recall
            balanced_accuracy = float(0.5 * (r0 + r1))

        return ValidationResult(
            loss=avg_loss,
            accuracy=accuracy,
            auc=auc,
            balanced_accuracy=balanced_accuracy,
            best_threshold=best_threshold,
            precision=precision,
            recall=recall,
            f1=f1,
            num_samples=num_samples,
            predictions=all_preds,
            labels=all_labels,
            indices=all_indices,
        )


def _safe_auc(labels, preds):
    """Compute AUC, returning None if sklearn is unavailable or data is degenerate/single-class."""
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(labels)) < 2:
            return None
        return float(roc_auc_score(labels, preds))
    except Exception:
        return None


def _precision_recall(labels, pred_binary):
    """Compute precision and recall without hard sklearn dependency."""
    tp = float(np.sum((pred_binary == 1) & (labels == 1)))
    fp = float(np.sum((pred_binary == 1) & (labels == 0)))
    fn = float(np.sum((pred_binary == 0) & (labels == 1)))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return precision, recall
