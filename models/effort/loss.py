"""Loss function for EFFORT_128 combining smoothed BCE, optional contrastive loss, and SVD orthogonality."""
from typing import Optional, Tuple
import torch
import torch.nn as nn

from models.drct.loss import MarginContrastiveLoss


class EFFORTCombinedLoss(nn.Module):
    """Multi-term generalization objective for EFFORT_128.

    Total Training Loss:
        (1 - alpha) * BCEWithLogitsLoss(logits, y_smooth)
        + alpha * MarginContrastiveLoss(embeddings, y_hard)
        + ortho_weight * L_orthogonality

    Validation uses pure hard-label BCEWithLogitsLoss for calibrated monitoring.
    """

    def __init__(
        self,
        label_smoothing: float = 0.05,
        contrastive_alpha: float = 0.0,
        pos_margin: float = 0.0,
        neg_margin: float = 1.0,
        ortho_weight: float = 0.1,
    ):
        super().__init__()
        if not (0.0 <= float(label_smoothing) < 0.5):
            raise ValueError(f"label_smoothing must be in [0.0, 0.5), got {label_smoothing}")
        self.label_smoothing = float(label_smoothing)
        self.contrastive_alpha = float(contrastive_alpha)
        self.ortho_weight = float(ortho_weight)
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.contrastive_criterion = MarginContrastiveLoss(pos_margin=pos_margin, neg_margin=neg_margin)

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
        ortho_loss: Optional[torch.Tensor] = None,
        is_train: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        logits_flat = logits.view(-1)
        labels_hard = labels.view(-1).float()

        if is_train and self.label_smoothing > 0.0:
            s = self.label_smoothing
            labels_cls = labels_hard * (1.0 - 2.0 * s) + s
        else:
            labels_cls = labels_hard

        cls_loss = self.cls_criterion(logits_flat, labels_cls)

        if not is_train:
            zero = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            return cls_loss, cls_loss, zero, zero

        if self.contrastive_alpha > 0.0 and embeddings is not None:
            cont_loss = self.contrastive_criterion(embeddings, labels_hard)
            base_loss = (1.0 - self.contrastive_alpha) * cls_loss + self.contrastive_alpha * cont_loss
        else:
            cont_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            base_loss = cls_loss

        if self.ortho_weight > 0.0 and ortho_loss is not None:
            o_loss = ortho_loss.to(device=logits.device, dtype=torch.float32)
            total_loss = base_loss + self.ortho_weight * o_loss
        else:
            o_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            total_loss = base_loss

        return total_loss, cls_loss, cont_loss, o_loss
