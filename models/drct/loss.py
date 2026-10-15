"""DRCT Loss Functions.

Implements the official DRCT margin-based contrastive loss and classification objective,
following upstream CombinedLoss and pytorch_metric_learning.losses.ContrastiveLoss.
"""
from typing import Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginContrastiveLoss(nn.Module):
    """Margin-based Contrastive Loss matching upstream DRCT and pytorch_metric_learning.losses.ContrastiveLoss.
    
    Upstream specification:
    1. Embeddings are L2-normalized onto the unit hypersphere: z = F.normalize(e, p=2, dim=1).
    2. Pairwise Euclidean distances: D_ij = ||z_i - z_j||_2.
    3. Positive pairs (y_i == y_j):
           loss_pos = max(0, D_ij - pos_margin)  [unsquared]
    4. Negative pairs (y_i != y_j):
           loss_neg = max(0, neg_margin - D_ij)  [unsquared]
    5. Reduction (AvgNonZeroReducer):
           pos_loss = mean of positive violations > 0 (or 0 if none)
           neg_loss = mean of negative violations > 0 (or 0 if none)
           total_loss = pos_loss + neg_loss
    
    All distance and reduction calculations are performed in float32 for AMP stability.
    Edge cases (singleton batch, homogeneous batch, zero violations) maintain graph connectivity
    with zero tensors.
    """
    def __init__(self, pos_margin: float = 0.0, neg_margin: float = 1.0):
        super().__init__()
        self.pos_margin = float(pos_margin)
        self.neg_margin = float(neg_margin)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        b = embeddings.shape[0]
        if b <= 1:
            return torch.sum(embeddings * 0.0)

        # Force FP32 for distance calculation to ensure numerical and AMP stability
        emb_fp32 = embeddings.float()
        labels_flat = labels.view(-1)

        # 1. L2 normalize embeddings
        emb_norm = F.normalize(emb_fp32, p=2, dim=1)

        # 2. Extract upper-triangular index pairs (i < j)
        triu_indices = torch.triu_indices(b, b, offset=1, device=embeddings.device)
        i_idx, j_idx = triu_indices[0], triu_indices[1]

        # Direct differences preserve exact zeros even for batches >25, where
        # cdist's default matrix-multiplication algorithm suffers cancellation.
        dists = torch.linalg.vector_norm(emb_norm[i_idx] - emb_norm[j_idx], dim=1)

        # 4. Partition pairs
        same_label = (labels_flat[i_idx] == labels_flat[j_idx])
        diff_label = ~same_label

        zero_anchor = torch.sum(embeddings * 0.0)

        # Positive pairs loss
        if same_label.any():
            pos_dists = dists[same_label]
            pos_violations = torch.clamp(pos_dists - self.pos_margin, min=0.0)
            pos_nz = pos_violations > 0
            if pos_nz.any():
                pos_loss = pos_violations[pos_nz].mean()
            else:
                pos_loss = zero_anchor
        else:
            pos_loss = zero_anchor

        # Negative pairs loss
        if diff_label.any():
            neg_dists = dists[diff_label]
            neg_violations = torch.clamp(self.neg_margin - neg_dists, min=0.0)
            neg_nz = neg_violations > 0
            if neg_nz.any():
                neg_loss = neg_violations[neg_nz].mean()
            else:
                neg_loss = zero_anchor
        else:
            neg_loss = zero_anchor

        return pos_loss + neg_loss


class DRCTCombinedLoss(nn.Module):
    """Combined Classification + Margin-based Contrastive Loss for DRCT-128 Adaptation.
    
    Total Loss = (1 - alpha) * BCEWithLogitsLoss + alpha * ContrastiveLoss
    
    Default parameters:
        alpha = 0.3
        pos_margin = 0.0
        neg_margin = 1.0
    """
    def __init__(
        self,
        alpha: float = 0.3,
        pos_margin: float = 0.0,
        neg_margin: float = 1.0,
        enable_contrastive: bool = True,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.enable_contrastive = bool(enable_contrastive)
        self.cls_criterion = nn.BCEWithLogitsLoss()
        self.contrastive_criterion = MarginContrastiveLoss(
            pos_margin=pos_margin, neg_margin=neg_margin
        )

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        embeddings: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute training losses.
        
        Args:
            logits: Predicted raw logits [B, 1] or [B]
            labels: Ground truth binary labels [B] or [B, 1]
            embeddings: Optional 128D embeddings [B, 128] for contrastive loss
            
        Returns:
            (total_loss, cls_loss, cont_loss)
        """
        logits_flat = logits.view(-1)
        labels_flat = labels.view(-1).float()

        cls_loss = self.cls_criterion(logits_flat, labels_flat)

        if self.enable_contrastive and self.alpha > 0.0 and embeddings is not None:
            cont_loss = self.contrastive_criterion(embeddings, labels_flat)
            total_loss = (1.0 - self.alpha) * cls_loss + self.alpha * cont_loss
        else:
            cont_loss = torch.tensor(0.0, device=logits.device, dtype=torch.float32)
            total_loss = cls_loss

        return total_loss, cls_loss, cont_loss
