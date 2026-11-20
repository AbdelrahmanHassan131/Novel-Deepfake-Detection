"""EFFORT_128: Multi-Layer + Orthogonal SVD CLIP Adaptation for Cross-Domain Deepfake Detection."""

from models.effort.svd_linear import (
    SVDResidualLinear,
    apply_svd_residual_adaptation,
    collect_orthogonality_loss,
)
from models.effort.effort_model import EFFORT128Model
from models.effort.loss import EFFORTCombinedLoss
from models.effort.trainer import EFFORT_128Trainer

__all__ = [
    'SVDResidualLinear',
    'apply_svd_residual_adaptation',
    'collect_orthogonality_loss',
    'EFFORT128Model',
    'EFFORTCombinedLoss',
    'EFFORT_128Trainer',
]
