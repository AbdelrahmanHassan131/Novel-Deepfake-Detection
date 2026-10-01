"""Adapter placeholder for Effort baseline (arXiv:2411.15633).

Paper: EFFORT: Efficient and Flexible Forensics with Orthogonal Adaptation.
Status: PENDING_VERIFICATION.
Official release / checkpoint weights are currently pending verification.
Marked pending per handoff instructions rather than pretending integration is complete.
"""
from typing import Dict, Any, Optional
from .base_adapter import BaseBaselineAdapter


class EffortAdapter(BaseBaselineAdapter):
    """Adapter for EFFORT: Efficient and Flexible Forensics with Orthogonal Adaptation."""

    @property
    def name(self) -> str:
        return "Effort"

    @property
    def paper_citation(self) -> str:
        return (
            "EFFORT: Efficient and Flexible Forensics with Orthogonal Adaptation "
            "(arXiv:2411.15633, 2024)."
        )

    @property
    def code_license(self) -> str:
        return "Pending official repository license release"

    @property
    def auxiliary_data_provenance(self) -> str:
        return (
            "Vision foundation models (e.g. DINOv2 / ViT) adapted via orthogonal Low-Rank "
            "adaptation matrices on forgery detection benchmarks."
        )

    @property
    def native_preprocessing(self) -> Dict[str, Any]:
        return {
            'image_size': 224,
            'crop_size': 224,
            'status': 'pending_verification_of_exact_backbone_norm',
        }

    @property
    def status(self) -> str:
        return 'pending_verification'

    def get_transforms(self):
        raise NotImplementedError(
            "Effort transforms are pending verification of the official backbone configuration."
        )

    def load_model(self):
        raise NotImplementedError(
            "Effort integration is pending release and verification of official model weights. "
            "Do not substitute random weights."
        )
