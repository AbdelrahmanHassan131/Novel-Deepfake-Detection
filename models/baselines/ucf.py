"""Adapter for UCF baseline (Yan et al., ICCV 2023).

Paper: UCF: Uncovering Common Features for Generalizable Deepfake Detection.
Repository: Integrated via DeepfakeBench (https://github.com/SCLBD/DeepfakeBench).
"""
from typing import Dict, Any, Optional
from .base_adapter import BaseBaselineAdapter


class UCFAdapter(BaseBaselineAdapter):
    """Adapter for Yan et al. ICCV 2023 UCF (Uncovering Common Features) baseline."""

    @property
    def name(self) -> str:
        return "UCF"

    @property
    def paper_citation(self) -> str:
        return (
            "Yan, Z., Zhang, Y., Fan, Y., & Wu, B. (2023). UCF: Uncovering common features "
            "for generalizable deepfake detection. In Proceedings of the IEEE/CVF "
            "International Conference on Computer Vision (pp. 22412-22423)."
        )

    @property
    def code_license(self) -> str:
        return "Apache License 2.0"

    @property
    def auxiliary_data_provenance(self) -> str:
        return (
            "Trained on FaceForensics++ (c23 compression); learns disentangled representations "
            "separating forgery-common features from manipulation-specific artifacts."
        )

    @property
    def native_preprocessing(self) -> Dict[str, Any]:
        return {
            'image_size': 224,
            'crop_size': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
        }

    @property
    def status(self) -> str:
        return 'incomplete_stub_future_work'

    def get_transforms(self):
        """Return native ImageNet transforms for UCF."""
        from torchvision import transforms
        p = self.native_preprocessing
        return transforms.Compose([
            transforms.Resize((p['image_size'], p['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=p['mean'], std=p['std']),
        ])

    def load_model(self):
        """Model instantiation stub; upstream weights and architecture definition deferred."""
        raise NotImplementedError(
            "UCF is an architectural specification stub. "
            "Official UCF checkpoint and disentangled representation modules are pending future external benchmark trials."
        )
