"""Adapter for Self-Blended Images (SBI) baseline (Shiohara & Matsuo, CVPR 2022).

Paper: Detecting Deepfakes with Self-Blended Images.
Official Repo: https://github.com/mapooon/SelfBlendedImages
"""
from typing import Dict, Any, Optional
from .base_adapter import BaseBaselineAdapter


class SBIAdapter(BaseBaselineAdapter):
    """Adapter for Shiohara & Matsuo CVPR 2022 Self-Blended Images baseline."""

    @property
    def name(self) -> str:
        return "SBI"

    @property
    def paper_citation(self) -> str:
        return (
            "Shiohara, K., & Matsuo, Y. (2022). Detecting deepfakes with self-blended images. "
            "In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (pp. 18720-18729)."
        )

    @property
    def code_license(self) -> str:
        return "Apache License 2.0"

    @property
    def auxiliary_data_provenance(self) -> str:
        return (
            "Trained on authentic pristine source images (e.g., FF++ pristine or CelebA) "
            "with dynamic landmark-guided self-blending masks; no external deepfake videos used during training."
        )

    @property
    def native_preprocessing(self) -> Dict[str, Any]:
        return {
            'image_size': 224,  # or 380 for EfficientNet-B4 variant
            'crop_size': 224,
            'mean': [0.485, 0.456, 0.406],
            'std': [0.229, 0.224, 0.225],
            'face_alignment': 'Dlib landmark 68-point or RetinaFace bounding box crop',
        }

    @property
    def status(self) -> str:
        return 'incomplete_stub_future_work'

    def get_transforms(self):
        """Return native ImageNet-standard transforms."""
        from torchvision import transforms
        p = self.native_preprocessing
        return transforms.Compose([
            transforms.Resize((p['image_size'], p['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=p['mean'], std=p['std']),
        ])

    def load_model(self):
        """Model instantiation stub; upstream weights and blending pipeline deferred."""
        raise NotImplementedError(
            "SBI is an architectural specification stub. "
            "Official EfficientNet-B4 checkpoint and dynamic blending pipeline are pending future external benchmark trials."
        )
