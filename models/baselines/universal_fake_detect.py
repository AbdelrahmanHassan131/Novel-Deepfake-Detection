"""Adapter for UniversalFakeDetect (Ojha et al., CVPR 2023).

Paper: Towards Universal Fake Image Detectors That Generalize Across Generative Models.
Official Repo: https://github.com/peterwang512/universal_fake_detect
"""
from typing import Dict, Any, Optional
from .base_adapter import BaseBaselineAdapter


class UniversalFakeDetectAdapter(BaseBaselineAdapter):
    """Adapter for Ojha et al. CVPR 2023 Universal Fake Image Detector."""

    @property
    def name(self) -> str:
        return "UniversalFakeDetect"

    @property
    def paper_citation(self) -> str:
        return (
            "Ojha, U., Li, Y., & Lee, Y. J. (2023). Towards universal fake image detectors "
            "that generalize across generative models. In Proceedings of the IEEE/CVF "
            "Conference on Computer Vision and Pattern Recognition (pp. 24480-24489)."
        )

    @property
    def code_license(self) -> str:
        return "MIT License"

    @property
    def auxiliary_data_provenance(self) -> str:
        return (
            "Pretrained OpenAI CLIP ViT-L/14 visual encoder (frozen); "
            "linear classifier probe trained on ProGAN authentic/generated images."
        )

    @property
    def native_preprocessing(self) -> Dict[str, Any]:
        return {
            'image_size': 224,
            'crop_size': 224,
            'resize_mode': 'bicubic',
            'mean': [0.48145466, 0.4578275, 0.40821073],
            'std': [0.26862954, 0.26130258, 0.27577711],
            'interpolation': 'bicubic',
        }

    @property
    def status(self) -> str:
        return 'incomplete_stub_future_work'

    def get_transforms(self):
        """Return native CLIP transforms."""
        from torchvision import transforms
        p = self.native_preprocessing
        return transforms.Compose([
            transforms.Resize(p['image_size'], interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.CenterCrop(p['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=p['mean'], std=p['std']),
        ])

    def load_model(self):
        """Model instantiation stub; upstream weights and CLIP linear probe deferred."""
        raise NotImplementedError(
            "UniversalFakeDetect is an architectural specification stub. "
            "Official linear probe weights and CLIP integration are pending future external benchmark trials."
        )
