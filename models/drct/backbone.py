"""DRCT Visual Backbone Factory and Implementations.

Supports the official CLIP ViT-L/14 backbone and provides an injectable
factory for offline CPU tests and mock backbones without requiring network access.
"""
from typing import Callable, Dict, Optional, Tuple
import torch
import torch.nn as nn


class DummyBackbone(nn.Module):
    """Deterministic tiny backbone for fast offline CPU testing.
    
    Transforms (B, 3, H, W) to (B, out_features) without downloading external weights.
    """
    def __init__(self, in_channels: int = 3, out_features: int = 768):
        super().__init__()
        self.out_features = out_features
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(in_channels, out_features, bias=True)
        # Initialize with deterministic weights
        nn.init.orthogonal_(self.proj.weight.data)
        if self.proj.bias is not None:
            nn.init.constant_(self.proj.bias.data, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W)
        b = x.shape[0]
        pooled = self.pool(x).view(b, -1)  # (B, 3)
        return self.proj(pooled)  # (B, out_features)


# Registry of backbone factory constructors
_BACKBONE_FACTORIES: Dict[str, Callable[..., Tuple[nn.Module, int]]] = {}

CLIP_CONFIGS = {
    'ViT-L/14': {'input_resolution': 224, 'patch_size': 14, 'width': 1024, 'layers': 24, 'heads': 16, 'output_dim': 768},
    'ViT-B/16': {'input_resolution': 224, 'patch_size': 16, 'width': 768, 'layers': 12, 'heads': 12, 'output_dim': 512},
    'ViT-B/32': {'input_resolution': 224, 'patch_size': 32, 'width': 768, 'layers': 12, 'heads': 12, 'output_dim': 512},
}


def normalize_backbone_name(name: str) -> str:
    """Normalize backbone identifier and resolve aliases."""
    cleaned = name.strip()
    lower = cleaned.lower().replace("-", "_").replace("/", "_")
    if lower in ('dummy', 'tiny', 'mock'):
        return 'dummy'
    if lower in ('clip_vit_l14', 'clip_vit_l_14', 'vit_l14', 'vit_l_14'):
        return 'ViT-L/14'
    if lower in ('clip_vit_b16', 'clip_vit_b_16', 'vit_b16', 'vit_b_16'):
        return 'ViT-B/16'
    if lower in ('clip_vit_b32', 'clip_vit_b_32', 'vit_b32', 'vit_b_32'):
        return 'ViT-B/32'
    if cleaned in CLIP_CONFIGS:
        return cleaned
    return cleaned


class CLIPViTBackbone(nn.Module):
    """Wrapper around official CLIP ViT visual encoder.
    
    Extracts the visual feature representation from CLIP ViT backbones without text modules.
    Supports:
    1. Pretrained initialization from local verified weights file (offline / cached).
    2. Architecture-only offline initialization without downloading weights (for unit testing).
    3. Explicit download from official repository when allow_download=True.
    """
    def __init__(
        self,
        model_name: str = "ViT-L/14",
        pretrained: bool = False,
        weights_path: Optional[str] = None,
        device: str = "cpu",
        download_if_missing: bool = False,
    ):
        super().__init__()
        resolved_name = normalize_backbone_name(model_name)
        if resolved_name not in CLIP_CONFIGS:
            raise ValueError(
                f"Unsupported CLIP model name: '{model_name}'. "
                f"Supported CLIP models: {list(CLIP_CONFIGS.keys())}"
            )
        config = CLIP_CONFIGS[resolved_name]
        self.model_name = resolved_name
        self.out_features = config['output_dim']

        try:
            import clip
            from clip.model import VisionTransformer
        except ImportError as e:
            raise ImportError(
                "The 'clip' package is required for real CLIPViTBackbone. "
                "Install it via 'pip install git+https://github.com/openai/CLIP.git' or use "
                "the 'dummy' backbone for offline CPU tests."
            ) from e

        if not pretrained and weights_path is None:
            # Construct real visual architecture directly without any weight download
            self.visual = VisionTransformer(**config).to(device)
        elif weights_path is not None:
            from pathlib import Path
            w_path = Path(weights_path).resolve()
            if not w_path.is_file():
                raise FileNotFoundError(f"Specified DRCT backbone weights file not found: {weights_path}")
            model, _ = clip.load(str(w_path), device=device, jit=False)
            self.visual = getattr(model, 'visual', model)
        else:
            # pretrained=True with no local weights_path
            if not download_if_missing:
                raise FileNotFoundError(
                    f"Pretrained weights for '{resolved_name}' requested with allow_download=False "
                    f"and no local weights_path provided. Supply a local path via --drct_backbone_weights or enable --drct_allow_download."
                )
            model, _ = clip.load(resolved_name, device=device, download_root=None, jit=False)
            self.visual = getattr(model, 'visual', model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.visual(x)


def register_backbone_factory(name: str, factory_fn: Callable[..., Tuple[nn.Module, int]]) -> None:
    """Register a custom backbone factory."""
    _BACKBONE_FACTORIES[name.lower()] = factory_fn


def build_drct_backbone(
    name: str = "ViT-L/14",
    pretrained: bool = False,
    weights_path: Optional[str] = None,
    freeze: bool = True,
    device: str = "cpu",
    allow_download: bool = False,
) -> Tuple[nn.Module, int]:
    """Construct a visual backbone for DRCT.
    
    Args:
        name: Backbone identifier ('ViT-L/14', 'clip_vit_b16', 'clip_vit_l14', 'dummy', etc.)
        pretrained: Whether to load pretrained weights
        weights_path: Optional local weights file
        freeze: If True, set requires_grad=False on all encoder parameters
        device: Target device for loading
        allow_download: If False, prohibits internet downloads
        
    Returns:
        (backbone_module, feature_dim)
    """
    resolved = normalize_backbone_name(name)
    
    # Check custom registry first
    if name.lower() in _BACKBONE_FACTORIES:
        backbone, dim = _BACKBONE_FACTORIES[name.lower()](
            name=name, pretrained=pretrained, weights_path=weights_path, device=device
        )
    elif resolved == "dummy":
        dim = 768
        backbone = DummyBackbone(in_channels=3, out_features=dim)
    elif resolved in CLIP_CONFIGS:
        dim = CLIP_CONFIGS[resolved]['output_dim']
        backbone = CLIPViTBackbone(
            model_name=resolved,
            pretrained=pretrained,
            weights_path=weights_path,
            device=device,
            download_if_missing=allow_download,
        )
    else:
        raise ValueError(
            f"Unsupported DRCT backbone: '{name}'. Supported backbones: "
            "['clip_vit_l14' (ViT-L/14), 'clip_vit_b16' (ViT-B/16), 'clip_vit_b32', 'dummy'] "
            "or register via register_backbone_factory."
        )

    if freeze:
        for param in backbone.parameters():
            param.requires_grad = False
        backbone.eval()

    return backbone, dim
