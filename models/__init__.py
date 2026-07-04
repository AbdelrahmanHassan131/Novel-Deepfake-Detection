"""
Refactored Models Package.

Public API:
    from models import build_model
    from models import BaseModel, init_weights
    from models import register_model, get_registered_models

Usage:
    opt.arch = 'Wang2020_128'
    model = build_model(opt)

Supported architectures:
    - Wang2020Raw         : ResNet-50, Linear(2048, 1), full resolution
    - Wang2020_128        : ResNet-50, 128-dim embedding head
    - WolterWavelet2021Raw: Wavelet packet CNN, 512->1 classifier
    - WolterWavelet2021_128: Wavelet packet CNN, 128-dim embedding head
    - Fusion_128          : Concatenation fusion of RGB + Wavelet
    - MHA_128             : Multi-head attention fusion of RGB + Wavelet
    - XceptionRaw         : Xception, Linear(2048, 1)
"""

from .registry import build_model, register_model, get_registered_models
from .base.base_model import BaseModel, init_weights

__all__ = [
    'build_model',
    'register_model',
    'get_registered_models',
    'BaseModel',
    'init_weights',
]
