"""Baseline models and adapters package.

All adapters provide lazy imports and do not download weights or instantiate
models on module import.
"""
from typing import Dict, Type
from .base_adapter import BaseBaselineAdapter
from .universal_fake_detect import UniversalFakeDetectAdapter
from .sbi import SBIAdapter
from .ucf import UCFAdapter
from .effort import EffortAdapter

BASELINE_REGISTRY: Dict[str, Type[BaseBaselineAdapter]] = {
    'universal_fake_detect': UniversalFakeDetectAdapter,
    'sbi': SBIAdapter,
    'ucf': UCFAdapter,
    'effort': EffortAdapter,
}


def get_baseline_adapter(name: str, checkpoint_path: str = None, evaluation_mode: str = 'supplied_checkpoint', device: str = 'cpu') -> BaseBaselineAdapter:
    """Retrieve an initialized baseline adapter."""
    key = name.lower().replace('-', '_').replace(' ', '_')
    if key not in BASELINE_REGISTRY:
        raise ValueError(
            f"Unknown baseline: '{name}'. Available baselines: {list(BASELINE_REGISTRY.keys())}"
        )
    return BASELINE_REGISTRY[key](checkpoint_path=checkpoint_path, evaluation_mode=evaluation_mode, device=device)
