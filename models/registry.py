"""
Model Registry.

Maps model names to their trainer classes.
Provides build_model(opt) to construct models without if/else chains.

Usage:
    from models import build_model

    opt.arch = 'Wang2020_128'
    model = build_model(opt)

Supported model names:
    - Wang2020Raw
    - Wang2020_128
    - WolterWavelet2021Raw
    - WolterWavelet2021_128
    - Fusion_128
    - MHA_128
    - XceptionRaw
"""

# Registry dict: model_name -> (module_path, class_name)
# Uses lazy imports to avoid loading all models at startup.
_REGISTRY = {}


def register_model(name, cls):
    """
    Register a model class under the given name.

    Args:
        name: String name for the model (used in opt.arch).
        cls: The model trainer class.
    """
    if name in _REGISTRY:
        raise ValueError(
            f"Model '{name}' is already registered. "
            f"Existing: {_REGISTRY[name]}, New: {cls}"
        )
    _REGISTRY[name] = cls


def get_registered_models():
    """Return a list of all registered model names."""
    _ensure_registered()
    return list(_REGISTRY.keys())


def _ensure_registered():
    """Lazily register all built-in models on first access."""
    if _REGISTRY:
        return

    from models.wang2020.trainer import Wang2020RawTrainer
    from models.wang2020_128.trainer import Wang2020_128Trainer
    from models.wolter2021.trainer_raw import WolterWaveletRawTrainer
    from models.wolter2021.trainer_128 import WolterWavelet128Trainer
    from models.fusion.trainer import ConcatenationFusionTrainer
    from models.mha.trainer import MHAFusionTrainer
    from models.xception.trainer import XceptionRawTrainer

    register_model('Wang2020Raw', Wang2020RawTrainer)
    register_model('Wang2020_128', Wang2020_128Trainer)
    register_model('WolterWavelet2021Raw', WolterWaveletRawTrainer)
    register_model('WolterWavelet2021_128', WolterWavelet128Trainer)
    register_model('Fusion_128', ConcatenationFusionTrainer)
    register_model('MHA_128', MHAFusionTrainer)
    register_model('XceptionRaw', XceptionRawTrainer)


def build_model(opt):
    """
    Build a model from an options object.

    Looks up opt.arch in the registry and instantiates the corresponding
    trainer class with opt.

    Args:
        opt: Options namespace. Must have opt.arch set to a registered model name.

    Returns:
        An instance of the model's trainer class.

    Raises:
        ValueError: If opt.arch is not found in the registry.
    """
    _ensure_registered()

    arch = getattr(opt, 'arch', None)
    if arch is None:
        raise ValueError(
            "opt.arch must be set. "
            f"Available models: {get_registered_models()}"
        )

    if arch not in _REGISTRY:
        raise ValueError(
            f"Model '{arch}' not found in registry. "
            f"Available models: {get_registered_models()}"
        )

    model_cls = _REGISTRY[arch]
    print(f"Building model: {arch} ({model_cls.__name__})")
    return model_cls(opt)
