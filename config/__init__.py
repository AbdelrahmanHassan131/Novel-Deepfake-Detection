"""
Refactored Configuration System.

Public API::

    # Core configuration
    from Refactored.config import Config, load_config, load_config_from_defaults

    # Section classes
    from Refactored.config import (
        DataConfig, AugmentationConfig, WaveletConfig, ModelConfig,
        TrainingConfig, DistributedConfig, ExperimentConfig,
        LoggingConfig, RuntimeConfig,
    )

    # Compatibility layer
    from Refactored.config import config_from_opt, config_to_opt

    # Validation
    from Refactored.config import ConfigValidator, ConfigurationError

    # Path management
    from Refactored.config import PathManager

    # Type definitions
    from Refactored.config import (
        WaveletBackend, OptimizerType, SchedulerType,
        ArchitectureType, DeviceType, InitType,
    )

Usage::

    # Standard usage: convert legacy opt to Config
    config = load_config(opt)

    # Access structured values
    lr = config.training.learning_rate
    crop = config.data.crop_size
    backend = config.wavelets.backend

    # Path management
    paths = PathManager(config)
    print(paths.experiment_dir)
    print(paths.best_checkpoint)

    # Convert back to opt for legacy modules
    opt_compat = config_to_opt(config)
"""

# --- Core ---
from .configuration import (
    Config,
    DataConfig,
    AugmentationConfig,
    WaveletConfig,
    ModelConfig,
    TrainingConfig,
    DistributedConfig,
    ExperimentConfig,
    LoggingConfig,
    RuntimeConfig,
)

# --- Loader ---
from .loader import load_config, load_config_from_defaults

# --- Compatibility ---
from .compatibility import config_from_opt, config_to_opt

# --- Validation ---
from .validator import ConfigValidator, ConfigurationError, ValidationReport

# --- Paths ---
from .paths import PathManager

# --- Types ---
from .types import (
    WaveletBackend,
    OptimizerType,
    SchedulerType,
    ArchitectureType,
    DeviceType,
    InitType,
)

__all__ = [
    # Core
    'Config',
    'DataConfig',
    'AugmentationConfig',
    'WaveletConfig',
    'ModelConfig',
    'TrainingConfig',
    'DistributedConfig',
    'ExperimentConfig',
    'LoggingConfig',
    'RuntimeConfig',
    # Loader
    'load_config',
    'load_config_from_defaults',
    # Compatibility
    'config_from_opt',
    'config_to_opt',
    # Validation
    'ConfigValidator',
    'ConfigurationError',
    'ValidationReport',
    # Paths
    'PathManager',
    # Types
    'WaveletBackend',
    'OptimizerType',
    'SchedulerType',
    'ArchitectureType',
    'DeviceType',
    'InitType',
]
