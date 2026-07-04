"""
Configuration Type Definitions.

Provides strongly-typed enums for configuration values that were
previously scattered as magic strings across the codebase.

Each enum has a ``from_string`` classmethod that accepts the legacy
string values (case-insensitive) and returns the corresponding enum
member.  This enables seamless conversion from the ``opt`` namespace.

Usage::

    from config.types import WaveletBackend, OptimizerType

    backend = WaveletBackend.from_string('cpu')
    optim = OptimizerType.from_string('adam')
"""

from enum import Enum


class WaveletBackend(Enum):
    """Wavelet computation backend.

    Used by ``data.wavelets.backends.factory`` and dataset
    classes to select between CPU, GPU, and precomputed wavelet
    processing.
    """
    CPU = 'cpu'
    GPU = 'gpu'
    PRECOMPUTED = 'precomputed'

    @classmethod
    def from_string(cls, value):
        """Convert a legacy string value to a WaveletBackend enum.

        Args:
            value (str): One of ``'cpu'``, ``'gpu'``, ``'precomputed'``
                (case-insensitive).

        Returns:
            WaveletBackend

        Raises:
            ValueError: If the string is not a valid backend name.
        """
        if isinstance(value, cls):
            return value
        lookup = {member.value: member for member in cls}
        key = str(value).lower().strip()
        if key in lookup:
            return lookup[key]
        raise ValueError(
            f"Invalid wavelet backend '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )


class OptimizerType(Enum):
    """Optimizer family.

    Used by ``training.optimizer_factory`` and individual
    model trainers to select the optimizer.
    """
    ADAM = 'adam'
    SGD = 'sgd'

    @classmethod
    def from_string(cls, value):
        """Convert a legacy string value to an OptimizerType enum.

        Args:
            value (str): One of ``'adam'``, ``'sgd'``
                (case-insensitive).

        Returns:
            OptimizerType

        Raises:
            ValueError: If the string is not a valid optimizer name.
        """
        if isinstance(value, cls):
            return value
        lookup = {member.value: member for member in cls}
        key = str(value).lower().strip()
        if key in lookup:
            return lookup[key]
        raise ValueError(
            f"Invalid optimizer type '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )


class SchedulerType(Enum):
    """Learning-rate scheduler policy.

    Used by ``training.scheduler_factory`` to select the
    scheduler.
    """
    STEP = 'step'
    PLATEAU = 'plateau'
    COSINE = 'cosine'
    NONE = 'none'

    @classmethod
    def from_string(cls, value):
        """Convert a legacy string value to a SchedulerType enum.

        Args:
            value (str): One of ``'step'``, ``'plateau'``, ``'cosine'``,
                ``'none'`` (case-insensitive).

        Returns:
            SchedulerType

        Raises:
            ValueError: If the string is not a valid scheduler name.
        """
        if isinstance(value, cls):
            return value
        lookup = {member.value: member for member in cls}
        key = str(value).lower().strip()
        if key in lookup:
            return lookup[key]
        raise ValueError(
            f"Invalid scheduler type '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )


class ArchitectureType(Enum):
    """Model architecture identifiers.

    Maps to the keys used in
    ``models.registry._REGISTRY``.  The ``value`` of each
    member is the exact string used in the registry and in ``opt.arch``.
    """
    WANG2020_RAW = 'Wang2020Raw'
    WANG2020_128 = 'Wang2020_128'
    WOLTER_RAW = 'WolterWavelet2021Raw'
    WOLTER_128 = 'WolterWavelet2021_128'
    FUSION_128 = 'Fusion_128'
    MHA_128 = 'MHA_128'
    XCEPTION_RAW = 'XceptionRaw'
    # Legacy aliases
    RES50 = 'res50'

    @classmethod
    def from_string(cls, value):
        """Convert a legacy string value to an ArchitectureType enum.

        Args:
            value (str): Architecture name as used in ``opt.arch``
                (case-sensitive to match registry keys).

        Returns:
            ArchitectureType

        Raises:
            ValueError: If the string is not a valid architecture name.
        """
        if isinstance(value, cls):
            return value
        lookup = {member.value: member for member in cls}
        if value in lookup:
            return lookup[value]
        raise ValueError(
            f"Invalid architecture type '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )


class DeviceType(Enum):
    """Execution device type.

    Used by ``training.runtime.distributed_runtime`` for
    device resolution.
    """
    CPU = 'cpu'
    CUDA = 'cuda'

    @classmethod
    def from_string(cls, value):
        """Convert a legacy string value to a DeviceType enum.

        Args:
            value (str): One of ``'cpu'``, ``'cuda'``
                (case-insensitive).

        Returns:
            DeviceType

        Raises:
            ValueError: If the string is not a valid device type.
        """
        if isinstance(value, cls):
            return value
        lookup = {member.value: member for member in cls}
        key = str(value).lower().strip()
        if key in lookup:
            return lookup[key]
        raise ValueError(
            f"Invalid device type '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )


class InitType(Enum):
    """Network weight initialization method.

    Used by ``models.base.base_model.init_weights``.
    """
    NORMAL = 'normal'
    XAVIER = 'xavier'
    KAIMING = 'kaiming'
    ORTHOGONAL = 'orthogonal'

    @classmethod
    def from_string(cls, value):
        """Convert a legacy string value to an InitType enum.

        Args:
            value (str): One of ``'normal'``, ``'xavier'``,
                ``'kaiming'``, ``'orthogonal'`` (case-insensitive).

        Returns:
            InitType

        Raises:
            ValueError: If the string is not a valid init type.
        """
        if isinstance(value, cls):
            return value
        lookup = {member.value: member for member in cls}
        key = str(value).lower().strip()
        if key in lookup:
            return lookup[key]
        raise ValueError(
            f"Invalid init type '{value}'. "
            f"Valid options: {[m.value for m in cls]}"
        )
