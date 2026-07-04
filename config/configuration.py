"""
Configuration Object.

The ``Config`` class is the single source of truth for every
configuration value used by the refactored codebase.  It organises
settings into logical sections accessible via clean attribute paths::

    config.training.batch_size
    config.data.crop_size
    config.wavelets.backend
    config.distributed.enabled
    config.experiment.name
    config.model.architecture

Construction:
    The primary construction path is ``Config.from_opt(opt)`` which
    delegates to the compatibility layer.  For testing or
    programmatic use, ``Config.from_defaults()`` creates a config
    with all default values.

Immutability:
    After construction, call ``config.freeze()`` to make the object
    read-only.  Any subsequent ``__setattr__`` will raise
    ``AttributeError``.

Serialization:
    ``config.to_dict()`` produces a nested dictionary.
    ``config.to_flat_dict()`` produces a flat dictionary with dotted
    keys (e.g. ``'training.learning_rate'``).

Usage::

    from config import Config, load_config

    # From legacy opt object
    config = Config.from_opt(opt)

    # From defaults (testing)
    config = Config.from_defaults()

    # Access values
    lr = config.training.learning_rate
    crop = config.data.crop_size
"""

from config.defaults import (
    DATA_DEFAULTS,
    AUGMENTATION_DEFAULTS,
    WAVELET_DEFAULTS,
    MODEL_DEFAULTS,
    TRAINING_DEFAULTS,
    DISTRIBUTED_DEFAULTS,
    EXPERIMENT_DEFAULTS,
    LOGGING_DEFAULTS,
    RUNTIME_DEFAULTS,
)


# =====================================================================
# Section Classes
# =====================================================================

class _FrozenMixin:
    """Mixin that supports freezing attribute assignment."""

    _frozen = False

    def freeze(self):
        """Make this section read-only."""
        object.__setattr__(self, '_frozen', True)

    @property
    def is_frozen(self):
        """Whether this section is frozen."""
        return self._frozen

    def __setattr__(self, name, value):
        if self._frozen:
            raise AttributeError(
                f"Cannot set '{name}' — configuration is frozen. "
                f"Create a new Config if you need different values."
            )
        super().__setattr__(name, value)


class DataConfig(_FrozenMixin):
    """Data pipeline configuration.

    Controls dataset roots, image dimensions, batching, and
    data loading behaviour.

    Attributes:
        dataroot (str): Root directory for the dataset.
        val_root (str or None): Separate validation root (if any).
        crop_size (int): Crop dimension applied during training.
        image_size (int): Resize dimension before cropping.
        batch_size (int): Training batch size.
        serial_batches (bool): If True, load images in order.
        no_flip (bool): Disable random horizontal flipping.
        no_crop (bool): Disable cropping.
        no_resize (bool): Disable resizing.
        class_bal (bool): Use class-balanced sampling.
        mode (str): Dataset mode (``'binary'`` or ``'filename'``).
        classes (str or list): Comma-separated class names or list.
        resize_or_crop (str): Resize/crop strategy string.
        compute_wavelets (bool): Whether to compute wavelets.
        train_split (str): Training split name.
        val_split (str): Validation split name.
    """

    def __init__(self, **kwargs):
        defaults = DATA_DEFAULTS.copy()
        defaults.update(kwargs)
        self.dataroot = defaults['dataroot']
        self.val_root = defaults['val_root']
        self.crop_size = defaults['crop_size']
        self.image_size = defaults['image_size']
        self.batch_size = defaults['batch_size']
        self.serial_batches = defaults['serial_batches']
        self.no_flip = defaults['no_flip']
        self.no_crop = defaults['no_crop']
        self.no_resize = defaults['no_resize']
        self.class_bal = defaults['class_bal']
        self.mode = defaults['mode']
        self.classes = defaults['classes']
        self.resize_or_crop = defaults['resize_or_crop']
        self.compute_wavelets = defaults['compute_wavelets']
        self.train_split = defaults['train_split']
        self.val_split = defaults['val_split']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class AugmentationConfig(_FrozenMixin):
    """Data augmentation configuration.

    Controls photometric augmentation, blurring, and JPEG compression
    applied during training.

    Attributes:
        blur_prob (float): Probability of applying Gaussian blur.
        blur_sig (list[float]): Blur sigma values.
        jpg_prob (float): Probability of applying JPEG compression.
        jpg_method (list[str]): JPEG method names.
        jpg_qual (list[int]): JPEG quality values.
        rz_interp (list[str]): Resize interpolation methods.
        data_aug (bool): Master switch for data augmentation.
    """

    def __init__(self, **kwargs):
        defaults = AUGMENTATION_DEFAULTS.copy()
        defaults.update(kwargs)
        self.blur_prob = defaults['blur_prob']
        self.blur_sig = defaults['blur_sig']
        self.jpg_prob = defaults['jpg_prob']
        self.jpg_method = defaults['jpg_method']
        self.jpg_qual = defaults['jpg_qual']
        self.rz_interp = defaults['rz_interp']
        self.data_aug = defaults['data_aug']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class WaveletConfig(_FrozenMixin):
    """Wavelet processing configuration.

    Controls the wavelet backend, decomposition parameters, and
    precomputed data paths.

    Attributes:
        backend (str): Backend name (``'cpu'``, ``'gpu'``,
            ``'precomputed'``).
        wavelet_type (str): Wavelet family (e.g. ``'haar'``).
        level (int): Decomposition level.
        mode (str): Signal extension mode (e.g. ``'reflect'``).
        log_packets (bool): Apply log-scaling to packet coefficients.
        precomputed_dir (str or None): Directory for precomputed
            wavelet packets.
    """

    def __init__(self, **kwargs):
        defaults = WAVELET_DEFAULTS.copy()
        defaults.update(kwargs)
        self.backend = defaults['backend']
        self.wavelet_type = defaults['wavelet_type']
        self.level = defaults['level']
        self.mode = defaults['mode']
        self.log_packets = defaults['log_packets']
        self.precomputed_dir = defaults['precomputed_dir']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class ModelConfig(_FrozenMixin):
    """Model architecture configuration.

    Controls model selection, initialization, and architecture-specific
    parameters.

    Attributes:
        architecture (str): Model architecture name (registry key).
        pretrained (bool): Use pretrained weights.
        num_classes (int): Number of output classes.
        init_type (str): Weight initialization method.
        init_gain (float): Initialization gain/scale.
        embed_dim (int): Embedding dimension (fusion/MHA models).
        num_heads (int): Number of attention heads (MHA model).
        dropout (float): Dropout probability.
        fusion_type (str): Fusion strategy (MHA model).
        freeze_base_models (bool): Freeze base model weights
            (fusion/MHA).
    """

    def __init__(self, **kwargs):
        defaults = MODEL_DEFAULTS.copy()
        defaults.update(kwargs)
        self.architecture = defaults['architecture']
        self.pretrained = defaults['pretrained']
        self.num_classes = defaults['num_classes']
        self.init_type = defaults['init_type']
        self.init_gain = defaults['init_gain']
        self.embed_dim = defaults['embed_dim']
        self.num_heads = defaults['num_heads']
        self.dropout = defaults['dropout']
        self.fusion_type = defaults['fusion_type']
        self.freeze_base_models = defaults['freeze_base_models']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class TrainingConfig(_FrozenMixin):
    """Training loop configuration.

    Controls epochs, optimizer, learning rate schedule, early stopping,
    and mixed precision settings.

    Attributes:
        epochs (int): Total training epochs (legacy ``niter``).
        epochs_decay (int): Extra decay epochs (legacy ``niter_decay``).
        learning_rate (float): Initial learning rate.
        optimizer (str): Optimizer family (``'adam'`` or ``'sgd'``).
        beta1 (float): Adam beta1 momentum.
        weight_decay (float): L2 regularisation.
        momentum (float): SGD momentum.
        lr_policy (str): LR scheduler policy.
        lr_decay_iters (int): Step scheduler step size.
        lr_gamma (float): LR decay factor.
        lr_patience (int): Plateau scheduler patience.
        earlystop_epoch (int): Early stopping patience.
        use_amp (bool): Enable automatic mixed precision.
        is_train (bool): Training mode flag.
        continue_train (bool): Resume from checkpoint.
        new_optim (bool): Use fresh optimizer (ignore checkpoint).
        epoch_count (int): Starting epoch counter.
        last_epoch (int): Last completed epoch (-1 = none).
    """

    def __init__(self, **kwargs):
        defaults = TRAINING_DEFAULTS.copy()
        defaults.update(kwargs)
        self.epochs = defaults['epochs']
        self.epochs_decay = defaults['epochs_decay']
        self.learning_rate = defaults['learning_rate']
        self.optimizer = defaults['optimizer']
        self.beta1 = defaults['beta1']
        self.weight_decay = defaults['weight_decay']
        self.momentum = defaults['momentum']
        self.lr_policy = defaults['lr_policy']
        self.lr_decay_iters = defaults['lr_decay_iters']
        self.lr_gamma = defaults['lr_gamma']
        self.lr_patience = defaults['lr_patience']
        self.earlystop_epoch = defaults['earlystop_epoch']
        self.use_amp = defaults['use_amp']
        self.is_train = defaults['is_train']
        self.continue_train = defaults['continue_train']
        self.new_optim = defaults['new_optim']
        self.epoch_count = defaults['epoch_count']
        self.last_epoch = defaults['last_epoch']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class DistributedConfig(_FrozenMixin):
    """Distributed training configuration.

    Controls DDP process group setup and multi-GPU behaviour.

    Attributes:
        enabled (bool): Whether distributed training is active.
        world_size (int): Total number of processes.
        rank (int): Global rank of this process.
        local_rank (int): Local rank on this node.
        backend (str or None): DDP backend (``'nccl'``, ``'gloo'``,
            or ``None`` for auto).
        dist_url (str): Process group initialisation URL.
        find_unused_parameters (bool): DDP flag for unused params.
    """

    def __init__(self, **kwargs):
        defaults = DISTRIBUTED_DEFAULTS.copy()
        defaults.update(kwargs)
        self.enabled = defaults['enabled']
        self.world_size = defaults['world_size']
        self.rank = defaults['rank']
        self.local_rank = defaults['local_rank']
        self.backend = defaults['backend']
        self.dist_url = defaults['dist_url']
        self.find_unused_parameters = defaults['find_unused_parameters']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class ExperimentConfig(_FrozenMixin):
    """Experiment identity and output configuration.

    Controls experiment naming and checkpoint directory layout.

    Attributes:
        name (str): Human-readable experiment name.
        checkpoints_dir (str): Root directory for checkpoints.
        epoch (str): Which epoch to load (``'latest'`` or number).
        suffix (str): Experiment name suffix template.
    """

    def __init__(self, **kwargs):
        defaults = EXPERIMENT_DEFAULTS.copy()
        defaults.update(kwargs)
        self.name = defaults['name']
        self.checkpoints_dir = defaults['checkpoints_dir']
        self.epoch = defaults['epoch']
        self.suffix = defaults['suffix']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class LoggingConfig(_FrozenMixin):
    """Logging and checkpoint frequency configuration.

    Controls how often metrics are logged, validation runs, and
    checkpoints are saved.

    Attributes:
        log_freq (int): General logging frequency (steps).
        loss_freq (int): Loss logging frequency (steps).
        val_epoch_freq (int): Validation frequency (epochs).
        save_epoch_freq (int): Epoch checkpoint frequency.
        save_latest_freq (int): Latest checkpoint frequency (steps).
    """

    def __init__(self, **kwargs):
        defaults = LOGGING_DEFAULTS.copy()
        defaults.update(kwargs)
        self.log_freq = defaults['log_freq']
        self.loss_freq = defaults['loss_freq']
        self.val_epoch_freq = defaults['val_epoch_freq']
        self.save_epoch_freq = defaults['save_epoch_freq']
        self.save_latest_freq = defaults['save_latest_freq']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


class RuntimeConfig(_FrozenMixin):
    """Runtime environment configuration.

    Controls device selection, data loading parallelism, and
    reproducibility settings.

    Attributes:
        gpu_ids (list[int]): GPU device IDs.
        num_workers (int): DataLoader worker processes.
        seed (int or None): RNG seed for reproducibility.
        deterministic (bool): Enable deterministic algorithms.
        pin_memory (bool): Pin DataLoader memory.
    """

    def __init__(self, **kwargs):
        defaults = RUNTIME_DEFAULTS.copy()
        defaults.update(kwargs)
        self.gpu_ids = defaults['gpu_ids']
        self.num_workers = defaults['num_workers']
        self.seed = defaults['seed']
        self.deterministic = defaults['deterministic']
        self.pin_memory = defaults['pin_memory']

    def to_dict(self):
        """Return this section as a dictionary."""
        return {k: v for k, v in self.__dict__.items()
                if not k.startswith('_')}


# =====================================================================
# Top-level Config
# =====================================================================

class Config(_FrozenMixin):
    """
    Top-level configuration container.

    Organises all configuration values into logical sections.
    Constructed from a legacy ``opt`` object via ``Config.from_opt``
    or with all defaults via ``Config.from_defaults``.

    Sections:
        - ``data`` (:class:`DataConfig`)
        - ``augmentation`` (:class:`AugmentationConfig`)
        - ``wavelets`` (:class:`WaveletConfig`)
        - ``model`` (:class:`ModelConfig`)
        - ``training`` (:class:`TrainingConfig`)
        - ``distributed`` (:class:`DistributedConfig`)
        - ``experiment`` (:class:`ExperimentConfig`)
        - ``logging`` (:class:`LoggingConfig`)
        - ``runtime`` (:class:`RuntimeConfig`)
    """

    def __init__(
        self,
        data=None,
        augmentation=None,
        wavelets=None,
        model=None,
        training=None,
        distributed=None,
        experiment=None,
        logging=None,
        runtime=None,
    ):
        self.data = data or DataConfig()
        self.augmentation = augmentation or AugmentationConfig()
        self.wavelets = wavelets or WaveletConfig()
        self.model = model or ModelConfig()
        self.training = training or TrainingConfig()
        self.distributed = distributed or DistributedConfig()
        self.experiment = experiment or ExperimentConfig()
        self.logging = logging or LoggingConfig()
        self.runtime = runtime or RuntimeConfig()

    # ------------------------------------------------------------------
    # Construction helpers
    # ------------------------------------------------------------------

    @classmethod
    def from_opt(cls, opt):
        """Create a Config from a legacy ``opt`` namespace.

        Delegates to :func:`config.compatibility.config_from_opt`.

        Args:
            opt: An ``argparse.Namespace`` (or similar) from the
                legacy options parser.

        Returns:
            A fully populated ``Config`` instance.
        """
        from config.compatibility import config_from_opt
        return config_from_opt(opt)

    @classmethod
    def from_defaults(cls):
        """Create a Config with all default values.

        Useful for testing and for modules that need a baseline
        configuration without parsing command-line arguments.

        Returns:
            A ``Config`` instance with all defaults.
        """
        return cls()

    # ------------------------------------------------------------------
    # Freeze
    # ------------------------------------------------------------------

    def freeze(self):
        """Freeze this config and all its sections.

        After freezing, any attempt to set an attribute on the config
        or any section will raise ``AttributeError``.
        """
        # Freeze all sections first
        for section_name in self._section_names():
            section = getattr(self, section_name)
            if hasattr(section, 'freeze'):
                section.freeze()
        # Freeze self
        object.__setattr__(self, '_frozen', True)

    def _section_names(self):
        """Return the names of all configuration sections."""
        return [
            'data', 'augmentation', 'wavelets', 'model',
            'training', 'distributed', 'experiment', 'logging',
            'runtime',
        ]

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        """Serialize to a nested dictionary.

        Returns:
            dict with section names as keys, each mapping to a
            dict of that section's attributes.
        """
        result = {}
        for name in self._section_names():
            section = getattr(self, name)
            result[name] = section.to_dict()
        return result

    def to_flat_dict(self):
        """Serialize to a flat dictionary with dotted keys.

        Example keys: ``'data.crop_size'``, ``'training.learning_rate'``.

        Returns:
            dict mapping dotted key strings to values.
        """
        flat = {}
        for section_name in self._section_names():
            section = getattr(self, section_name)
            for key, value in section.to_dict().items():
                flat[f'{section_name}.{key}'] = value
        return flat

    # ------------------------------------------------------------------
    # Display
    # ------------------------------------------------------------------

    def __repr__(self):
        lines = ['Config(']
        for name in self._section_names():
            section = getattr(self, name)
            section_dict = section.to_dict()
            lines.append(f'  {name}={{')
            for k, v in section_dict.items():
                lines.append(f'    {k}: {v!r},')
            lines.append('  },')
        lines.append(')')
        return '\n'.join(lines)

    def summary(self):
        """Print a concise human-readable summary."""
        print('=' * 60)
        print('Configuration Summary')
        print('=' * 60)
        for name in self._section_names():
            section = getattr(self, name)
            section_dict = section.to_dict()
            print(f'\n[{name}]')
            for k, v in section_dict.items():
                print(f'  {k}: {v!r}')
        print('=' * 60)
