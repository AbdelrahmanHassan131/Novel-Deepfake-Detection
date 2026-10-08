"""
Configuration Validator.

Runs pre-flight consistency checks on a ``Config`` object before
training starts.  Catches common misconfigurations early with clear,
actionable error messages.

Checks are separated into **errors** (fatal — training cannot proceed)
and **warnings** (non-fatal — training can proceed but results may be
unexpected).

Usage::

    from config import ConfigValidator

    report = ConfigValidator.validate(config)
    report.raise_on_errors()   # raises ConfigurationError if any errors
    report.print_warnings()    # prints non-fatal warnings

    # Or use the convenience one-liner:
    ConfigValidator.validate_or_raise(config)
"""

import os

from config.types import (
    WaveletBackend,
    OptimizerType,
    SchedulerType,
)


class ConfigurationError(Exception):
    """Raised when configuration validation fails with fatal errors."""

    def __init__(self, errors):
        self.errors = errors
        msg = 'Configuration validation failed:\n'
        for i, err in enumerate(errors, 1):
            msg += f'  {i}. {err}\n'
        super().__init__(msg)


class ValidationReport:
    """Container for validation results.

    Attributes:
        errors (list[str]): Fatal errors that prevent training.
        warnings (list[str]): Non-fatal warnings.
    """

    def __init__(self):
        self.errors = []
        self.warnings = []

    @property
    def is_valid(self):
        """True if there are no fatal errors."""
        return len(self.errors) == 0

    def add_error(self, message):
        """Add a fatal error."""
        self.errors.append(message)

    def add_warning(self, message):
        """Add a non-fatal warning."""
        self.warnings.append(message)

    def raise_on_errors(self):
        """Raise ``ConfigurationError`` if there are fatal errors."""
        if self.errors:
            raise ConfigurationError(self.errors)

    def print_warnings(self):
        """Print all warnings to stdout."""
        if self.warnings:
            print('[ConfigValidator] Warnings:')
            for w in self.warnings:
                print(f'  - {w}')

    def print_report(self):
        """Print the full validation report."""
        if self.is_valid and not self.warnings:
            print('[ConfigValidator] Configuration is valid.')
            return

        if self.errors:
            print('[ConfigValidator] ERRORS:')
            for e in self.errors:
                print(f'  - {e}')

        self.print_warnings()


class ConfigValidator:
    """Validates a ``Config`` object for consistency.

    All checks are static methods so validation can be called without
    instantiation::

        report = ConfigValidator.validate(config)
    """

    @staticmethod
    def validate(config):
        """Run all validation checks on a Config.

        Args:
            config: A ``Config`` instance.

        Returns:
            ValidationReport with errors and warnings.
        """
        report = ValidationReport()

        ConfigValidator._check_data(config, report)
        ConfigValidator._check_augmentation(config, report)
        ConfigValidator._check_training(config, report)
        ConfigValidator._check_wavelets(config, report)
        ConfigValidator._check_model(config, report)
        ConfigValidator._check_runtime(config, report)
        ConfigValidator._check_logging(config, report)

        return report

    @staticmethod
    def validate_or_raise(config):
        """Validate and raise on errors, print warnings otherwise.

        Convenience method that combines ``validate``, ``raise_on_errors``,
        and ``print_warnings``.

        Args:
            config: A ``Config`` instance.

        Returns:
            ValidationReport (only reachable if no fatal errors).

        Raises:
            ConfigurationError: If there are fatal validation errors.
        """
        report = ConfigValidator.validate(config)
        report.raise_on_errors()
        report.print_warnings()
        return report

    # ------------------------------------------------------------------
    # Section checks
    # ------------------------------------------------------------------

    @staticmethod
    def _check_data(config, report):
        """Validate data configuration."""
        d = config.data

        if d.batch_size <= 0:
            report.add_error(
                f'data.batch_size must be > 0, got {d.batch_size}')

        if d.crop_size <= 0:
            report.add_error(
                f'data.crop_size must be > 0, got {d.crop_size}')

        if d.image_size <= 0:
            report.add_error(
                f'data.image_size must be > 0, got {d.image_size}')

        if d.crop_size > d.image_size:
            report.add_error(
                f'data.crop_size ({d.crop_size}) must be <= '
                f'data.image_size ({d.image_size})')

        if d.val_batch_size is not None and d.val_batch_size <= 0:
            report.add_error(f'data.val_batch_size must be > 0, got {d.val_batch_size}')
        if getattr(d, 'crop_policy', 'scale_and_crop') not in ('scale_and_crop', 'random_resized_crop', 'patch_crop'):
            report.add_error(f"data.crop_policy must be 'scale_and_crop', 'random_resized_crop', or 'patch_crop', got {getattr(d, 'crop_policy', None)}")

        if d.val_num_workers is not None and d.val_num_workers < 0:
            report.add_error(f'data.val_num_workers must be >= 0, got {d.val_num_workers}')

        if d.dataroot and not os.path.exists(d.dataroot):
            report.add_warning(
                f'data.dataroot does not exist: {d.dataroot}')

    @staticmethod
    def _check_augmentation(config, report):
        a = config.augmentation
        if getattr(a, 'aug_recipe', 'legacy') not in ('legacy', 'rgb_v1', 'custom'):
            report.add_error(f"augmentation.aug_recipe must be 'legacy', 'rgb_v1', or 'custom', got {getattr(a, 'aug_recipe', None)}")

    @staticmethod
    def _check_training(config, report):
        """Validate training configuration."""
        t = config.training

        if t.epochs <= 0:
            report.add_error(
                f'training.epochs must be > 0, got {t.epochs}')

        if t.learning_rate <= 0:
            report.add_error(
                f'training.learning_rate must be > 0, '
                f'got {t.learning_rate}')

        # Validate optimizer type
        try:
            OptimizerType.from_string(t.optimizer)
        except ValueError:
            report.add_error(
                f'training.optimizer is not a supported optimizer: '
                f'{t.optimizer!r}. '
                f'Valid: {[m.value for m in OptimizerType]}')

        # Validate scheduler type
        try:
            SchedulerType.from_string(t.lr_policy)
        except ValueError:
            report.add_error(
                f'training.lr_policy is not a supported scheduler: '
                f'{t.lr_policy!r}. '
                f'Valid: {[m.value for m in SchedulerType]}')

        if getattr(t, 'amp_dtype', 'fp16') not in ('fp16', 'bf16'):
            report.add_error(f"training.amp_dtype must be 'fp16' or 'bf16', got {getattr(t, 'amp_dtype', None)}")

        if getattr(t, 'val_precision', 'fp32') not in ('fp32', 'amp', 'fp16', 'bf16'):
            report.add_error(f"training.val_precision must be 'fp32', 'amp', 'fp16', or 'bf16', got {getattr(t, 'val_precision', None)}")

        valid_monitors = ('auc', 'balanced_accuracy', 'accuracy', 'source_macro_auc', 'worst_source_recall_05')
        if getattr(t, 'monitor_metric', 'auc') not in valid_monitors:
            report.add_error(f"training.monitor_metric must be one of {valid_monitors}, got {getattr(t, 'monitor_metric', None)}")

        if getattr(t, 'early_stopping_patience', 5) < 0:
            report.add_error(f"training.early_stopping_patience must be >= 0, got {getattr(t, 'early_stopping_patience', None)}")

        if getattr(t, 'early_stopping_min_delta', 0.0) < 0.0:
            report.add_error(f"training.early_stopping_min_delta must be >= 0.0, got {getattr(t, 'early_stopping_min_delta', None)}")

        if getattr(t, 'early_stopping_min_epochs', 0) < 0:
            report.add_error(f"training.early_stopping_min_epochs must be >= 0, got {getattr(t, 'early_stopping_min_epochs', None)}")

        if t.weight_decay < 0:
            report.add_warning(
                f'training.weight_decay is negative: {t.weight_decay}')

    @staticmethod
    def _check_wavelets(config, report):
        """Validate wavelet configuration."""
        w = config.wavelets

        try:
            WaveletBackend.from_string(w.backend)
        except ValueError:
            report.add_error(
                f'wavelets.backend is not a supported backend: '
                f'{w.backend!r}. '
                f'Valid: {[m.value for m in WaveletBackend]}')

        if w.level <= 0:
            report.add_error(
                f'wavelets.level must be > 0, got {w.level}')

        # Precomputed backend requires precomputed_dir
        if w.backend == 'precomputed' and not w.precomputed_dir:
            report.add_error(
                'wavelets.precomputed_dir must be set when using '
                "the 'precomputed' backend.")

    @staticmethod
    def _check_model(config, report):
        """Validate model configuration."""
        m = config.model

        if not m.architecture:
            report.add_error('model.architecture must not be empty.')

        # Check against known architectures (non-fatal — could be
        # a custom registered model).
        from config.types import ArchitectureType
        known = [member.value for member in ArchitectureType]
        if m.architecture not in known:
            report.add_warning(
                f'model.architecture {m.architecture!r} is not in '
                f'the known architecture list: {known}. '
                f'Ensure it is registered in the model registry.')

        if getattr(m, 'rgb_head_type', '128d') not in ('128d', 'linear'):
            report.add_error(f"model.rgb_head_type must be '128d' or 'linear', got {getattr(m, 'rgb_head_type', None)}")

        if getattr(m, 'rgb_dropout', 0.5) < 0.0 or getattr(m, 'rgb_dropout', 0.5) > 1.0:
            report.add_error(f"model.rgb_dropout must be between 0.0 and 1.0, got {getattr(m, 'rgb_dropout', None)}")

        if getattr(m, 'fine_tune_policy', 'full') not in ('full', 'head_only', 'layer4_and_head'):
            report.add_error(f"model.fine_tune_policy must be 'full', 'head_only', or 'layer4_and_head', got {getattr(m, 'fine_tune_policy', None)}")

        if getattr(m, 'backbone_lr_mult', 1.0) < 0.0:
            report.add_error(f"model.backbone_lr_mult must be >= 0.0, got {getattr(m, 'backbone_lr_mult', None)}")

        if getattr(m, 'bn_policy', 'train') not in ('train', 'frozen'):
            report.add_error(f"model.bn_policy must be 'train' or 'frozen', got {getattr(m, 'bn_policy', None)}")

        if m.init_gain <= 0:
            report.add_warning(
                f'model.init_gain is non-positive: {m.init_gain}')

    @staticmethod
    def _check_runtime(config, report):
        """Validate runtime configuration."""
        r = config.runtime

        if isinstance(r.gpu_ids, list):
            for gid in r.gpu_ids:
                if gid < -1:
                    report.add_error(
                        f'runtime.gpu_ids contains invalid ID: {gid}')

        if r.prefetch_factor is not None and r.prefetch_factor <= 0:
            report.add_error(f'runtime.prefetch_factor must be > 0, got {r.prefetch_factor}')

        if r.num_workers < 0:
            report.add_error(
                f'runtime.num_workers must be >= 0, '
                f'got {r.num_workers}')

    @staticmethod
    def _check_logging(config, report):
        """Validate logging configuration."""
        lg = config.logging

        if lg.log_freq <= 0:
            report.add_warning(
                f'logging.log_freq should be > 0, got {lg.log_freq}')

        if lg.save_epoch_freq <= 0:
            report.add_warning(
                f'logging.save_epoch_freq should be > 0, '
                f'got {lg.save_epoch_freq}')
