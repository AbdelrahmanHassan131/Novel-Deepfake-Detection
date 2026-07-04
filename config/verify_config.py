"""
Configuration System Verification.

Standalone script that verifies every component of the configuration
system works correctly.

Tests:
    1.  All imports resolve (no circular dependencies)
    2.  Enum types convert from strings correctly
    3.  Config.from_defaults() produces valid config
    4.  Config.from_opt(mock_opt) correctly maps all attributes
    5.  ConfigValidator catches known-bad configs
    6.  ConfigValidator accepts known-good configs
    7.  Default values match expected values
    8.  PathManager generates correct paths
    9.  config_to_opt() round-trip produces consistent values
    10. Nested section access works correctly
    11. to_dict() serialization works
    12. to_flat_dict() produces dotted keys
    13. Freeze mechanism prevents mutation
    14. Compatibility layer handles edge cases

Run::

    python -m config.verify_config
"""

import os
import sys
import argparse
import traceback


# ======================================================================
# Test infrastructure
# ======================================================================

_results = []


def _test(name, fn):
    """Run a test function, record pass/fail."""
    try:
        fn()
        _results.append(('PASS', name, None))
        print(f'  [PASS] {name}')
    except Exception as e:
        _results.append(('FAIL', name, str(e)))
        print(f'  [FAIL] {name}')
        traceback.print_exc()
        print()


def _assert(condition, message='Assertion failed'):
    """Assert with a descriptive message."""
    if not condition:
        raise AssertionError(message)


# ======================================================================
# Tests
# ======================================================================

def test_imports():
    """1. Verify all imports resolve without circular dependencies."""
    from config import (
        Config,
        DataConfig, AugmentationConfig, WaveletConfig,
        ModelConfig, TrainingConfig, DistributedConfig,
        ExperimentConfig, LoggingConfig, RuntimeConfig,
        load_config, load_config_from_defaults,
        config_from_opt, config_to_opt,
        ConfigValidator, ConfigurationError, ValidationReport,
        PathManager,
        WaveletBackend, OptimizerType, SchedulerType,
        ArchitectureType, DeviceType, InitType,
    )
    # Verify all are importable (no None values)
    _assert(Config is not None, 'Config is None')
    _assert(PathManager is not None, 'PathManager is None')
    _assert(WaveletBackend is not None, 'WaveletBackend is None')


def test_enum_types():
    """2. Verify enum conversions from legacy strings."""
    from config.types import (
        WaveletBackend, OptimizerType, SchedulerType,
        ArchitectureType, DeviceType, InitType,
    )

    # WaveletBackend
    _assert(WaveletBackend.from_string('cpu') == WaveletBackend.CPU)
    _assert(WaveletBackend.from_string('GPU') == WaveletBackend.GPU)
    _assert(WaveletBackend.from_string('precomputed')
            == WaveletBackend.PRECOMPUTED)

    # OptimizerType
    _assert(OptimizerType.from_string('adam') == OptimizerType.ADAM)
    _assert(OptimizerType.from_string('SGD') == OptimizerType.SGD)

    # SchedulerType
    _assert(SchedulerType.from_string('step') == SchedulerType.STEP)
    _assert(SchedulerType.from_string('none') == SchedulerType.NONE)
    _assert(SchedulerType.from_string('cosine') == SchedulerType.COSINE)

    # ArchitectureType
    _assert(ArchitectureType.from_string('Wang2020_128')
            == ArchitectureType.WANG2020_128)
    _assert(ArchitectureType.from_string('XceptionRaw')
            == ArchitectureType.XCEPTION_RAW)

    # DeviceType
    _assert(DeviceType.from_string('cpu') == DeviceType.CPU)
    _assert(DeviceType.from_string('cuda') == DeviceType.CUDA)

    # InitType
    _assert(InitType.from_string('normal') == InitType.NORMAL)
    _assert(InitType.from_string('xavier') == InitType.XAVIER)

    # Invalid string should raise ValueError
    try:
        WaveletBackend.from_string('invalid')
        _assert(False, 'Should have raised ValueError')
    except ValueError:
        pass  # expected


def test_config_from_defaults():
    """3. Config.from_defaults() produces a valid config."""
    from config import Config

    config = Config.from_defaults()
    _assert(config is not None, 'Config is None')
    _assert(config.data is not None, 'data section is None')
    _assert(config.training is not None, 'training section is None')
    _assert(config.wavelets is not None, 'wavelets section is None')
    _assert(config.model is not None, 'model section is None')
    _assert(config.experiment is not None, 'experiment section is None')


def test_config_from_opt():
    """4. Config.from_opt(mock_opt) correctly maps all attributes."""
    from config import Config

    opt = _create_mock_opt()
    config = Config.from_opt(opt)

    # Data mappings
    _assert(config.data.crop_size == 224,
            f'crop_size: {config.data.crop_size}')
    _assert(config.data.image_size == 256,
            f'image_size: {config.data.image_size}')
    _assert(config.data.batch_size == 32,
            f'batch_size: {config.data.batch_size}')
    _assert(config.data.dataroot == './dataset/',
            f'dataroot: {config.data.dataroot}')

    # Training mappings
    _assert(config.training.epochs == 100,
            f'epochs: {config.training.epochs}')
    _assert(config.training.learning_rate == 0.001,
            f'learning_rate: {config.training.learning_rate}')
    _assert(config.training.optimizer == 'adam',
            f'optimizer: {config.training.optimizer}')

    # Wavelet mappings
    _assert(config.wavelets.backend == 'cpu',
            f'backend: {config.wavelets.backend}')
    _assert(config.wavelets.wavelet_type == 'haar',
            f'wavelet_type: {config.wavelets.wavelet_type}')
    _assert(config.wavelets.level == 3,
            f'level: {config.wavelets.level}')

    # Model mappings
    _assert(config.model.architecture == 'Wang2020_128',
            f'architecture: {config.model.architecture}')

    # Experiment mappings
    _assert(config.experiment.name == 'test_experiment',
            f'name: {config.experiment.name}')
    _assert(config.experiment.checkpoints_dir == './checkpoints',
            f'checkpoints_dir: {config.experiment.checkpoints_dir}')

    # Runtime mappings
    _assert(config.runtime.gpu_ids == [0],
            f'gpu_ids: {config.runtime.gpu_ids}')


def test_validator_catches_bad_config():
    """5. ConfigValidator catches known-bad configs."""
    from config import Config, ConfigValidator

    # Create config with bad values
    config = Config.from_defaults()
    config.data.batch_size = -1
    config.data.crop_size = 300
    config.data.image_size = 256  # crop > image
    config.training.learning_rate = -0.001
    config.training.epochs = 0

    report = ConfigValidator.validate(config)
    _assert(not report.is_valid,
            'Expected validation to fail')
    _assert(len(report.errors) >= 3,
            f'Expected at least 3 errors, got {len(report.errors)}')


def test_validator_accepts_good_config():
    """6. ConfigValidator accepts known-good configs."""
    from config import Config, ConfigValidator

    config = Config.from_defaults()
    # Fix dataroot to avoid path warning
    config.data.dataroot = ''

    report = ConfigValidator.validate(config)
    _assert(report.is_valid,
            f'Expected validation to pass but got errors: '
            f'{report.errors}')


def test_default_values():
    """7. Default values match expected values."""
    from config.defaults import (
        DATA_DEFAULTS, TRAINING_DEFAULTS, WAVELET_DEFAULTS,
        MODEL_DEFAULTS, LOGGING_DEFAULTS, RUNTIME_DEFAULTS,
    )

    _assert(DATA_DEFAULTS['batch_size'] == 64)
    _assert(DATA_DEFAULTS['crop_size'] == 224)
    _assert(DATA_DEFAULTS['image_size'] == 256)
    _assert(TRAINING_DEFAULTS['learning_rate'] == 0.0001)
    _assert(TRAINING_DEFAULTS['optimizer'] == 'adam')
    _assert(WAVELET_DEFAULTS['backend'] == 'cpu')
    _assert(WAVELET_DEFAULTS['level'] == 3)
    _assert(MODEL_DEFAULTS['init_type'] == 'normal')
    _assert(LOGGING_DEFAULTS['loss_freq'] == 400)
    _assert(RUNTIME_DEFAULTS['gpu_ids'] == [0])


def test_path_manager():
    """8. PathManager generates correct paths."""
    from config import Config, PathManager

    config = Config.from_defaults()
    config.experiment.name = 'test_exp'
    config.experiment.checkpoints_dir = './ckpts'

    paths = PathManager(config)

    exp_dir = os.path.join('./ckpts', 'test_exp')
    _assert(paths.experiment_dir == exp_dir,
            f'experiment_dir: {paths.experiment_dir}')
    _assert(paths.best_checkpoint == os.path.join(exp_dir, 'best.pth'),
            f'best_checkpoint: {paths.best_checkpoint}')
    _assert(paths.last_checkpoint == os.path.join(exp_dir, 'last.pth'),
            f'last_checkpoint: {paths.last_checkpoint}')
    _assert(paths.history_json == os.path.join(exp_dir, 'history.json'),
            f'history_json: {paths.history_json}')
    _assert(paths.metrics_csv == os.path.join(exp_dir, 'metrics.csv'),
            f'metrics_csv: {paths.metrics_csv}')
    _assert(paths.epoch_checkpoint(5)
            == os.path.join(exp_dir, 'model_epoch_5.pth'),
            f'epoch_checkpoint(5): {paths.epoch_checkpoint(5)}')
    _assert(paths.logs_dir == os.path.join(exp_dir, 'logs'),
            f'logs_dir: {paths.logs_dir}')
    _assert(paths.plots_dir == os.path.join(exp_dir, 'plots'),
            f'plots_dir: {paths.plots_dir}')

    # Test from_values constructor
    paths2 = PathManager.from_values('./ckpts', 'test_exp')
    _assert(paths2.experiment_dir == paths.experiment_dir)


def test_round_trip():
    """9. config_to_opt() round-trip produces consistent values."""
    from config import Config, config_from_opt, config_to_opt

    opt = _create_mock_opt()
    config = config_from_opt(opt)
    opt_back = config_to_opt(config)

    # Check key round-trip mappings
    _assert(opt_back.cropSize == opt.cropSize,
            f'cropSize: {opt_back.cropSize} != {opt.cropSize}')
    _assert(opt_back.loadSize == opt.loadSize,
            f'loadSize: {opt_back.loadSize} != {opt.loadSize}')
    _assert(opt_back.batch_size == opt.batch_size,
            f'batch_size: {opt_back.batch_size} != {opt.batch_size}')
    _assert(opt_back.niter == opt.niter,
            f'niter: {opt_back.niter} != {opt.niter}')
    _assert(opt_back.lr == opt.lr,
            f'lr: {opt_back.lr} != {opt.lr}')
    _assert(opt_back.optim == opt.optim,
            f'optim: {opt_back.optim} != {opt.optim}')
    _assert(opt_back.arch == opt.arch,
            f'arch: {opt_back.arch} != {opt.arch}')
    _assert(opt_back.name == opt.name,
            f'name: {opt_back.name} != {opt.name}')
    _assert(opt_back.gpu_ids == opt.gpu_ids,
            f'gpu_ids: {opt_back.gpu_ids} != {opt.gpu_ids}')
    _assert(opt_back.wavelet_backend == opt.wavelet_backend,
            f'wavelet_backend: {opt_back.wavelet_backend} '
            f'!= {opt.wavelet_backend}')


def test_nested_access():
    """10. Nested section access works correctly."""
    from config import Config

    config = Config.from_defaults()

    # Test nested attribute access
    _assert(hasattr(config, 'data'), 'Missing data section')
    _assert(hasattr(config, 'training'), 'Missing training section')
    _assert(hasattr(config, 'wavelets'), 'Missing wavelets section')
    _assert(hasattr(config, 'model'), 'Missing model section')
    _assert(hasattr(config, 'distributed'), 'Missing distributed section')
    _assert(hasattr(config, 'experiment'), 'Missing experiment section')
    _assert(hasattr(config, 'logging'), 'Missing logging section')
    _assert(hasattr(config, 'runtime'), 'Missing runtime section')
    _assert(hasattr(config, 'augmentation'),
            'Missing augmentation section')

    # Test attribute access on sections
    _assert(hasattr(config.data, 'batch_size'))
    _assert(hasattr(config.training, 'learning_rate'))
    _assert(hasattr(config.wavelets, 'backend'))
    _assert(hasattr(config.model, 'architecture'))


def test_to_dict():
    """11. to_dict() serialization produces correct structure."""
    from config import Config

    config = Config.from_defaults()
    d = config.to_dict()

    _assert(isinstance(d, dict), 'to_dict() should return a dict')
    _assert('data' in d, 'Missing data key')
    _assert('training' in d, 'Missing training key')
    _assert('wavelets' in d, 'Missing wavelets key')
    _assert('model' in d, 'Missing model key')

    # Check nested values
    _assert('batch_size' in d['data'],
            'Missing batch_size in data dict')
    _assert(d['data']['batch_size'] == 64,
            f'batch_size should be 64, got {d["data"]["batch_size"]}')


def test_to_flat_dict():
    """12. to_flat_dict() produces dotted keys."""
    from config import Config

    config = Config.from_defaults()
    flat = config.to_flat_dict()

    _assert(isinstance(flat, dict), 'to_flat_dict() should return a dict')
    _assert('data.batch_size' in flat,
            'Missing data.batch_size key')
    _assert('training.learning_rate' in flat,
            'Missing training.learning_rate key')
    _assert(flat['data.batch_size'] == 64)
    _assert(flat['training.learning_rate'] == 0.0001)


def test_freeze():
    """13. Freeze mechanism prevents mutation."""
    from config import Config

    config = Config.from_defaults()

    # Before freeze: mutation should work
    config.data.batch_size = 128
    _assert(config.data.batch_size == 128,
            'Should be able to mutate before freeze')

    # Freeze
    config.freeze()
    _assert(config.is_frozen, 'Config should be frozen')
    _assert(config.data.is_frozen, 'Data section should be frozen')

    # After freeze: mutation should fail
    try:
        config.data.batch_size = 256
        _assert(False, 'Should have raised AttributeError')
    except AttributeError:
        pass  # expected

    try:
        config.training.learning_rate = 0.01
        _assert(False, 'Should have raised AttributeError')
    except AttributeError:
        pass  # expected


def test_compatibility_edge_cases():
    """14. Compatibility layer handles edge cases."""
    from config import config_from_opt

    # Test with minimal opt (missing many attributes)
    minimal_opt = argparse.Namespace()
    minimal_opt.isTrain = True
    config = config_from_opt(minimal_opt)
    _assert(config is not None, 'Should handle minimal opt')

    # Defaults should fill in
    _assert(config.data.batch_size == 64,
            f'Should default batch_size to 64, '
            f'got {config.data.batch_size}')
    _assert(config.training.learning_rate == 0.0001,
            f'Should default lr to 0.0001, '
            f'got {config.training.learning_rate}')

    # Test with string gpu_ids (pre-parse format)
    opt_str_gpu = argparse.Namespace()
    opt_str_gpu.gpu_ids = '0,1,2'
    config2 = config_from_opt(opt_str_gpu)
    _assert(config2.runtime.gpu_ids == [0, 1, 2],
            f'Should parse gpu_ids string, got '
            f'{config2.runtime.gpu_ids}')

    # Test with string blur_sig (pre-parse format)
    opt_str_blur = argparse.Namespace()
    opt_str_blur.blur_sig = '0.5,1.0'
    config3 = config_from_opt(opt_str_blur)
    _assert(config3.augmentation.blur_sig == [0.5, 1.0],
            f'Should parse blur_sig string, got '
            f'{config3.augmentation.blur_sig}')


# ======================================================================
# Mock opt factory
# ======================================================================

def _create_mock_opt():
    """Create a mock opt namespace mimicking the legacy parser output."""
    opt = argparse.Namespace()

    # Data
    opt.dataroot = './dataset/'
    opt.cropSize = 224
    opt.loadSize = 256
    opt.batch_size = 32
    opt.serial_batches = False
    opt.no_flip = False
    opt.no_crop = False
    opt.no_resize = False
    opt.class_bal = False
    opt.mode = 'binary'
    opt.classes = ['fake', 'real']
    opt.resize_or_crop = 'scale_and_crop'
    opt.compute_wavelets = True
    opt.train_split = 'train'
    opt.val_split = 'val'

    # Augmentation
    opt.blur_prob = 0.0
    opt.blur_sig = [0.5]
    opt.jpg_prob = 0.0
    opt.jpg_method = ['cv2']
    opt.jpg_qual = [75]
    opt.rz_interp = ['bilinear']
    opt.data_aug = False

    # Wavelets
    opt.wavelet_backend = 'cpu'
    opt.wavelet_type = 'haar'
    opt.wavelet_level = 3
    opt.wavelet_mode = 'reflect'
    opt.use_log_packets = True
    opt.precomputed_dir = None

    # Model
    opt.arch = 'Wang2020_128'
    opt.pretrained = True
    opt.num_classes = 1
    opt.init_type = 'normal'
    opt.init_gain = 0.02
    opt.embed_dim = 128
    opt.num_heads = 4
    opt.dropout = 0.1
    opt.fusion_type = 'cross_attention'
    opt.freeze_base_models = True

    # Training
    opt.niter = 100
    opt.niter_decay = 0
    opt.lr = 0.001
    opt.optim = 'adam'
    opt.beta1 = 0.9
    opt.weight_decay = 0.0
    opt.momentum = 0.0
    opt.lr_policy = 'none'
    opt.lr_decay_iters = 10
    opt.lr_gamma = 0.1
    opt.lr_patience = 5
    opt.earlystop_epoch = 5
    opt.use_amp = False
    opt.isTrain = True
    opt.continue_train = False
    opt.new_optim = False
    opt.epoch_count = 1
    opt.last_epoch = -1

    # Distributed
    opt.dist_backend = None
    opt.dist_url = 'env://'
    opt.find_unused_parameters = False

    # Experiment
    opt.name = 'test_experiment'
    opt.checkpoints_dir = './checkpoints'
    opt.epoch = 'latest'
    opt.suffix = ''

    # Logging
    opt.log_freq = 50
    opt.loss_freq = 400
    opt.val_epoch_freq = 1
    opt.save_epoch_freq = 20
    opt.save_latest_freq = 2000

    # Runtime
    opt.gpu_ids = [0]
    opt.num_threads = 4
    opt.seed = None
    opt.deterministic = False
    opt.pin_memory = True

    return opt


# ======================================================================
# Main
# ======================================================================

def main():
    """Run all verification tests."""
    print('=' * 60)
    print('Configuration System Verification')
    print('=' * 60)
    print()

    tests = [
        ('1. Imports (no circular deps)', test_imports),
        ('2. Enum type conversions', test_enum_types),
        ('3. Config.from_defaults()', test_config_from_defaults),
        ('4. Config.from_opt(mock_opt)', test_config_from_opt),
        ('5. Validator catches bad config', test_validator_catches_bad_config),
        ('6. Validator accepts good config', test_validator_accepts_good_config),
        ('7. Default values', test_default_values),
        ('8. PathManager paths', test_path_manager),
        ('9. Round-trip opt->Config->opt', test_round_trip),
        ('10. Nested section access', test_nested_access),
        ('11. to_dict() serialization', test_to_dict),
        ('12. to_flat_dict() dotted keys', test_to_flat_dict),
        ('13. Freeze immutability', test_freeze),
        ('14. Compatibility edge cases', test_compatibility_edge_cases),
    ]

    for name, fn in tests:
        _test(name, fn)

    # Summary
    print()
    print('=' * 60)
    passed = sum(1 for r in _results if r[0] == 'PASS')
    failed = sum(1 for r in _results if r[0] == 'FAIL')
    total = len(_results)
    print(f'Results: {passed}/{total} passed, {failed} failed')

    if failed > 0:
        print()
        print('Failed tests:')
        for status, name, error in _results:
            if status == 'FAIL':
                print(f'  [FAIL] {name}: {error}')

    print('=' * 60)
    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
