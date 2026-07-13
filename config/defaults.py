"""
Configuration Defaults.

Single source of truth for every default value used by the refactored
codebase.  Previously these defaults were scattered across
``argparse.add_argument`` definitions, ``getattr(opt, ..., default)``
calls, and hard-coded literals in model trainers.

All defaults are organized by configuration section and exposed as
module-level dictionaries that the ``Config`` section classes import.

Usage::

    from config.defaults import DATA_DEFAULTS, TRAINING_DEFAULTS
    batch_size = DATA_DEFAULTS['batch_size']   # 64
"""

# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------
DATA_DEFAULTS = {
    'dataroot': './dataset/',
    'val_root': None,
    'crop_size': 224,
    'image_size': 256,
    'batch_size': 64,
    'serial_batches': False,
    'no_flip': False,
    'no_crop': False,
    'no_resize': False,
    'class_bal': False,
    'mode': 'binary',
    'classes': '',
    'resize_or_crop': 'scale_and_crop',
    'compute_wavelets': False,
    'train_split': 'train',
    'val_split': 'val',
}

# ------------------------------------------------------------------
# Augmentation
# ------------------------------------------------------------------
AUGMENTATION_DEFAULTS = {
    'blur_prob': 0.0,
    'blur_sig': [0.5],
    'jpg_prob': 0.0,
    'jpg_method': ['cv2'],
    'jpg_qual': [75],
    'rz_interp': ['bilinear'],
    'data_aug': False,
}

# ------------------------------------------------------------------
# Wavelets
# ------------------------------------------------------------------
WAVELET_DEFAULTS = {
    'backend': 'cpu',
    'wavelet_type': 'haar',
    'level': 3,
    'mode': 'reflect',
    'log_packets': True,
    'precomputed_dir': None,
}

# ------------------------------------------------------------------
# Model
# ------------------------------------------------------------------
MODEL_DEFAULTS = {
    'architecture': 'res50',
    'pretrained': True,
    'num_classes': 1,
    'init_type': 'normal',
    'init_gain': 0.02,
    'embed_dim': 128,
    'num_heads': 4,
    'dropout': 0.1,
    'fusion_type': 'cross_attention',
    'freeze_base_models': True,
    'rgb_model_path': None,
    'wavelet_model_path': None,
    'xception_model_path': None,
    'convnext_model_path': None,
}

# ------------------------------------------------------------------
# Training
# ------------------------------------------------------------------
TRAINING_DEFAULTS = {
    'epochs': 10000,
    'epochs_decay': 0,
    'learning_rate': 0.0001,
    'optimizer': 'adam',
    'beta1': 0.9,
    'weight_decay': 0.0,
    'momentum': 0.0,
    'lr_policy': 'none',
    'lr_decay_iters': 10,
    'lr_gamma': 0.1,
    'lr_patience': 5,
    'earlystop_epoch': 5,
    'use_amp': False,
    'is_train': True,
    'continue_train': False,
    'new_optim': False,
    'epoch_count': 1,
    'last_epoch': -1,
}

# ------------------------------------------------------------------
# Distributed
# ------------------------------------------------------------------
DISTRIBUTED_DEFAULTS = {
    'enabled': False,
    'world_size': 1,
    'rank': 0,
    'local_rank': 0,
    'backend': None,       # auto-detected: 'nccl' or 'gloo'
    'dist_url': 'env://',
    'find_unused_parameters': False,
}

# ------------------------------------------------------------------
# Experiment
# ------------------------------------------------------------------
EXPERIMENT_DEFAULTS = {
    'name': 'experiment_name',
    'checkpoints_dir': './checkpoints',
    'epoch': 'latest',
    'suffix': '',
}

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
LOGGING_DEFAULTS = {
    'log_freq': 50,
    'loss_freq': 400,
    'val_epoch_freq': 1,
    'save_epoch_freq': 20,
    'save_latest_freq': 2000,
}

# ------------------------------------------------------------------
# Runtime
# ------------------------------------------------------------------
RUNTIME_DEFAULTS = {
    'gpu_ids': [0],
    'num_workers': 4,
    'seed': None,
    'deterministic': False,
    'pin_memory': True,
}
