"""
Compatibility Layer.

Provides bidirectional conversion between the legacy ``opt`` namespace
(``argparse.Namespace``) and the new ``Config`` object.

Functions:
    ``config_from_opt(opt) -> Config``
        Convert a legacy opt object into a structured Config.

    ``config_to_opt(config) -> argparse.Namespace``
        Convert a Config back into an opt-shaped namespace for modules
        that still expect the legacy interface.

The attribute mapping handles naming differences between the legacy
flat namespace and the new structured sections::

    opt.cropSize       → config.data.crop_size
    opt.loadSize       → config.data.image_size
    opt.niter          → config.training.epochs
    opt.lr             → config.training.learning_rate
    opt.wavelet_backend → config.wavelets.backend

Usage::

    from config.compatibility import config_from_opt, config_to_opt

    config = config_from_opt(opt)
    opt_compat = config_to_opt(config)
"""

import argparse

from config.configuration import (
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


def _get(opt, name, default=None):
    """Safely get an attribute from an opt-like object.

    Works with ``argparse.Namespace``, plain objects, and dicts.
    """
    if isinstance(opt, dict):
        return opt.get(name, default)
    return getattr(opt, name, default)


def config_from_opt(opt):
    """Convert a legacy ``opt`` namespace to a ``Config``.

    Handles all known attribute mappings including legacy naming
    differences (``cropSize`` vs ``crop_size``, ``niter`` vs
    ``epochs``, etc.).

    Args:
        opt: An ``argparse.Namespace`` or any object with the legacy
            configuration attributes.

    Returns:
        Config: A fully populated configuration object.
    """
    # ------------------------------------------------------------------
    # Data section
    # ------------------------------------------------------------------
    data = DataConfig(
        dataroot=_get(opt, 'dataroot', './dataset/'),
        val_root=_get(opt, 'val_root', None),
        crop_size=_get(opt, 'cropSize', _get(opt, 'crop_size', 224)),
        image_size=_get(opt, 'loadSize', _get(opt, 'image_size', 256)),
        batch_size=_get(opt, 'batch_size', 64),
        serial_batches=_get(opt, 'serial_batches', False),
        no_flip=_get(opt, 'no_flip', False),
        no_crop=_get(opt, 'no_crop', False),
        no_resize=_get(opt, 'no_resize', False),
        class_bal=_get(opt, 'class_bal', False),
        mode=_get(opt, 'mode', 'binary'),
        classes=_get(opt, 'classes', ''),
        resize_or_crop=_get(opt, 'resize_or_crop', 'scale_and_crop'),
        compute_wavelets=_get(opt, 'compute_wavelets', True),
        train_split=_get(opt, 'train_split', 'train'),
        val_split=_get(opt, 'val_split', 'val'),
    )

    # ------------------------------------------------------------------
    # Augmentation section
    # ------------------------------------------------------------------
    # Handle both pre-processed list and raw comma-separated strings
    blur_sig = _get(opt, 'blur_sig', [0.5])
    if isinstance(blur_sig, str):
        blur_sig = [float(s) for s in blur_sig.split(',')]

    jpg_method = _get(opt, 'jpg_method', ['cv2'])
    if isinstance(jpg_method, str):
        jpg_method = jpg_method.split(',')

    jpg_qual = _get(opt, 'jpg_qual', [75])
    if isinstance(jpg_qual, str):
        jpg_qual = [int(s) for s in jpg_qual.split(',')]

    rz_interp = _get(opt, 'rz_interp', ['bilinear'])
    if isinstance(rz_interp, str):
        rz_interp = rz_interp.split(',')

    augmentation = AugmentationConfig(
        blur_prob=_get(opt, 'blur_prob', 0.0),
        blur_sig=blur_sig,
        jpg_prob=_get(opt, 'jpg_prob', 0.0),
        jpg_method=jpg_method,
        jpg_qual=jpg_qual,
        rz_interp=rz_interp,
        data_aug=_get(opt, 'data_aug', False),
    )

    # ------------------------------------------------------------------
    # Wavelets section
    # ------------------------------------------------------------------
    wavelets = WaveletConfig(
        backend=_get(opt, 'wavelet_backend', 'cpu'),
        wavelet_type=_get(opt, 'wavelet_type', 'haar'),
        level=_get(opt, 'wavelet_level', 3),
        mode=_get(opt, 'wavelet_mode', 'reflect'),
        log_packets=_get(opt, 'use_log_packets', True),
        precomputed_dir=_get(opt, 'precomputed_dir', None),
    )

    # ------------------------------------------------------------------
    # Model section
    # ------------------------------------------------------------------
    model = ModelConfig(
        architecture=_get(opt, 'arch', 'res50'),
        pretrained=_get(opt, 'pretrained', True),
        num_classes=_get(opt, 'num_classes', 1),
        init_type=_get(opt, 'init_type', 'normal'),
        init_gain=_get(opt, 'init_gain', 0.02),
        embed_dim=_get(opt, 'embed_dim', 128),
        num_heads=_get(opt, 'num_heads', 4),
        dropout=_get(opt, 'dropout', 0.1),
        fusion_type=_get(opt, 'fusion_type', 'cross_attention'),
        freeze_base_models=_get(opt, 'freeze_base_models', True),
    )

    # ------------------------------------------------------------------
    # Training section
    # ------------------------------------------------------------------
    training = TrainingConfig(
        epochs=_get(opt, 'niter', 10000),
        epochs_decay=_get(opt, 'niter_decay', 0),
        learning_rate=_get(opt, 'lr', 0.0001),
        optimizer=_get(opt, 'optim', 'adam'),
        beta1=_get(opt, 'beta1', 0.9),
        weight_decay=_get(opt, 'weight_decay', 0.0),
        momentum=_get(opt, 'momentum', 0.0),
        lr_policy=_get(opt, 'lr_policy', 'none'),
        lr_decay_iters=_get(opt, 'lr_decay_iters', 10),
        lr_gamma=_get(opt, 'lr_gamma', 0.1),
        lr_patience=_get(opt, 'lr_patience', 5),
        earlystop_epoch=_get(opt, 'earlystop_epoch', 5),
        use_amp=_get(opt, 'use_amp', False),
        is_train=_get(opt, 'isTrain', True),
        continue_train=_get(opt, 'continue_train', False),
        new_optim=_get(opt, 'new_optim', False),
        epoch_count=_get(opt, 'epoch_count', 1),
        last_epoch=_get(opt, 'last_epoch', -1),
    )

    # ------------------------------------------------------------------
    # Distributed section
    # ------------------------------------------------------------------
    import os
    world_size = int(os.environ.get('WORLD_SIZE', 1))

    distributed = DistributedConfig(
        enabled=(world_size > 1),
        world_size=world_size,
        rank=int(os.environ.get('RANK', 0)),
        local_rank=int(os.environ.get('LOCAL_RANK', 0)),
        backend=_get(opt, 'dist_backend', None),
        dist_url=_get(opt, 'dist_url', 'env://'),
        find_unused_parameters=_get(
            opt, 'find_unused_parameters', False),
    )

    # ------------------------------------------------------------------
    # Experiment section
    # ------------------------------------------------------------------
    experiment = ExperimentConfig(
        name=_get(opt, 'name', 'experiment_name'),
        checkpoints_dir=_get(opt, 'checkpoints_dir', './checkpoints'),
        epoch=_get(opt, 'epoch', 'latest'),
        suffix=_get(opt, 'suffix', ''),
    )

    # ------------------------------------------------------------------
    # Logging section
    # ------------------------------------------------------------------
    logging_cfg = LoggingConfig(
        log_freq=_get(opt, 'log_freq', 50),
        loss_freq=_get(opt, 'loss_freq', 400),
        val_epoch_freq=_get(opt, 'val_epoch_freq', 1),
        save_epoch_freq=_get(opt, 'save_epoch_freq', 20),
        save_latest_freq=_get(opt, 'save_latest_freq', 2000),
    )

    # ------------------------------------------------------------------
    # Runtime section
    # ------------------------------------------------------------------
    gpu_ids = _get(opt, 'gpu_ids', [0])
    if isinstance(gpu_ids, str):
        gpu_ids = [int(x) for x in gpu_ids.split(',') if x.strip()]

    runtime = RuntimeConfig(
        gpu_ids=gpu_ids,
        num_workers=_get(opt, 'num_threads',
                         _get(opt, 'num_workers', 4)),
        seed=_get(opt, 'seed', None),
        deterministic=_get(opt, 'deterministic', False),
        pin_memory=_get(opt, 'pin_memory', True),
    )

    return Config(
        data=data,
        augmentation=augmentation,
        wavelets=wavelets,
        model=model,
        training=training,
        distributed=distributed,
        experiment=experiment,
        logging=logging_cfg,
        runtime=runtime,
    )


def config_to_opt(config):
    """Convert a ``Config`` back to a legacy ``opt``-shaped namespace.

    Produces an ``argparse.Namespace`` that existing modules (model
    trainers, dataset classes, etc.) can consume without modification.

    The reverse mapping restores legacy naming conventions
    (``cropSize``, ``loadSize``, ``niter``, ``lr``, etc.).

    Args:
        config (Config): A structured configuration object.

    Returns:
        argparse.Namespace: A flat namespace with legacy attribute names.
    """
    opt = argparse.Namespace()

    # --- Data ---
    opt.dataroot = config.data.dataroot
    opt.cropSize = config.data.crop_size
    opt.loadSize = config.data.image_size
    opt.batch_size = config.data.batch_size
    opt.serial_batches = config.data.serial_batches
    opt.no_flip = config.data.no_flip
    opt.no_crop = config.data.no_crop
    opt.no_resize = config.data.no_resize
    opt.class_bal = config.data.class_bal
    opt.mode = config.data.mode
    opt.classes = config.data.classes
    opt.resize_or_crop = config.data.resize_or_crop
    opt.compute_wavelets = config.data.compute_wavelets
    opt.train_split = config.data.train_split
    opt.val_split = config.data.val_split

    # --- Augmentation ---
    opt.blur_prob = config.augmentation.blur_prob
    opt.blur_sig = config.augmentation.blur_sig
    opt.jpg_prob = config.augmentation.jpg_prob
    opt.jpg_method = config.augmentation.jpg_method
    opt.jpg_qual = config.augmentation.jpg_qual
    opt.rz_interp = config.augmentation.rz_interp
    opt.data_aug = config.augmentation.data_aug

    # --- Wavelets ---
    opt.wavelet_backend = config.wavelets.backend
    opt.wavelet_type = config.wavelets.wavelet_type
    opt.wavelet_level = config.wavelets.level
    opt.wavelet_mode = config.wavelets.mode
    opt.use_log_packets = config.wavelets.log_packets
    opt.precomputed_dir = config.wavelets.precomputed_dir

    # --- Model ---
    opt.arch = config.model.architecture
    opt.pretrained = config.model.pretrained
    opt.num_classes = config.model.num_classes
    opt.init_type = config.model.init_type
    opt.init_gain = config.model.init_gain
    opt.embed_dim = config.model.embed_dim
    opt.num_heads = config.model.num_heads
    opt.dropout = config.model.dropout
    opt.fusion_type = config.model.fusion_type
    opt.freeze_base_models = config.model.freeze_base_models

    # --- Training ---
    opt.niter = config.training.epochs
    opt.niter_decay = config.training.epochs_decay
    opt.lr = config.training.learning_rate
    opt.optim = config.training.optimizer
    opt.beta1 = config.training.beta1
    opt.weight_decay = config.training.weight_decay
    opt.momentum = config.training.momentum
    opt.lr_policy = config.training.lr_policy
    opt.lr_decay_iters = config.training.lr_decay_iters
    opt.lr_gamma = config.training.lr_gamma
    opt.lr_patience = config.training.lr_patience
    opt.earlystop_epoch = config.training.earlystop_epoch
    opt.use_amp = config.training.use_amp
    opt.isTrain = config.training.is_train
    opt.continue_train = config.training.continue_train
    opt.new_optim = config.training.new_optim
    opt.epoch_count = config.training.epoch_count
    opt.last_epoch = config.training.last_epoch

    # --- Distributed ---
    opt.dist_backend = config.distributed.backend
    opt.dist_url = config.distributed.dist_url
    opt.find_unused_parameters = config.distributed.find_unused_parameters

    # --- Experiment ---
    opt.name = config.experiment.name
    opt.checkpoints_dir = config.experiment.checkpoints_dir
    opt.epoch = config.experiment.epoch
    opt.suffix = config.experiment.suffix

    # --- Logging ---
    opt.log_freq = config.logging.log_freq
    opt.loss_freq = config.logging.loss_freq
    opt.val_epoch_freq = config.logging.val_epoch_freq
    opt.save_epoch_freq = config.logging.save_epoch_freq
    opt.save_latest_freq = config.logging.save_latest_freq

    # --- Runtime ---
    opt.gpu_ids = config.runtime.gpu_ids
    opt.num_threads = config.runtime.num_workers
    opt.num_workers = config.runtime.num_workers
    opt.seed = config.runtime.seed
    opt.deterministic = config.runtime.deterministic
    opt.pin_memory = config.runtime.pin_memory

    return opt
