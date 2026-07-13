#!/usr/bin/env python
"""
Main Training Script for Deepfake Detection Models.

This is the primary entry point for launching training in the refactored codebase.
It integrates configuration loading, dataset construction, experiment management,
and the training engine.

Usage:
    # Single GPU training for Wang2020Raw:
    python train.py --arch Wang2020Raw --dataroot ./dataset/train --name wang2020_run --gpu_ids 0

    # Multi-GPU training via torchrun:
    torchrun --nproc_per_node=4 train.py --arch Wang2020_128 --dataroot ./dataset/train ...
"""

import os
import sys
import argparse
import torch

from config import load_config, config_to_opt, ConfigValidator
from config.defaults import DATA_DEFAULTS, TRAINING_DEFAULTS, MODEL_DEFAULTS, RUNTIME_DEFAULTS, EXPERIMENT_DEFAULTS, WAVELET_DEFAULTS, AUGMENTATION_DEFAULTS
from models import build_model, get_registered_models
from data.loaders.dataloader_factory import create_dataloader, create_mha_dataloader
from training import Trainer, seed_everything
from experiment import ExperimentManager


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Models")

    # Model / Architecture
    parser.add_argument('--arch', type=str, default='Wang2020Raw',
                        help=f"Model architecture to train. Available: {get_registered_models()}")
    parser.add_argument('--pretrained', action='store_true', default=MODEL_DEFAULTS['pretrained'],
                        help="Use pretrained weights for backbone networks")
    parser.add_argument('--num_classes', type=int, default=MODEL_DEFAULTS['num_classes'],
                        help="Number of output classes")
    parser.add_argument('--init_type', type=str, default=MODEL_DEFAULTS['init_type'],
                        help="Network initialization method (normal, xavier, etc.)")
    parser.add_argument('--init_gain', type=float, default=MODEL_DEFAULTS['init_gain'],
                        help="Scaling factor for initialization")

    # Data & Dataset
    parser.add_argument('--dataroot', type=str, required=True,
                        help="Path to dataset root directory (must contain class subfolders e.g., fake/ and real/)")
    parser.add_argument('--val_root', type=str, default=None,
                        help="Path to validation dataset root directory (optional)")
    parser.add_argument('--classes', type=str, default='fake,real',
                        help="Comma-separated list of class names (subfolders in dataroot)")
    parser.add_argument('--mode', type=str, default=DATA_DEFAULTS['mode'], choices=['binary', 'filename'],
                        help="Dataset mode")
    parser.add_argument('--batch_size', type=int, default=DATA_DEFAULTS['batch_size'],
                        help="Input batch size")
    parser.add_argument('--image_size', type=int, default=DATA_DEFAULTS['image_size'],
                        help="Scale images to this size")
    parser.add_argument('--crop_size', type=int, default=DATA_DEFAULTS['crop_size'],
                        help="Crop images to this size")
    parser.add_argument('--serial_batches', action='store_true', default=DATA_DEFAULTS['serial_batches'],
                        help="If true, takes images in order without shuffling")
    parser.add_argument('--no_flip', action='store_true', default=DATA_DEFAULTS['no_flip'],
                        help="If specified, do not flip the images for data augmentation")
    parser.add_argument('--no_crop', action='store_true', default=DATA_DEFAULTS['no_crop'],
                        help="If specified, do not crop images")
    parser.add_argument('--no_resize', action='store_true', default=DATA_DEFAULTS['no_resize'],
                        help="If specified, do not resize images")
    parser.add_argument('--class_bal', action='store_true', default=DATA_DEFAULTS['class_bal'],
                        help="Use class balanced sampler")
    parser.add_argument('--compute_wavelets', action='store_true', default=DATA_DEFAULTS['compute_wavelets'],
                        help="Compute wavelets online if required by dataset")

    # Augmentation parameters
    parser.add_argument('--blur_prob', type=float, default=AUGMENTATION_DEFAULTS['blur_prob'],
                        help="Probability of applying Gaussian blur data augmentation (e.g., 0.5)")
    parser.add_argument('--blur_sig', type=str, default='0.5',
                        help="Comma-separated blur sigma range (e.g., '0.0,3.0' or '0.5')")
    parser.add_argument('--jpg_prob', type=float, default=AUGMENTATION_DEFAULTS['jpg_prob'],
                        help="Probability of applying JPEG compression data augmentation (e.g., 0.5)")
    parser.add_argument('--jpg_method', type=str, default='cv2',
                        help="Comma-separated JPEG compression methods (e.g., 'cv2,pil')")
    parser.add_argument('--jpg_qual', type=str, default='75',
                        help="Comma-separated JPEG quality values (e.g., '30,75')")

    # Wavelet parameters
    parser.add_argument('--wavelet_backend', type=str, default=WAVELET_DEFAULTS['backend'], choices=['cpu', 'gpu', 'precomputed'],
                        help="Wavelet computation backend: cpu, gpu, or precomputed")
    parser.add_argument('--wavelet_type', type=str, default=WAVELET_DEFAULTS['wavelet_type'],
                        help="Wavelet family name (e.g. haar, db2)")
    parser.add_argument('--wavelet_level', type=int, default=WAVELET_DEFAULTS['level'],
                        help="Wavelet decomposition level")
    parser.add_argument('--wavelet_mode', type=str, default=WAVELET_DEFAULTS['mode'],
                        help="Signal extension mode")
    parser.add_argument('--use_log_packets', action='store_true', default=WAVELET_DEFAULTS['log_packets'],
                        help="Apply log scaling to wavelet packets")

    # Training Hyperparameters
    parser.add_argument('--epochs', '--niter', dest='epochs', type=int, default=TRAINING_DEFAULTS['epochs'],
                        help="Number of epochs to train")
    parser.add_argument('--epochs_decay', '--niter_decay', dest='epochs_decay', type=int, default=TRAINING_DEFAULTS['epochs_decay'],
                        help="Number of epochs to linearly decay learning rate to zero")
    parser.add_argument('--lr', type=float, default=TRAINING_DEFAULTS['learning_rate'],
                        help="Initial learning rate")
    parser.add_argument('--optim', dest='optimizer', type=str, default=TRAINING_DEFAULTS['optimizer'], choices=['adam', 'sgd'],
                        help="Optimizer type")
    parser.add_argument('--beta1', type=float, default=TRAINING_DEFAULTS['beta1'],
                        help="Momentum term beta1 for Adam")
    parser.add_argument('--weight_decay', type=float, default=TRAINING_DEFAULTS['weight_decay'],
                        help="Weight decay for optimizer")
    parser.add_argument('--momentum', type=float, default=TRAINING_DEFAULTS['momentum'],
                        help="Momentum for SGD")
    parser.add_argument('--lr_policy', type=str, default=TRAINING_DEFAULTS['lr_policy'],
                        help="Learning rate policy (none, step, cosine, plateau)")
    parser.add_argument('--continue_train', action='store_true', default=TRAINING_DEFAULTS['continue_train'],
                        help="Continue training from latest checkpoint")
    parser.add_argument('--epoch', type=str, default='latest',
                        help="Which epoch to load when continue_train is set (e.g., 'latest' or '10')")
    parser.add_argument('--use_amp', action='store_true', default=TRAINING_DEFAULTS['use_amp'],
                        help="Enable Automatic Mixed Precision (AMP)")

    # Runtime & GPU
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help="Comma-separated list of GPU IDs to use (e.g., '0' or '0,1'). Use '-1' for CPU.")
    parser.add_argument('--num_workers', '--num_threads', dest='num_workers', type=int, default=0,
                        help="Number of data loading threads (default 0 for Windows stability)")
    parser.add_argument('--seed', type=int, default=None,
                        help="Random seed for reproducibility")
    parser.add_argument('--deterministic', action='store_true', default=False,
                        help="Enable deterministic mode for reproducibility")

    # MHA / Fusion parameters
    parser.add_argument('--rgb_model_path', type=str, default=None,
                        help="Path to pre-trained RGB checkpoint for fusion models")
    parser.add_argument('--wavelet_model_path', type=str, default=None,
                        help="Path to pre-trained Wavelet checkpoint for fusion models")
    parser.add_argument('--xception_model_path', type=str, default=None,
                        help="Path to pre-trained Xception_128 checkpoint for WWXC fusion models")
    parser.add_argument('--convnext_model_path', type=str, default=None,
                        help="Path to pre-trained ConvNeXt_128 checkpoint for WWXC fusion models")
    parser.add_argument('--embed_dim', type=int, default=128,
                        help="Embedding dimension for fusion models")
    parser.add_argument('--num_heads', type=int, default=4,
                        help="Number of attention heads for MHA fusion")
    parser.add_argument('--dropout', type=float, default=0.1,
                        help="Dropout rate for fusion models")
    parser.add_argument('--fusion_type', type=str, default='cross_attention', choices=['cross_attention', 'self_attention', 'concat'],
                        help="Fusion strategy for MHA model")
    parser.add_argument('--freeze_base_models', action='store_true', default=True,
                        help="Freeze pre-trained base models during fusion training")
    parser.add_argument('--no_freeze_base_models', dest='freeze_base_models', action='store_false',
                        help="Do not freeze base models during fusion training")

    # Experiment & Logging
    parser.add_argument('--name', type=str, default='wang2020_experiment',
                        help="Name of the experiment run")
    parser.add_argument('--checkpoints_dir', type=str, default='./experiments',
                        help="Base directory for saving experiments and checkpoints")
    parser.add_argument('--log_freq', type=int, default=50,
                        help="Frequency of logging training status (in steps)")
    parser.add_argument('--loss_freq', type=int, default=400,
                        help="Frequency of saving loss summaries")
    parser.add_argument('--save_epoch_freq', type=int, default=1,
                        help="Frequency of saving checkpoints (in epochs)")
    parser.add_argument('--val_epoch_freq', type=int, default=1,
                        help="Frequency of running validation (in epochs)")

    # Parse args
    args = parser.parse_args()

    # Post-process list fields
    if isinstance(args.classes, str):
        args.classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    if isinstance(args.gpu_ids, str):
        if args.gpu_ids.strip() == '-1' or not args.gpu_ids.strip():
            args.gpu_ids = []
        else:
            args.gpu_ids = [int(id_.strip()) for id_ in args.gpu_ids.split(',') if id_.strip()]

    # Auto-enable wavelets if using a Wolter wavelet architecture
    if 'Wolter' in args.arch:
        args.compute_wavelets = True

    # Compatibility attributes required by models and dataloaders
    args.isTrain = True
    args.num_threads = args.num_workers
    args.cropSize = args.crop_size
    args.loadSize = args.image_size
    args.niter = args.epochs
    args.niter_decay = args.epochs_decay

    return args


def main():
    is_main = int(os.environ.get('RANK', 0)) == 0
    if is_main:
        print("=" * 70)
        print("        NOVEL DEEPFAKE DETECTION — TRAINING ENGINE")
        print("=" * 70)

    # 1. Parse command line arguments
    opt = parse_args()
    if is_main:
        print(f"Architecture : {opt.arch}")
        print(f"Dataset Root : {opt.dataroot}")
        print(f"Classes      : {opt.classes}")
        print(f"GPUs         : {opt.gpu_ids if opt.gpu_ids else 'CPU'}")
        print(f"Batch Size   : {opt.batch_size}")
        print(f"Learning Rate: {opt.lr}")

    # 2. Convert to structured Config & validate
    if is_main:
        print("\n[1/5] Validating configuration...")
    try:
        config = load_config(opt, validate=True, freeze=True)
        opt_clean = config_to_opt(config)
        # Re-apply list conversion for classes in case config_to_opt formatted it differently
        if isinstance(opt_clean.classes, str):
            opt_clean.classes = [c.strip() for c in opt_clean.classes.split(',') if c.strip()]
    except Exception as e:
        if is_main:
            print(f"\nConfiguration validation failed:\n{e}")
        sys.exit(1)

    # Instantiate runtime early so process group is ready and we know is_main
    from training.runtime import DistributedRuntime
    runtime = DistributedRuntime(opt_clean)

    # 3. Setup Experiment Manager
    if runtime.is_main:
        print("[2/5] Setting up experiment environment...")
    manager = ExperimentManager(base_dir=opt_clean.checkpoints_dir)
    experiment = manager.create(opt_clean.name, opt_clean)
    if runtime.is_main:
        print(f"  -> Experiment directory: {experiment.root_dir}")
        print(f"  -> Checkpoint directory: {experiment.checkpoint_dir}")
        print(f"  -> Metrics CSV log     : {experiment.metrics_csv_path}")
        print(f"  -> Steps CSV log       : {experiment.steps_csv_path}")

    from experiment.logger import ExperimentLogger
    experiment_logger = ExperimentLogger(experiment) if runtime.is_main else None

    # Point trainer checkpoint save directory to the experiment directory
    opt_clean.checkpoints_dir = experiment.checkpoint_dir
    opt_clean.name = ""  # Let CheckpointManager use the directory directly without appending extra subfolders

    # 4. Build DataLoaders
    if runtime.is_main:
        print("[3/5] Building dataloaders...")
    if opt_clean.arch in ['MHA_128', 'Fusion_128', 'Fusion_WWXC', 'MHA_WWXC']:
        train_loader = create_mha_dataloader(opt_clean)
    else:
        train_loader = create_dataloader(opt_clean)
    if runtime.is_main:
        print(f"  -> Training dataset size: {len(train_loader.dataset)} samples")

    val_loader = None
    val_root = getattr(opt_clean, 'val_root', None)
    if val_root and os.path.exists(val_root):
        if runtime.is_main:
            print("  -> Building validation dataloader...")
        val_opt = argparse.Namespace(**vars(opt_clean))
        val_opt.dataroot = val_root
        val_opt.isTrain = False
        val_opt.serial_batches = True
        if val_opt.arch in ['MHA_128', 'Fusion_128', 'Fusion_WWXC', 'MHA_WWXC']:
            val_loader = create_mha_dataloader(val_opt)
        else:
            val_loader = create_dataloader(val_opt)
        if runtime.is_main:
            print(f"  -> Validation dataset size: {len(val_loader.dataset)} samples")

    # 5. Build Model & Trainer
    if runtime.is_main:
        print(f"[4/5] Constructing model ({opt_clean.arch})...")
    model = build_model(opt_clean)

    if runtime.is_main:
        print("[5/5] Initializing Trainer...")
    trainer = Trainer(model, train_loader, opt_clean, val_loader=val_loader, runtime=runtime, experiment_logger=experiment_logger)

    if runtime.is_main:
        print("\n" + "=" * 70)
        print("STARTING TRAINING LOOP")
        print("=" * 70)
    try:
        trainer.fit(num_epochs=opt_clean.niter)
        if runtime.is_main:
            print("\n[SUCCESS] Training completed successfully!")
    except KeyboardInterrupt:
        if runtime.is_main:
            print("\n[INFO] Training interrupted by user.")
    except Exception as e:
        if runtime.is_main:
            print(f"\n[ERROR] Training failed with error: {e}")
        raise


if __name__ == "__main__":
    main()
