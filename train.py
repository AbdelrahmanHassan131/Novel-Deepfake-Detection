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
import json
from datetime import datetime

# Notebook shells pipe stdout: announce worker startup BEFORE expensive imports,
# and flush subsequent progress instead of buffering it until process exit.
if __name__ == '__main__':
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, 'reconfigure'):
            stream.reconfigure(line_buffering=True, write_through=True)
    print(f"[startup rank={os.environ.get('RANK', '0')} pid={os.getpid()}] Loading PyTorch...", flush=True)
import torch

if __name__ == '__main__':
    print(f"[startup rank={os.environ.get('RANK', '0')}] PyTorch loaded; loading project modules...", flush=True)

from config import load_config, config_to_opt, ConfigValidator
from config.defaults import DATA_DEFAULTS, TRAINING_DEFAULTS, MODEL_DEFAULTS, RUNTIME_DEFAULTS, EXPERIMENT_DEFAULTS, WAVELET_DEFAULTS, AUGMENTATION_DEFAULTS
from models import build_model, get_registered_models
from data.loaders.dataloader_factory import create_dataloader, create_mha_dataloader
from training import Trainer, seed_everything
from experiment import ExperimentManager

if __name__ == '__main__':
    print(f"[startup rank={os.environ.get('RANK', '0')}] Project imports complete.", flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description="Train Deepfake Detection Models")

    # Model / Architecture
    parser.add_argument('--arch', type=str, default='Wang2020Raw',
                        help=f"Model architecture to train. Available: {get_registered_models()}")
    parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction, default=MODEL_DEFAULTS['pretrained'],
                        help="Use pretrained weights for backbone networks")
    parser.add_argument('--backbone_weights', type=str, default=None,
                        help="Path to local backbone weights for offline initialization without downloads")
    parser.add_argument('--rgb_head_type', choices=['128d', 'linear'], default='128d',
                        help="RGB classifier head architecture: '128d' (legacy 128D MLP) or 'linear' (direct linear probe)")
    parser.add_argument('--rgb_dropout', type=float, default=0.5,
                        help="Dropout rate for the RGB classifier head (defaults to 0.5 for Wang2020 legacy compatibility)")
    parser.add_argument('--fine_tune_policy', choices=['full', 'head_only', 'layer4_and_head'], default='full',
                        help="Fine-tuning parameter freezing policy ('full', 'head_only', or 'layer4_and_head')")
    parser.add_argument('--backbone_lr_mult', type=float, default=1.0,
                        help="Learning rate scaling multiplier for backbone layers relative to head")
    parser.add_argument('--bn_policy', choices=['train', 'frozen'], default='train',
                        help="BatchNorm policy: 'train' (updates running stats) or 'frozen' (freezes running stats in eval mode)")
    parser.add_argument('--decay_bias_norm', action=argparse.BooleanOptionalAction, default=False,
                        help="Whether to apply weight decay to 1D bias and normalization layers")
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
    parser.add_argument('--val_batch_size', type=int, default=None,
                        help="Batch size for validation DataLoader (defaults to --batch_size if not specified)")
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
    parser.add_argument('--crop_policy', choices=['scale_and_crop', 'random_resized_crop', 'patch_crop'], default='scale_and_crop',
                        help="Cropping strategy: scale_and_crop (legacy default), random_resized_crop, or patch_crop")

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
                        help="Comma-separated JPEG quality values (e.g., '30,75' or '50,60,70,80,90,95')")
    parser.add_argument('--rz_interp', type=str, default='bilinear',
                        help="Resize interpolation method (bilinear, bicubic, lanczos, nearest)")
    parser.add_argument('--aug_recipe', choices=['legacy', 'rgb_v1', 'custom'], default='legacy',
                        help="Versioned augmentation recipe: 'legacy' (unaugmented baseline), 'rgb_v1' (moderate blur 0.5 + jpeg 0.5 preset), or 'custom'")

    # Wavelet parameters
    parser.add_argument('--wavelet_backend', type=str, default=WAVELET_DEFAULTS['backend'], choices=['cpu', 'gpu', 'precomputed'],
                        help="Wavelet computation backend: cpu, gpu, or precomputed")
    parser.add_argument('--wavelet_type', type=str, default=WAVELET_DEFAULTS['wavelet_type'],
                        help="Wavelet family name (e.g. haar, db2)")
    parser.add_argument('--wavelet_level', type=int, default=WAVELET_DEFAULTS['level'],
                        help="Wavelet decomposition level")
    parser.add_argument('--wavelet_mode', type=str, default=WAVELET_DEFAULTS['mode'],
                        help="Signal extension mode")
    parser.add_argument('--use_log_packets', action=argparse.BooleanOptionalAction, default=WAVELET_DEFAULTS['log_packets'],
                        help="Apply log scaling to wavelet packets")

    # Training Hyperparameters
    parser.add_argument('--wavelet_log_mode', choices=['signed_log1p', 'legacy'], default='signed_log1p',
                        help='Stable signed log1p for new runs; legacy reproduces old checkpoints')
    parser.add_argument('--epochs', '--niter', dest='epochs', type=int, default=TRAINING_DEFAULTS['epochs'],
                        help="Number of epochs to train")
    parser.add_argument('--epochs_decay', '--niter_decay', dest='epochs_decay', type=int, default=TRAINING_DEFAULTS['epochs_decay'],
                        help="Number of epochs to linearly decay learning rate to zero")
    parser.add_argument('--lr', type=float, default=TRAINING_DEFAULTS['learning_rate'],
                        help="Initial learning rate")
    parser.add_argument('--optim', dest='optimizer', type=str, default=TRAINING_DEFAULTS['optimizer'], choices=['adam', 'sgd', 'adamw'],
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
    parser.add_argument('--resume_checkpoint', type=str, default=None,
                        help="Explicit full-protocol checkpoint to resume. Required with --continue_train.")
    parser.add_argument('--epoch', type=str, default='latest',
                        help="Which epoch to load when continue_train is set (e.g., 'latest' or '10')")
    parser.add_argument('--use_amp', action='store_true', default=TRAINING_DEFAULTS['use_amp'],
                        help="Enable Automatic Mixed Precision (AMP)")
    parser.add_argument('--amp_dtype', choices=['fp16', 'bf16'], default='fp16',
                        help="AMP precision dtype: fp16 (recommended for T4) or bf16 (supported Ampere/L4 only)")
    parser.add_argument('--val_precision', choices=['fp32', 'amp', 'fp16', 'bf16'], default='fp32',
                        help="Validation precision: fp32 (reference), amp, fp16, or bf16")
    parser.add_argument('--monitor_metric', type=str, default=TRAINING_DEFAULTS.get('monitor_metric', 'auc'),
                        choices=['auc', 'balanced_accuracy', 'accuracy', 'source_macro_auc', 'worst_source_recall_05'],
                        help="Validation metric monitored for saving best.pth (auc, balanced_accuracy, accuracy, source_macro_auc, or worst_source_recall_05)")
    parser.add_argument('--early_stopping', action=argparse.BooleanOptionalAction, default=False,
                        help="Enable early stopping based on monitored validation metric")
    parser.add_argument('--early_stopping_patience', type=int, default=5,
                        help="Number of validation checks without improvement before stopping")
    parser.add_argument('--early_stopping_min_delta', type=float, default=0.0,
                        help="Minimum change in monitored metric to qualify as an improvement")
    parser.add_argument('--early_stopping_min_epochs', type=int, default=0,
                        help="Minimum number of epochs before early stopping can trigger")
    parser.add_argument('--eligible_sources', type=str, default=None,
                        help="Comma-separated list of eligible dataset sources for source-macro AUC calculation")
    parser.add_argument('--allow_aggregate_sources', action='store_true', default=False,
                        help="Allow aggregate/unknown source metadata (e.g. diffgan) for engineering diagnostic runs")
    parser.add_argument('--allow_source_overlap', action='store_true', default=False,
                        help="Allow training and development sources to overlap (disables strict held-out check)")
    parser.add_argument('--grad_accum_steps', type=int, default=1,
                        help="Number of gradient accumulation steps before optimizer step (simulates larger batch size on single/dual GPU)")

    # Runtime & GPU
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help="Comma-separated list of GPU IDs to use (e.g., '0' or '0,1'). Use '-1' for CPU.")
    parser.add_argument('--num_workers', '--num_threads', dest='num_workers', type=int, default=0,
                        help="Number of data loading threads (default 0 for Windows stability)")
    parser.add_argument('--val_num_workers', type=int, default=None,
                        help="Number of workers for validation DataLoader (defaults to --num_workers if not specified)")
    parser.add_argument('--pin_memory', action=argparse.BooleanOptionalAction, default=True,
                        help="Pin DataLoader memory to accelerate GPU transfer")
    parser.add_argument('--prefetch_factor', type=int, default=2,
                        help="Number of batches loaded in advance by each worker (when num_workers > 0)")
    parser.add_argument('--persistent_workers', action=argparse.BooleanOptionalAction, default=True,
                        help="Keep worker processes alive across DataLoader iterations")
    parser.add_argument('--channels_last', action=argparse.BooleanOptionalAction, default=False,
                        help="Use channels_last (NHWC) memory format for faster ResNet computation on Tensor Cores")
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
    parser.add_argument('--fusion_type', type=str, default='token_attention', choices=['token_attention', 'gated', 'concat', 'cross_attention', 'self_attention'],
                        help="Fusion strategy for MHA model")
    parser.add_argument('--freeze_base_models', action='store_true', default=True,
                        help="Freeze pre-trained base models during fusion training")
    parser.add_argument('--no_freeze_base_models', dest='freeze_base_models', action='store_false',
                        help="Do not freeze base models during fusion training")

    # Experiment & Logging
    parser.add_argument('--name', type=str, default='wang2020_experiment',
                        help="Name of the experiment run")
    parser.add_argument('--run_id', type=str, default=None,
                        help="Explicit immutable run/experiment identifier (e.g. 'seed42' or timestamp)")
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

    parser.add_argument('--manifest', help='CSV with explicit labels and source groups')
    parser.add_argument('--manifest_split', default='train')
    parser.add_argument('--val_manifest')
    parser.add_argument('--val_manifest_split', default='dev')
    parser.add_argument('--audit_hashes', action='store_true')
    parser.add_argument('--require_verified_manifest', action=argparse.BooleanOptionalAction, default=True,
                        help='Require preparation audit gate to pass before training')
    parser.add_argument('--allow_folder_training', action='store_true', help='Smoke/legacy experiments only: group independence cannot be audited')
    parser.add_argument('--noise_prob', type=float, default=0.0)
    parser.add_argument('--noise_std', type=float, nargs=2, default=[0.0, 3.0], help='Noise sigma in 0-255 pixel units')
    parser.add_argument('--downscale_prob', type=float, default=0.0)
    parser.add_argument('--downscale_range', type=float, nargs=2, default=[0.5, 1.0])
    # Parse args
    args = parser.parse_args()
    # Preserve explicit zero/default-valued overrides when resolving a preset.
    args._explicit_options = [token[2:].split('=', 1)[0] for token in sys.argv[1:]
                              if token.startswith('--')]

    # Post-process list fields
    if isinstance(args.classes, str):
        args.classes = [c.strip() for c in args.classes.split(',') if c.strip()]
    if isinstance(args.eligible_sources, str):
        args.eligible_sources = [s.strip() for s in args.eligible_sources.split(',') if s.strip()]
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
    print(f"[rank={os.environ.get('RANK', '0')}] Parsing options and loading model registry...", flush=True)
    opt = parse_args()
    from data.transforms.augmentations import resolve_augmentation_recipe
    opt = resolve_augmentation_recipe(opt)
    from training.runtime.profiler import PhaseProfiler
    startup_profiler = PhaseProfiler()
    startup_profiler.start_phase('configuration_and_manifest_preflight')
    if is_main:
        print(f"Architecture : {opt.arch}")
        print(f"Dataset Root : {opt.dataroot}")
        print(f"Classes      : {opt.classes}")
        print(f"GPUs         : {opt.gpu_ids if opt.gpu_ids else 'CPU'}")
        print(f"Batch Size   : {opt.batch_size} (Accum steps: {getattr(opt, 'grad_accum_steps', 1)})")
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

    from data.manifest import read_manifest, audit_rows, PROTECTED_SPLITS, verify_manifest_gate
    if not opt_clean.manifest and not opt_clean.allow_folder_training:
        raise ValueError('Use --manifest for auditable training, or --allow_folder_training for a smoke experiment.')
    if opt_clean.manifest:
        require_hashes = getattr(opt_clean, 'audit_hashes', False)
        if getattr(opt, 'require_verified_manifest', True):
            if is_main:
                print(f"Verifying preparation audit gate for {opt_clean.manifest}...")
            verify_manifest_gate(opt_clean.manifest, enforce_class_coverage=True, require_hashes=require_hashes)
            if opt_clean.val_manifest and opt_clean.val_manifest != opt_clean.manifest:
                if is_main:
                    print(f"Verifying preparation audit gate for val_manifest: {opt_clean.val_manifest}...")
                verify_manifest_gate(opt_clean.val_manifest, enforce_class_coverage=True, require_hashes=require_hashes)
        if opt_clean.manifest_split in PROTECTED_SPLITS or opt_clean.val_manifest_split in PROTECTED_SPLITS:
            raise ValueError('Test splits must not be used for training or checkpoint selection.')
        if opt_clean.manifest == opt_clean.val_manifest and opt_clean.manifest_split == opt_clean.val_manifest_split:
            raise ValueError('Training and development split names must differ when using the same manifest.')
        if not opt_clean.val_manifest:
            opt_clean.val_manifest = opt_clean.manifest
        print(f"[rank={os.environ.get('RANK', '0')}] Checking training/development metadata and image paths; "
              f"content rehashing={require_hashes}. No model batches have started yet.", flush=True)
        rows = read_manifest(opt_clean.manifest, opt_clean.dataroot, opt_clean.manifest_split, progress_every=10000)
        dev_rows = read_manifest(opt_clean.val_manifest, opt_clean.val_root or opt_clean.dataroot,
                                 opt_clean.val_manifest_split, progress_every=10000)
        # Ensure dev_rows have split distinct from manifest_split for cross-manifest overlap detection
        tagged_dev_rows = []
        for r in dev_rows:
            r_copy = dict(r)
            if r_copy['split'] == opt_clean.manifest_split:
                r_copy['split'] = 'dev'
            tagged_dev_rows.append(r_copy)
        report = audit_rows(rows + tagged_dev_rows, hash_files=require_hashes, progress_every=10000)
        if not report['passed']:
            raise ValueError('Dataset audit failed: ' + '; '.join(report['errors'][:20]))
        if {r['label'] for r in rows} != {0, 1} or {r['label'] for r in dev_rows} != {0, 1}:
            raise ValueError('Training and development each require real and fake samples.')
        from training.validator import verify_source_readiness
        verify_source_readiness(
            train_rows=rows,
            dev_rows=dev_rows,
            eligible_sources=getattr(opt_clean, 'eligible_sources', None),
            monitor_metric=getattr(opt_clean, 'monitor_metric', 'auc'),
            allow_aggregate_sources=getattr(opt_clean, 'allow_aggregate_sources', False),
            allow_source_overlap=getattr(opt_clean, 'allow_source_overlap', False),
        )
        print(f"[rank={os.environ.get('RANK', '0')}] Dataset audit passed: "
              f"{len(rows):,} train / {len(dev_rows):,} development rows.", flush=True)
    startup_profiler.end_phase('configuration_and_manifest_preflight')
    # Seed BEFORE constructing datasets and model weights, not just the training loop.
    if opt_clean.seed is not None:
        seed_everything(opt_clean.seed, deterministic=opt_clean.deterministic)

    # Instantiate runtime early so process group is ready and we know is_main
    from training.runtime import DistributedRuntime
    runtime = DistributedRuntime(opt_clean)
    startup_profiler.rank = runtime.rank
    startup_profiler.world_size = runtime.world_size
    startup_profiler.device = runtime.device
    startup_profiler.use_cuda = runtime.device.type == 'cuda'

    # 3. Setup Experiment Manager.  A resumed run must use the original
    # checkpoint directory; creating a timestamped experiment first would make
    # --continue_train silently start from scratch.
    if runtime.is_main:
        print("[2/5] Setting up experiment environment...")
    manager = ExperimentManager(base_dir=opt_clean.checkpoints_dir)
    if opt_clean.continue_train:
        if not opt_clean.resume_checkpoint:
            raise ValueError('--continue_train requires --resume_checkpoint pointing to a full-protocol checkpoint.')
        resume_path = os.path.abspath(opt_clean.resume_checkpoint)
        if not os.path.isfile(resume_path):
            raise FileNotFoundError(f'Resume checkpoint not found: {resume_path}')
        checkpoint_dir = os.path.dirname(resume_path)
        experiment = manager.load(os.path.dirname(checkpoint_dir))
    else:
        resume_path = None
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
    # Attach cached rows after saving the options file, and remove them before
    # spawning workers. Only dataset.records should carry the image inventory.
    if opt_clean.manifest:
        opt_clean._manifest_records = rows
        opt_clean._validation_records = dev_rows
    startup_profiler.start_phase('loader_construction')
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
    if opt_clean.val_manifest or (val_root and os.path.exists(val_root)):
        if runtime.is_main:
            print("  -> Building validation dataloader...")
        val_opt = argparse.Namespace(**vars(opt_clean))
        val_opt.dataroot = val_root or opt_clean.dataroot
        val_opt.manifest = opt_clean.val_manifest
        val_opt.manifest_split = opt_clean.val_manifest_split
        val_opt.class_bal = False
        val_opt.isTrain = False
        val_opt.serial_batches = True
        if hasattr(opt_clean, '_validation_records'):
            val_opt._manifest_records = opt_clean._validation_records
        if val_opt.arch in ['MHA_128', 'Fusion_128', 'Fusion_WWXC', 'MHA_WWXC']:
            val_loader = create_mha_dataloader(val_opt)
        else:
            val_loader = create_dataloader(val_opt)
        if runtime.is_main:
            print(f"  -> Validation dataset size: {len(val_loader.dataset)} samples")

    # 5. Build Model & Trainer
    for namespace in (opt_clean, val_opt if val_loader is not None else opt_clean):
        for key in ('_manifest_records', '_validation_records'):
            vars(namespace).pop(key, None)
    startup_profiler.end_phase('loader_construction')
    startup_profiler.start_phase('pretrained_and_model_construction')
    if runtime.is_main:
        print(f"[4/5] Constructing model ({opt_clean.arch})...")
        if opt_clean.pretrained and not resume_path:
            print('Pretrained backbone requested: loading cached/local weights or downloading if absent.', flush=True)
    model = build_model(opt_clean)
    startup_profiler.end_phase('pretrained_and_model_construction')
    if runtime.is_main and hasattr(model, 'head_class_name'):
        print(f"  -> Fusion head class   : {model.head_class_name}")
        print(f"  -> Head param count    : {getattr(model, 'head_param_count', 0):,}")

    if runtime.is_main:
        print("[5/5] Initializing Trainer...")
    trainer = Trainer(model, train_loader, opt_clean, val_loader=val_loader, runtime=runtime, experiment_logger=experiment_logger)
    trainer.profiler = startup_profiler
    if resume_path:
        trainer.resume_training(resume_path)

    if runtime.is_main:
        print("\n" + "=" * 70)
        print("STARTING TRAINING LOOP")
        print("=" * 70)
    try:
        total_epochs = opt_clean.niter + getattr(opt_clean, 'niter_decay', 0)
        trainer.fit(num_epochs=total_epochs)
        if runtime.is_main:
            print("\n[SUCCESS] Training completed successfully!")
            manifest_path = os.path.join(experiment.root_dir, 'run_manifest.json')
            if os.path.isfile(manifest_path):
                with open(manifest_path, 'r', encoding='utf-8') as f:
                    m_data = json.load(f)
                m_data['status'] = 'completed'
                m_data['completed_at'] = datetime.now().isoformat()
                if hasattr(model, 'head_class_name'):
                    m_data['head_class'] = model.head_class_name
                    m_data['head_params'] = getattr(model, 'head_param_count', 0)
                with open(manifest_path, 'w', encoding='utf-8') as f:
                    json.dump(m_data, f, indent=2)
    except KeyboardInterrupt:
        if runtime.is_main:
            print("\n[INFO] Training interrupted by user.")
        sys.exit(130)
    except Exception as e:
        if runtime.is_main:
            print(f"\n[ERROR] Training failed with error: {e}")
        raise



if __name__ == "__main__":
    main()
