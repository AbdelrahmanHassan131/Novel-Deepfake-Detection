"""
Checkpoint Loader.

Loads a checkpoint file, detects the model architecture, builds the
correct model via the model registry, and restores weights.

Supports both the refactored checkpoint format (from CheckpointManager)
and the legacy format (``{'model': state_dict, ...}``).

Usage::

    loader = CheckpointLoader(checkpoint_path, device='cuda:0')
    model = loader.load()
    print(loader.metadata)
"""

import os
import torch


class CheckpointLoader:
    """
    Architecture-agnostic checkpoint loader.

    Inspects the checkpoint dict to determine which model to build,
    then instantiates it via the model registry and restores weights.

    Args:
        checkpoint_path (str): Path to the ``.pth`` checkpoint file.
        arch (str or None): Force a specific architecture name.
            If ``None``, the loader tries to detect it from the
            checkpoint metadata.
        device (str or torch.device): Target device.
        gpu_ids (list[int] or None): GPU IDs for the model.
    """

    def __init__(self, checkpoint_path, arch=None, device=None,
                 gpu_ids=None):
        is_special = (checkpoint_path is None or str(checkpoint_path).lower() in ('pretrained', 'none', 'default', 'scratch'))
        if not is_special and not os.path.isfile(checkpoint_path):
            raise FileNotFoundError(
                f'Checkpoint not found: {checkpoint_path}'
            )
        self.checkpoint_path = checkpoint_path
        self.forced_arch = arch
        self.device = device or ('cuda:0' if torch.cuda.is_available()
                                 else 'cpu')
        self.gpu_ids = gpu_ids if gpu_ids is not None else (
            [0] if torch.cuda.is_available() else []
        )
        self.is_pretrained_only = is_special
        self._checkpoint = None
        self._metadata = {}

    @property
    def metadata(self):
        """Return checkpoint metadata (epoch, model_name, etc.)."""
        return dict(self._metadata)

    def load(self, opt_overrides=None):
        """
        Load checkpoint and build the model.

        Args:
            opt_overrides (dict, optional): Additional attributes to
                set on the options object before model construction.

        Returns:
            A ``BaseModel`` subclass instance with weights restored
            and in eval mode.
        """
        if self.is_pretrained_only:
            arch = self.forced_arch
            if not arch:
                raise ValueError("Architecture (--arch) must be provided when evaluating without a checkpoint file.")
            print(f'[CheckpointLoader] Evaluating architecture without checkpoint file (using default/pretrained weights): {arch}')
            self._metadata = {
                'arch': arch,
                'epoch': 'pretrained',
                'checkpoint_path': 'Pretrained/Default Weights',
            }
            opt = self._build_opt(arch, opt_overrides)
            from models import build_model
            model = build_model(opt)
            opt.isTrain = False
            model.eval()
            return model

        # Load raw checkpoint
        self._checkpoint = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )

        # Detect architecture
        arch = self._detect_architecture()
        print(f'[CheckpointLoader] Detected architecture: {arch}')

        # Extract metadata
        self._metadata = self._extract_metadata(arch)

        # Build the model through the registry
        opt = self._build_opt(arch, opt_overrides)
        model = self._build_and_restore(opt, arch)

        return model

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _detect_architecture(self):
        """Determine which architecture the checkpoint belongs to."""
        if self.forced_arch is not None:
            return self.forced_arch

        ckpt = self._checkpoint

        # Refactored checkpoint format stores model_name
        if isinstance(ckpt, dict) and 'model_name' in ckpt:
            return self._model_name_to_arch(ckpt['model_name'])

        raise ValueError(
            'Cannot auto-detect model architecture from checkpoint. '
            'Pass --arch explicitly. '
            f'Checkpoint keys: {list(ckpt.keys()) if isinstance(ckpt, dict) else "not a dict"}'
        )

    @staticmethod
    def _model_name_to_arch(model_name):
        """Map a model's ``name()`` output to its registry key."""
        _NAME_MAP = {
            'Wang2020Raw': 'Wang2020Raw',
            'Wang2020_128': 'Wang2020_128',
            'WolterWaveletRaw': 'WolterWavelet2021Raw',
            'WolterWavelet128': 'WolterWavelet2021_128',
            'Fusion_128': 'Fusion_128',
            'MHA_128': 'MHA_128',
            'XceptionRaw': 'XceptionRaw',
            'Xception_128': 'Xception_128',
            'ConvNeXtRaw': 'ConvNeXtRaw',
            'ConvNeXt_128': 'ConvNeXt_128',
            'Fusion_WWXC': 'Fusion_WWXC',
            'MHA_WWXC': 'MHA_WWXC',
            # Also accept direct registry names
            'Wang2020Raw': 'Wang2020Raw',
            'WolterWavelet2021Raw': 'WolterWavelet2021Raw',
            'WolterWavelet2021_128': 'WolterWavelet2021_128',
        }
        if model_name in _NAME_MAP:
            return _NAME_MAP[model_name]
        # Fall through: try as-is (may already be a registry key)
        return model_name

    def _extract_metadata(self, arch):
        """Extract useful metadata from the checkpoint."""
        ckpt = self._checkpoint
        if not isinstance(ckpt, dict):
            return {'arch': arch}

        return {
            'arch': arch,
            'epoch': ckpt.get('epoch', ckpt.get('total_steps', None)),
            'best_metric': ckpt.get('best_metric', None),
            'global_step': ckpt.get('global_step',
                                    ckpt.get('total_steps', None)),
            'model_name': ckpt.get('model_name', arch),
            'checkpoint_path': self.checkpoint_path,
            'checkpoint_size_mb': os.path.getsize(
                self.checkpoint_path) / (1024 * 1024),
        }

    def _build_opt(self, arch, opt_overrides=None):
        """
        Build a minimal options namespace for model construction.

        The model trainers expect an ``opt`` object with various
        attributes.  We construct one with evaluation defaults.
        """
        import argparse
        opt = argparse.Namespace()

        # Core settings
        opt.arch = arch
        opt.isTrain = True  # Set True initially to avoid auto-load
        opt.continue_train = False
        opt.gpu_ids = self.gpu_ids
        opt.init_gain = 0.02
        opt.optim = 'adam'
        opt.lr = 0.0001
        opt.beta1 = 0.9

        # Checkpoint settings — point to a dummy dir so the legacy
        # load_networks doesn't interfere; we handle loading ourselves.
        ckpt_path_str = self.checkpoint_path if (isinstance(self.checkpoint_path, str) and not self.is_pretrained_only) else './dummy/eval.pth'
        opt.checkpoints_dir = os.path.dirname(
            os.path.dirname(ckpt_path_str))
        opt.name = os.path.basename(
            os.path.dirname(ckpt_path_str)) or 'eval'
        opt.epoch = 'latest'

        # Data & transform settings required by datasets and resize transforms
        opt.no_crop = False
        opt.no_flip = True
        opt.no_resize = False
        opt.cropSize = 224
        opt.loadSize = 256
        opt.rz_interp = ['bilinear']
        opt.blur_prob = 0.0
        opt.blur_sig = [0.5]
        opt.jpg_prob = 0.0
        opt.jpg_method = ['cv2']
        opt.jpg_qual = [75]
        opt.data_aug = False
        opt.classes = ['0_real', '1_fake']
        opt.mode = 'binary'
        opt.serial_batches = True
        opt.class_bal = False
        opt.num_threads = 0

        # Data settings (required by some model __init__ paths)
        opt.compute_wavelets = arch in (
            'WolterWavelet2021Raw', 'WolterWavelet2021_128',
            'Fusion_128', 'MHA_128',
            'Fusion_WWXC', 'MHA_WWXC',
        )
        opt.wavelet_level = 3

        # Fusion / MHA models need base model paths
        if arch in ('Fusion_128', 'MHA_128'):
            opt.rgb_model_path = getattr(opt, 'rgb_model_path', '')
            opt.wavelet_model_path = getattr(opt, 'wavelet_model_path', '')

        # WWXC models need all 4 base model paths
        if arch in ('Fusion_WWXC', 'MHA_WWXC'):
            opt.rgb_model_path = getattr(opt, 'rgb_model_path', '')
            opt.wavelet_model_path = getattr(opt, 'wavelet_model_path', '')
            opt.xception_model_path = getattr(opt, 'xception_model_path', '')
            opt.convnext_model_path = getattr(opt, 'convnext_model_path', '')

        # Apply user overrides
        if opt_overrides:
            for key, value in opt_overrides.items():
                setattr(opt, key, value)

        return opt

    def _build_and_restore(self, opt, arch):
        """Build model via registry and restore checkpoint weights."""
        from models import build_model

        # Build the model (isTrain=True prevents auto-load in __init__)
        model = build_model(opt)

        # Now set to eval
        opt.isTrain = False

        # Restore weights
        ckpt = self._checkpoint
        state_dict = self._extract_state_dict(ckpt)

        # Get the raw model (unwrap DDP if needed)
        raw_model = model.model
        if hasattr(raw_model, 'module'):
            raw_model = raw_model.module

        raw_model.load_state_dict(state_dict, strict=True)
        print(f'[CheckpointLoader] Weights restored successfully')

        model.eval()
        return model

    @staticmethod
    def _extract_state_dict(checkpoint):
        """Extract model state_dict from various checkpoint formats."""
        if not isinstance(checkpoint, dict):
            return checkpoint

        # Refactored format: model_state_dict
        if 'model_state_dict' in checkpoint:
            return checkpoint['model_state_dict']

        # Legacy format: model
        if 'model' in checkpoint:
            return checkpoint['model']

        # Direct state_dict
        return checkpoint
