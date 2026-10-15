import os
from functools import partial
import torch
import numpy as np
from torchvision import transforms
from .base_dataset import BaseDataset
from ..transforms.augmentations import data_augment
from ..transforms.resize import custom_resize
from ..transforms.identity import identity_image
from ..transforms.drct_transforms import DRCTValTransform, DRCTTrainTransform, DRCT_IMAGE_SIZE
from ..wavelets.backends import create_wavelet_backend


class FusionDataset(BaseDataset):
    """
    Dataset that returns both RGB images and Wavelet packets.
    Returns: (rgb_tensor, wavelet_tensor, label)
    Represents behavior used by: MHA_128, Fusion_128.

    Supports three backends controlled by ``opt.wavelet_backend``:
        - 'cpu' (default): RGB images loaded, augmented; wavelets computed on CPU.
        - 'gpu': RGB images loaded, augmented on CPU; wavelets computed on GPU.
        - 'precomputed': Wavelet tensors loaded from precomputed .npy files.

    Branch-aware Preprocessing:
        When the RGB expert is DRCT_128, applies versioned DRCT preprocessing
        (longest-edge resize to 224, symmetric zero padding, ImageNet normalization)
        to the RGB branch, while preserving separate wavelet branch preprocessing.
    """
    def __init__(self, opt, root):
        # --- Resolve backend ---
        precomputed_dir = getattr(opt, 'precomputed_dir', None)
        backend_name = getattr(opt, 'wavelet_backend', None)

        if backend_name is None:
            if precomputed_dir:
                backend_name = 'precomputed'
            else:
                backend_name = 'cpu'
            opt.wavelet_backend = backend_name

        self._backend = create_wavelet_backend(opt)

        # --- Detect RGB expert architecture from metadata ---
        self.rgb_is_drct = False
        rgb_arch = getattr(opt, 'rgb_expert_arch', None)
        if rgb_arch == 'DRCT_128':
            self.rgb_is_drct = True
        elif getattr(opt, '_expert_metadata', {}).get('rgb_model', {}).get('arch') == 'DRCT_128':
            self.rgb_is_drct = True
        external_input_sz = None
        if getattr(opt, 'rgb_model_path', None) and os.path.isfile(opt.rgb_model_path):
            try:
                ckpt = torch.load(opt.rgb_model_path, map_location='cpu', weights_only=False)
                arch_found = (
                    ckpt.get('protocol', {}).get('options', {}).get('arch')
                    or ckpt.get('protocol', {}).get('options', {}).get('model_name')
                    or ckpt.get('model_name')
                    or ckpt.get('arch')
                )
                if arch_found == 'DRCT_128':
                    self.rgb_is_drct = True
                    proto_opts = ckpt.get('protocol', {}).get('options', {})
                    external_input_sz = proto_opts.get('drct_crop_size') or proto_opts.get('cropSize') or proto_opts.get('crop_size')
            except Exception:
                pass

        expert_meta = getattr(opt, '_expert_metadata', {}).get('rgb_model', {})
        drct_input_sz = expert_meta.get('input_size') or external_input_sz or getattr(opt, 'drct_crop_size', DRCT_IMAGE_SIZE)

        if self.rgb_is_drct:
            if drct_input_sz != 224:
                raise ValueError(
                    f"DRCT visual encoder requires input resolution 224x224 (got {drct_input_sz}). "
                    "CLIP ViT positional embeddings do not support alternative resolutions without interpolation."
                )
            if opt.isTrain and not getattr(opt, 'no_flip', False):
                self.drct_transform = DRCTTrainTransform(size=drct_input_sz)
            else:
                self.drct_transform = DRCTValTransform(size=drct_input_sz)
        else:
            self.drct_transform = None

        # --- Precomputed root (only for precomputed backend) ---
        self._precomputed_root = None
        if self._backend.name == 'precomputed':
            if opt.isTrain:
                raise ValueError('Precomputed fusion training can mismatch augmented RGB and wavelets. Use cpu or gpu wavelets.')
            if not precomputed_dir:
                raise ValueError(
                    "opt.precomputed_dir must be set when using the "
                    "'precomputed' wavelet backend."
                )
            dataroot = os.path.normpath(opt.dataroot)
            root_norm = os.path.normpath(root)
            rel = os.path.relpath(root_norm, dataroot)
            self._precomputed_root = os.path.normpath(
                os.path.join(precomputed_dir, rel)
            )

        # --- Image transforms for Wavelet and standard RGB branches ---
        if opt.isTrain:
            crop_func = transforms.RandomCrop(opt.cropSize)
        elif opt.no_crop:
            crop_func = transforms.Lambda(identity_image)
        else:
            crop_func = transforms.CenterCrop(opt.cropSize)

        if opt.isTrain and not opt.no_flip:
            flip_func = transforms.RandomHorizontalFlip()
        else:
            flip_func = transforms.Lambda(identity_image)

        if not opt.isTrain and opt.no_resize:
            rz_func = transforms.Lambda(identity_image)
        else:
            rz_func = transforms.Lambda(partial(custom_resize, opt=opt))

        self.image_transform = transforms.Compose([
            rz_func,
            transforms.Lambda(partial(data_augment, opt=opt)),
            crop_func,
            flip_func,
        ])

        self.rgb_normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # --- GPU backend needs ToTensor before wavelet computation ---
        self._needs_to_tensor = (self._backend.name == 'gpu')

        super().__init__(opt, root, transform=None)

    # ------------------------------------------------------------------
    # Precomputed helpers
    # ------------------------------------------------------------------

    def _get_npy_path(self, image_path):
        """Map an image file path to its precomputed ``.npy`` counterpart."""
        rel = os.path.relpath(image_path, self.root)
        npy_rel = os.path.splitext(rel)[0] + '.npy'
        return os.path.join(self._precomputed_root, npy_rel)

    # ------------------------------------------------------------------
    # __getitem__
    # ------------------------------------------------------------------

    def __getitem__(self, index):
        path, target = self.samples[index]
        raw_img = self.loader(path)

        if self.rgb_is_drct:
            rgb_tensor = self.drct_transform(raw_img)
            wav_img = self.image_transform(raw_img)
            if self._backend.name == 'precomputed':
                npy_path = self._get_npy_path(path)
                wavelet_tensor = self._backend(npy_path)
            elif self._needs_to_tensor:
                img_array = np.array(wav_img, dtype=np.float32)
                if img_array.ndim == 3 and img_array.shape[2] == 3:
                    wavelet_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
                else:
                    wavelet_tensor = torch.from_numpy(img_array)
            else:
                wavelet_tensor = self._backend(wav_img)
        else:
            img = self.image_transform(raw_img) if self.image_transform is not None else raw_img
            rgb_tensor = self.rgb_normalize(img)
            if self._backend.name == 'precomputed':
                npy_path = self._get_npy_path(path)
                wavelet_tensor = self._backend(npy_path)
            elif self._needs_to_tensor:
                img_array = np.array(img, dtype=np.float32)
                if img_array.ndim == 3 and img_array.shape[2] == 3:
                    wavelet_tensor = torch.from_numpy(img_array.transpose(2, 0, 1))
                else:
                    wavelet_tensor = torch.from_numpy(img_array)
            else:
                wavelet_tensor = self._backend(img)

        return rgb_tensor, wavelet_tensor, target
