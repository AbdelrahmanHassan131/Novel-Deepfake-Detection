"""Symmetric Forensic CLIP Preprocessing and Augmentations (clip_forensic_v1).

Designed specifically to eliminate dataset/format/resolution shortcuts while preserving
native OpenAI CLIP visual statistics and high-frequency generative/VAE artifacts:
  - Uses native OpenAI CLIP mean/std and bicubic interpolation.
  - Mixes native-resolution random crops (preserving high-frequency artifacts without
    destructive downscaling) with random resized crops (handling varied face crops).
  - Applies identical stochastic JPEG, Gaussian blur, downscale-upscale resampling,
    and mild color jitter symmetrically to both real and fake images during training.
"""
from random import random, uniform, choice
from typing import Sequence, Tuple, Union
import numpy as np
import torch
from PIL import Image, ImageEnhance
import torchvision.transforms.functional as TF

from data.transforms.augmentations import gaussian_blur, jpeg_from_key

CLIP_FORENSIC_VERSION = "clip_forensic_v1"
CLIP_IMAGE_SIZE = 224
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.26862954]


def _to_pil_rgb(img: Union[Image.Image, np.ndarray, torch.Tensor]) -> Image.Image:
    if isinstance(img, Image.Image):
        return img.convert("RGB")
    if isinstance(img, torch.Tensor):
        if img.dim() == 3:
            arr = img.detach().cpu().permute(1, 2, 0).numpy()
            if arr.dtype != np.uint8:
                arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
            return Image.fromarray(arr, mode="RGB")
        raise ValueError(f"Unsupported tensor shape: {tuple(img.shape)}")
    if isinstance(img, np.ndarray):
        arr = img
        if arr.dtype != np.uint8:
            arr = np.clip(arr * (255.0 if arr.max() <= 1.0 else 1.0), 0, 255).astype(np.uint8)
        if arr.ndim == 2:
            return Image.fromarray(arr).convert("RGB")
        if arr.shape[2] == 3:
            return Image.fromarray(arr, mode="RGB")
        if arr.shape[2] == 4:
            return Image.fromarray(arr[:, :, :3], mode="RGB")
        raise ValueError(f"Unsupported numpy image shape: {arr.shape}")
    raise TypeError(f"Unsupported image input type: {type(img)}")


class CLIPForensicValTransform:
    """Deterministic native OpenAI CLIP evaluation transform (bicubic resize + center crop)."""

    def __init__(self, size: int = CLIP_IMAGE_SIZE, mean=None, std=None):
        self.size = int(size)
        self.mean = list(mean) if mean is not None else list(CLIP_MEAN)
        self.std = list(std) if std is not None else list(CLIP_STD)
        self.version = CLIP_FORENSIC_VERSION

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        pil_img = _to_pil_rgb(img)
        resized = TF.resize(pil_img, self.size, interpolation=TF.InterpolationMode.BICUBIC)
        cropped = TF.center_crop(resized, [self.size, self.size])
        tensor = TF.to_tensor(cropped)
        return TF.normalize(tensor, mean=self.mean, std=self.std)


class CLIPForensicTrainTransform:
    """Symmetric forensic training transform to break compression, sharpness, and lighting shortcuts."""

    def __init__(
        self,
        size: int = CLIP_IMAGE_SIZE,
        native_crop_prob: float = 0.6,
        jpg_prob: float = 0.8,
        jpg_qual: Sequence[int] = (50, 60, 70, 75, 80, 85, 90, 95),
        jpg_method: Sequence[str] = ("cv2", "pil"),
        blur_prob: float = 0.35,
        blur_sig: Tuple[float, float] = (0.1, 2.0),
        downscale_prob: float = 0.35,
        downscale_range: Tuple[float, float] = (0.55, 1.0),
        color_jitter_prob: float = 0.3,
        p_flip: float = 0.5,
        mean=None,
        std=None,
    ):
        self.size = int(size)
        self.native_crop_prob = float(native_crop_prob)
        self.jpg_prob = float(jpg_prob)
        self.jpg_qual = list(jpg_qual)
        self.jpg_method = list(jpg_method)
        self.blur_prob = float(blur_prob)
        self.blur_sig = tuple(blur_sig)
        self.downscale_prob = float(downscale_prob)
        self.downscale_range = tuple(downscale_range)
        self.color_jitter_prob = float(color_jitter_prob)
        self.p_flip = float(p_flip)
        self.mean = list(mean) if mean is not None else list(CLIP_MEAN)
        self.std = list(std) if std is not None else list(CLIP_STD)
        self.version = CLIP_FORENSIC_VERSION

    def _random_spatial_crop(self, pil_img: Image.Image) -> Image.Image:
        """Resolution-invariant spatial crop.

        Eliminates size/aspect-ratio shortcuts (e.g., 178x218 JPG reals vs 256x256 PNG fakes)
        by always taking a square crop from the image and resampling both classes through a
        shared effective-resolution distribution before producing the final 224x224 input.
        """
        w, h = pil_img.size
        min_side = min(w, h)

        # Always take a square crop with scale in [0.75, 1.0] of min(w, h)
        scale = uniform(0.75, 1.0)
        crop_side = max(32, int(round(min_side * scale)))
        left = int(uniform(0, max(1, w - crop_side + 1))) if w > crop_side else 0
        top = int(uniform(0, max(1, h - crop_side + 1))) if h > crop_side else 0
        cropped = pil_img.crop((left, top, min(w, left + crop_side), min(h, top + crop_side)))

        # Harmonize effective Nyquist bandwidth across mismatched source resolutions
        # (e.g. 178px CelebA vs 256px/512px diffusion outputs) via shared intermediate scale
        if self.native_crop_prob > 0.0 and random() < self.native_crop_prob:
            shared_res = int(choice((160, 178, 192, 208, 224)))
            if cropped.size[0] > shared_res:
                interp_down = choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
                cropped = cropped.resize((shared_res, shared_res), resample=interp_down)

        interp_final = choice([Image.Resampling.BICUBIC, Image.Resampling.BILINEAR])
        if cropped.size != (self.size, self.size):
            cropped = cropped.resize((self.size, self.size), resample=interp_final)
        return cropped

    def _apply_forensic_degradations(self, pil_img: Image.Image) -> Image.Image:
        # 1. Optional downscale-upscale resampling (breaks 112px vs 256px Nyquist shortcut)
        if self.downscale_prob > 0.0 and random() < self.downscale_prob:
            lo, hi = self.downscale_range
            s = uniform(lo, hi)
            w, h = pil_img.size
            dw, dh = max(16, int(round(w * s))), max(16, int(round(h * s)))
            interp = choice([Image.Resampling.BILINEAR, Image.Resampling.BICUBIC])
            pil_img = pil_img.resize((dw, dh), resample=interp).resize((w, h), resample=interp)

        # 2. Optional mild color/exposure variation (breaks studio lighting shortcut)
        if self.color_jitter_prob > 0.0 and random() < self.color_jitter_prob:
            pil_img = ImageEnhance.Brightness(pil_img).enhance(uniform(0.85, 1.15))
            pil_img = ImageEnhance.Contrast(pil_img).enhance(uniform(0.85, 1.15))
            pil_img = ImageEnhance.Color(pil_img).enhance(uniform(0.85, 1.15))

        img_np = np.array(pil_img, dtype=np.uint8)

        # 3. Optional Gaussian blur (breaks sharpness shortcut)
        if self.blur_prob > 0.0 and random() < self.blur_prob:
            lo, hi = self.blur_sig if len(self.blur_sig) == 2 else (0.0, self.blur_sig[0])
            sig = uniform(lo, hi)
            if sig > 0.0:
                gaussian_blur(img_np, sig)

        # 4. Optional JPEG recompression (breaks PNG-vs-JPG shortcut)
        if self.jpg_prob > 0.0 and random() < self.jpg_prob:
            qual = int(choice(self.jpg_qual))
            method = str(choice(self.jpg_method))
            img_np = jpeg_from_key(img_np, qual, method)

        return Image.fromarray(img_np, mode="RGB")

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        pil_img = _to_pil_rgb(img)
        pil_img = self._random_spatial_crop(pil_img)
        pil_img = self._apply_forensic_degradations(pil_img)
        if self.p_flip > 0.0 and random() < self.p_flip:
            pil_img = TF.hflip(pil_img)
        tensor = TF.to_tensor(pil_img)
        return TF.normalize(tensor, mean=self.mean, std=self.std)


def get_clip_forensic_preprocessing_metadata(size: int = CLIP_IMAGE_SIZE) -> dict:
    return {
        "preprocessing_version": CLIP_FORENSIC_VERSION,
        "input_size": int(size),
        "mean": list(CLIP_MEAN),
        "std": list(CLIP_STD),
        "resize_policy": "bicubic_center_crop",
        "interpolation": "bicubic",
        "color_order": "RGB",
    }
