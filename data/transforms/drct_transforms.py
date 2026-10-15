"""DRCT-specific Preprocessing and Transforms.

Versioned adaptation using longest-edge resize, constant-zero padding, ImageNet
normalization and 224-pixel inputs. The PIL/torchvision implementation is not
certified pixel-equivalent to the upstream OpenCV/albumentations pipeline.
"""
from typing import Any, Tuple, Union
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image


DRCT_PREPROCESSING_VERSION = "drct_clip_v1"
DRCT_IMAGE_SIZE = 224
DRCT_MEAN = [0.485, 0.456, 0.406]
DRCT_STD = [0.229, 0.224, 0.225]


def resize_longest_max_size(img: Image.Image, max_size: int = 224, resample=Image.Resampling.BILINEAR) -> Image.Image:
    """Resize image so its longest edge equals max_size while maintaining aspect ratio."""
    w, h = img.size
    if w == 0 or h == 0:
        return img
    scale = max_size / max(w, h)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return img.resize((new_w, new_h), resample=resample)


def pad_to_square(img: Image.Image, target_size: int = 224, fill: int = 0) -> Image.Image:
    """Pad image with constant fill value so that size becomes (target_size, target_size).
    
    Pads symmetrically if possible, placing remainder on right/bottom.
    Matches upstream PadIfNeeded(border_mode=cv2.BORDER_CONSTANT, value=0).
    """
    w, h = img.size
    pad_w = max(0, target_size - w)
    pad_h = max(0, target_size - h)
    
    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top
    
    return TF.pad(img, [pad_left, pad_top, pad_right, pad_bottom], fill=fill)


class DRCTValTransform:
    """Validation / Evaluation transform for DRCT.
    
    1. Ensures RGB format.
    2. Resizes longest edge to target_size (default 224).
    3. Pads with 0 to (target_size, target_size).
    4. Converts to Tensor in [0, 1].
    5. Normalizes with ImageNet mean/std.
    """
    def __init__(self, size: int = DRCT_IMAGE_SIZE, mean=None, std=None):
        self.size = size
        self.mean = mean or DRCT_MEAN
        self.std = std or DRCT_STD
        self.version = DRCT_PREPROCESSING_VERSION

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        # 1. Convert to PIL Image in RGB mode
        if isinstance(img, np.ndarray):
            # Check if uint8 or float
            if img.dtype != np.uint8:
                if img.max() <= 1.0:
                    img = (img * 255.0).astype(np.uint8)
                else:
                    img = img.astype(np.uint8)
            if img.ndim == 2:
                pil_img = Image.fromarray(img).convert("RGB")
            elif img.shape[2] == 3:
                pil_img = Image.fromarray(img, mode="RGB")
            elif img.shape[2] == 4:
                pil_img = Image.fromarray(img[:, :, :3], mode="RGB")
            else:
                raise ValueError(f"Unsupported numpy image shape: {img.shape}")
        elif isinstance(img, Image.Image):
            pil_img = img.convert("RGB")
        elif isinstance(img, torch.Tensor):
            # If already tensor (C, H, W) in [0, 1]
            if img.dim() == 3:
                arr = img.permute(1, 2, 0).cpu().numpy()
                return self.__call__(arr)
            raise ValueError(f"Unsupported tensor shape: {img.shape}")
        else:
            raise TypeError(f"Unsupported image input type: {type(img)}")

        # 2. Resize longest edge to self.size
        resized = resize_longest_max_size(pil_img, max_size=self.size)

        # 3. Pad to (self.size, self.size) with 0
        padded = pad_to_square(resized, target_size=self.size, fill=0)

        # 4. To tensor and normalize
        tensor = TF.to_tensor(padded)  # (3, H, W) in [0.0, 1.0]
        norm_tensor = TF.normalize(tensor, mean=self.mean, std=self.std)
        return norm_tensor


class DRCTTrainTransform:
    """Training transform for DRCT matching upstream preprocessing with optional horizontal flip."""
    def __init__(self, size: int = DRCT_IMAGE_SIZE, p_flip: float = 0.5, mean=None, std=None):
        self.size = size
        self.p_flip = p_flip
        self.val_transform = DRCTValTransform(size=size, mean=mean, std=std)

    def __call__(self, img: Union[Image.Image, np.ndarray, torch.Tensor]) -> torch.Tensor:
        tensor = self.val_transform(img)
        if self.p_flip > 0 and torch.rand(1).item() < self.p_flip:
            tensor = TF.hflip(tensor)
        return tensor


def get_drct_preprocessing_metadata(size: int = DRCT_IMAGE_SIZE) -> dict:
    """Return dictionary of preprocessing configuration for DRCT checkpoint metadata."""
    return {
        'preprocessing_version': DRCT_PREPROCESSING_VERSION,
        'input_size': size,
        'mean': list(DRCT_MEAN),
        'std': list(DRCT_STD),
        'resize_policy': 'longest_max_size',
        'padding_policy': 'constant_zero_symmetric',
        'interpolation': 'bilinear',
        'color_order': 'RGB',
    }

