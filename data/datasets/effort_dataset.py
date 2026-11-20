"""Dataset implementation for EFFORT_128 (supports folders, standard manifests, and DRCT manifests)."""
from .base_dataset import BaseDataset
from ..transforms.clip_forensic_transforms import (
    CLIPForensicTrainTransform,
    CLIPForensicValTransform,
    CLIP_IMAGE_SIZE,
)


def _as_float_list(val, default):
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple)):
        return [float(x) for x in val]
    if isinstance(val, str):
        return [float(x.strip()) for x in val.split(",") if x.strip()]
    return [float(val)]


def _as_int_list(val, default):
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple)):
        return [int(x) for x in val]
    if isinstance(val, str):
        return [int(x.strip()) for x in val.split(",") if x.strip()]
    return [int(val)]


def _as_str_list(val, default):
    if val is None:
        return list(default)
    if isinstance(val, (list, tuple)):
        return [str(x) for x in val]
    if isinstance(val, str):
        return [x.strip() for x in val.split(",") if x.strip()]
    return [str(val)]


class EFFORTDataset(BaseDataset):
    """Dataset for EFFORT_128 with symmetric forensic CLIP preprocessing."""

    def __init__(self, opt, root: str):
        self.opt = opt
        crop_size = int(getattr(opt, "cropSize", getattr(opt, "crop_size", CLIP_IMAGE_SIZE)))
        backbone = str(getattr(opt, "drct_backbone", "clip-ViT-L/14")).lower()
        if backbone not in ("dummy", "tiny", "mock", "dummy_multilayer") and crop_size != CLIP_IMAGE_SIZE:
            raise ValueError(
                f"EFFORT_128 CLIP backbones require 224-pixel inputs (got {crop_size})."
            )

        is_train = bool(getattr(opt, "isTrain", False))
        if is_train:
            # Use forensic defaults unless caller explicitly customized non-zero values or custom recipe
            aug_recipe = getattr(opt, "aug_recipe", "legacy")
            if aug_recipe == "custom":
                jpg_prob = float(getattr(opt, "jpg_prob", 0.5))
                jpg_qual = _as_int_list(getattr(opt, "jpg_qual", None), (50, 60, 70, 80, 90, 95))
                jpg_method = _as_str_list(getattr(opt, "jpg_method", None), ("cv2", "pil"))
                blur_prob = float(getattr(opt, "blur_prob", 0.3))
                blur_sig_list = _as_float_list(getattr(opt, "blur_sig", None), (0.1, 2.0))
                blur_sig = (blur_sig_list[0], blur_sig_list[-1]) if len(blur_sig_list) >= 2 else (0.0, blur_sig_list[0])
                downscale_prob = float(getattr(opt, "downscale_prob", 0.25))
                ds_list = _as_float_list(getattr(opt, "downscale_range", None), (0.5, 1.0))
                downscale_range = (ds_list[0], ds_list[-1]) if len(ds_list) >= 2 else (0.5, 1.0)
            else:
                jpg_prob = 0.5
                jpg_qual = (50, 60, 70, 80, 90, 95)
                jpg_method = ("cv2", "pil")
                blur_prob = 0.3
                blur_sig = (0.1, 2.0)
                downscale_prob = 0.25
                downscale_range = (0.5, 1.0)

            p_flip = 0.0 if getattr(opt, "no_flip", False) else 0.5
            transform = CLIPForensicTrainTransform(
                size=crop_size,
                native_crop_prob=0.4,
                jpg_prob=jpg_prob,
                jpg_qual=jpg_qual,
                jpg_method=jpg_method,
                blur_prob=blur_prob,
                blur_sig=blur_sig,
                downscale_prob=downscale_prob,
                downscale_range=downscale_range,
                color_jitter_prob=0.3,
                p_flip=p_flip,
            )
        else:
            transform = CLIPForensicValTransform(size=crop_size)

        super().__init__(opt, root, transform=transform)
