import copy
import numpy as np
from random import random, choice
from io import BytesIO
from PIL import Image
import cv2
from scipy.ndimage import gaussian_filter

AUGMENTATION_RECIPES = {
    'legacy': {
        'blur_prob': 0.0,
        'blur_sig': [0.5],
        'jpg_prob': 0.0,
        'jpg_method': ['cv2'],
        'jpg_qual': [75],
        'noise_prob': 0.0,
        'downscale_prob': 0.0,
    },
    'rgb_v1': {
        'blur_prob': 0.5,
        'blur_sig': [0.0, 3.0],
        'jpg_prob': 0.5,
        'jpg_method': ['cv2'],
        'jpg_qual': [50, 60, 70, 80, 90, 95],
        'noise_prob': 0.0,
        'downscale_prob': 0.0,
    },
}

def resolve_augmentation_recipe(opt, in_place=True):
    """
    Resolve named augmentation recipes with explicit precedence rules.

    Precedence:
    - 'legacy': preserves all user-supplied / existing settings on opt (e.g. --blur_prob 0.5 --blur_sig 0.0,3.0).
      Only fills in default unaugmented legacy baseline (0.0 probabilities) for attributes that are absent.
    - 'rgb_v1': applies the rgb_v1 preset (blur 0.5 [0-3], jpeg 0.5 [50..95]),
      while allowing explicit user overrides to take precedence.
    - 'custom': preserves all user parameters as-is.
    """
    target = opt if in_place else copy.deepcopy(opt)
    recipe_name = getattr(target, 'aug_recipe', 'legacy')

    if getattr(target, '_augmentation_resolved', False):
        return target
    if recipe_name == 'rgb_v1':
        explicit = getattr(target, '_explicit_options', None)
        for k, v in AUGMENTATION_RECIPES['rgb_v1'].items():
            # CLI namespaces record supplied flags. Programmatic/saved options
            # have no such provenance, so every existing value is authoritative.
            if (explicit is not None and k not in explicit) or not hasattr(target, k) or getattr(target, k) is None:
                setattr(target, k, copy.deepcopy(v))
    elif recipe_name == 'legacy':
        preset = AUGMENTATION_RECIPES['legacy']
        for k, v in preset.items():
            if not hasattr(target, k) or getattr(target, k) is None:
                setattr(target, k, copy.deepcopy(v))
    elif recipe_name != 'custom':
        raise ValueError(f'Unknown augmentation recipe: {recipe_name}')
    target._augmentation_resolved = True
    return target

def validate_augmentation_params(opt):
    """Validate all augmentation parameters, ranges, and probabilities."""
    for prob_name in ('blur_prob', 'jpg_prob', 'noise_prob', 'downscale_prob'):
        val = getattr(opt, prob_name, 0.0)
        if not (0.0 <= float(val) <= 1.0):
            raise ValueError(f"Augmentation {prob_name} must be in [0.0, 1.0], got {val}")

    sig = _parse_list(getattr(opt, 'blur_sig', [0.5]))
    if isinstance(sig, (list, tuple)):
        for s in sig:
            if float(s) < 0.0:
                raise ValueError(f"Blur sigma must be >= 0.0, got {s}")

    quals = _parse_list(getattr(opt, 'jpg_qual', [75]), int)
    if isinstance(quals, (list, tuple)):
        for q in quals:
            if not (1 <= int(q) <= 100):
                raise ValueError(f"JPEG quality must be in [1, 100], got {q}")

def _parse_list(val, cast_fn=float):
    if isinstance(val, (list, tuple)):
        return [cast_fn(x) for x in val]
    if isinstance(val, str):
        return [cast_fn(x.strip()) for x in val.split(',') if x.strip()]
    return [cast_fn(val)]

def data_augment(img, opt):
    """Apply photometric, blur, and compression augmentations during training."""
    if not getattr(opt, 'isTrain', False):
        return img
    img_np = np.array(img)

    blur_prob = float(getattr(opt, 'blur_prob', 0.0))
    if blur_prob > 0.0 and random() < blur_prob:
        blur_sig = _parse_list(getattr(opt, 'blur_sig', [0.5]), cast_fn=float)
        sig = sample_continuous(blur_sig)
        gaussian_blur(img_np, sig)

    jpg_prob = float(getattr(opt, 'jpg_prob', 0.0))
    if jpg_prob > 0.0 and random() < jpg_prob:
        jpg_method = _parse_list(getattr(opt, 'jpg_method', ['cv2']), cast_fn=str)
        jpg_qual = _parse_list(getattr(opt, 'jpg_qual', [75]), cast_fn=int)
        method = sample_discrete(jpg_method)
        qual = sample_discrete(jpg_qual)
        img_np = jpeg_from_key(img_np, qual, method)

    noise_prob = float(getattr(opt, 'noise_prob', 0.0))
    if noise_prob > 0.0 and random() < noise_prob:
        noise_std = _parse_list(getattr(opt, 'noise_std', [0.0, 3.0]), cast_fn=float)
        sigma = sample_continuous(noise_std)
        img_np = np.clip(img_np.astype(np.float32) + np.random.normal(0, sigma, img_np.shape), 0, 255).astype(np.uint8)

    image = Image.fromarray(img_np)
    downscale_prob = float(getattr(opt, 'downscale_prob', 0.0))
    if downscale_prob > 0.0 and random() < downscale_prob:
        downscale_range = _parse_list(getattr(opt, 'downscale_range', [0.5, 1.0]), cast_fn=float)
        scale = sample_continuous(downscale_range)
        size = image.size
        image = image.resize((max(1, round(size[0] * scale)), max(1, round(size[1] * scale))), Image.Resampling.BILINEAR)
        image = image.resize(size, Image.Resampling.BILINEAR)
    return image

def sample_continuous(s):
    if len(s) == 1:
        return s[0]
    if len(s) == 2:
        rg = s[1] - s[0]
        return random() * rg + s[0]
    raise ValueError(f"Length of iterable s should be 1 or 2, got {len(s)}")

def sample_discrete(s):
    if len(s) == 1:
        return s[0]
    return choice(s)

def gaussian_blur(img, sigma):
    gaussian_filter(img[:, :, 0], output=img[:, :, 0], sigma=sigma)
    gaussian_filter(img[:, :, 1], output=img[:, :, 1], sigma=sigma)
    gaussian_filter(img[:, :, 2], output=img[:, :, 2], sigma=sigma)

def cv2_jpg(img, compress_val):
    img_cv2 = img[:, :, ::-1]
    encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), compress_val]
    result, encimg = cv2.imencode('.jpg', img_cv2, encode_param)
    decimg = cv2.imdecode(encimg, 1)
    return decimg[:, :, ::-1]

def pil_jpg(img, compress_val):
    out = BytesIO()
    img = Image.fromarray(img)
    img.save(out, format='jpeg', quality=compress_val)
    img = Image.open(out)
    img_np = np.array(img)
    out.close()
    return img_np

jpeg_dict = {'cv2': cv2_jpg, 'pil': pil_jpg}

def jpeg_from_key(img, compress_val, key):
    method = jpeg_dict[key]
    return method(img, compress_val)
