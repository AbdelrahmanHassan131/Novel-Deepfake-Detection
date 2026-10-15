"""Dataset implementation for DRCT models.

Supports:
1. DRCT manifest loading (with sample kinds, parent linkage, and opposite labels).
2. Base manifest loading (standard format).
3. Directory fallback (real/fake folders).
4. DRCT versioned transforms (drct_clip_v1).
"""
import os
from pathlib import Path
from typing import Optional
from torchvision.datasets.folder import default_loader
from torchvision.datasets.vision import VisionDataset
from .base_dataset import BaseDataset
from ..transforms.drct_transforms import DRCTValTransform, DRCTTrainTransform, DRCT_IMAGE_SIZE
from ..drct_manifest import DRCTManifest


class DRCTDataset(BaseDataset):
    """Dataset for DRCT training and evaluation."""

    def __init__(self, opt, root: str):
        self.opt = opt
        crop_size = getattr(opt, 'cropSize', getattr(opt, 'crop_size', DRCT_IMAGE_SIZE))
        if str(getattr(opt, 'drct_backbone', 'clip_vit_l14')).lower() not in ('dummy', 'tiny', 'mock') and crop_size != DRCT_IMAGE_SIZE:
            raise ValueError('DRCT CLIP backbones require 224-pixel inputs; restore the saved preprocessing configuration.')
        is_train = getattr(opt, 'isTrain', False)

        if is_train and not getattr(opt, 'no_flip', False):
            transform = DRCTTrainTransform(size=crop_size)
        else:
            transform = DRCTValTransform(size=crop_size)

        if not is_train:
            # During evaluation: ONLY use explicit opt.manifest if requested for manifest evaluation.
            # Never fall back to saved checkpoint training or validation manifests.
            manifest_path = getattr(opt, 'manifest', None)
        else:
            manifest_path = getattr(opt, 'manifest', None) or getattr(opt, 'drct_manifest_path', None)

        loaded_from_drct_manifest = False

        if manifest_path:
            manifest_file = Path(manifest_path).resolve()
            if not manifest_file.is_file():
                raise FileNotFoundError(f"Requested manifest not found: {manifest_file}")

            manifest_obj = DRCTManifest.load(str(manifest_file))
            val_res = manifest_obj.validate()
            if not val_res['passed']:
                raise ValueError(f"DRCT manifest validation failed for '{manifest_file}': {'; '.join(val_res['errors'][:5])}")

            split = getattr(opt, 'manifest_split', None) or ('train' if is_train else 'val')
            records = manifest_obj.get_split(split)
            if not records:
                available_splits = sorted(list(set(r.get('split', 'train') for r in manifest_obj.records)))
                raise ValueError(
                    f"Requested split '{split}' not found in manifest '{manifest_file}'. "
                    f"Available splits in manifest: {available_splits}"
                )

            VisionDataset.__init__(self, root, transform=transform)
            # CSV labels are strings; reports consume records directly, whereas
            # the image loader consumes samples. Keep both interfaces canonical.
            records = [dict(r, label=int(r['label'])) for r in records]
            self.records = records
            self.samples = []
            for r in records:
                fpath = r.get('path') or r.get('file_path') or r.get('relative_path')
                if not fpath:
                    raise KeyError(f"Manifest record missing path: {r}")
                if not os.path.isabs(fpath) and root:
                    candidate = os.path.join(root, fpath)
                    if os.path.exists(candidate):
                        fpath = candidate
                if not os.path.isfile(fpath):
                    raise FileNotFoundError(f"Image in manifest not found on disk: {fpath}")
                self.samples.append((fpath, int(r['label'])))

            self.targets = [label for _, label in self.samples]
            self.imgs = self.samples
            self.loader = default_loader
            self.classes = ['real', 'fake']
            self.class_to_idx = {'real': 0, 'fake': 1}
            loaded_from_drct_manifest = True

        if not loaded_from_drct_manifest:
            super().__init__(opt, root, transform=transform)
