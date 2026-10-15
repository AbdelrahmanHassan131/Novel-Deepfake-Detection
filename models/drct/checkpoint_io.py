"""DRCT Checkpoint Import, Restoration, and Pretrained Initialization.

Provides strict, distinct operations for:
1. Restoring a locally trained DRCT-128 checkpoint.
2. Importing an official upstream detector at native width for reference evaluation.
3. Initializing a new 128D adaptation model from pretrained encoder weights without confusing it with a trained detector.
"""
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union
import hashlib
import torch
import torch.nn as nn

from models.drct.drct_model import DRCT128Model, DRCTNativeReference
from data.transforms.drct_transforms import get_drct_preprocessing_metadata


TEXT_KEYS_PREFIXES = (
    'model.model.positional_embedding',
    'model.model.logit_scale',
    'model.model.token_embedding',
    'model.model.text_projection',
    'model.model.ln_final',
    'model.model.transformer',
    'token_embedding',
    'text_projection',
    'ln_final',
)


def _sha256_of_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def extract_weights(checkpoint: Any) -> Dict[str, torch.Tensor]:
    """Extract raw weights dict unwrapping module prefix."""
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'model', 'state_dict'):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected dict of weights, got {type(checkpoint)}")
    return {k.removeprefix('module.'): v for k, v in checkpoint.items()}


def restore_drct128_checkpoint(
    model: DRCT128Model,
    checkpoint: Dict[str, Any],
    strict: bool = True,
    allow_legacy: bool = False,
) -> Dict[str, Any]:
    """Restore a locally trained DRCT-128 checkpoint strictly.
    
    Verifies claimed architecture, label mapping, and embedding dimension.
    Reports all loaded, missing, and unexpected keys.
    """
    weights = extract_weights(checkpoint)
    protocol = checkpoint.get('protocol', {})
    saved_opts = protocol.get('options', {})

    # 1. Architecture validation
    claimed_arch = saved_opts.get('arch') or checkpoint.get('model_name')
    if claimed_arch and claimed_arch != 'DRCT_128' and not allow_legacy:
        raise ValueError(f"Checkpoint claimed architecture '{claimed_arch}', expected 'DRCT_128'")

    # 2. Label mapping validation
    label_mapping = protocol.get('label_mapping')
    if label_mapping is not None:
        if label_mapping.get('real') != 0 or label_mapping.get('fake') != 1:
            raise ValueError(f"Incompatible label mapping in DRCT checkpoint: {label_mapping}. Expected real=0, fake=1.")

    # 3. Embedding dimension validation
    if 'fc.weight' in weights:
        actual_in_dim = weights['fc.weight'].shape[1]
        if actual_in_dim != 128:
            raise ValueError(f"DRCT_128 checkpoint dimension mismatch: expected 128, got {actual_in_dim}")
    elif 'fc.0.weight' in weights:
        raise ValueError("Checkpoint contains Wang2020 'fc.0.weight' key; this is not a DRCT_128 checkpoint.")

    # 4. Strict load into model
    missing_keys, unexpected_keys = model.load_state_dict(weights, strict=strict)
    if missing_keys and strict:
        raise RuntimeError(f"Missing keys restoring DRCT_128 checkpoint: {missing_keys}")
    if unexpected_keys and strict:
        raise RuntimeError(f"Unexpected keys restoring DRCT_128 checkpoint: {unexpected_keys}")

    return {
        'status': 'success',
        'claimed_arch': claimed_arch or 'DRCT_128',
        'missing_keys': missing_keys,
        'unexpected_keys': unexpected_keys,
        'protocol': protocol,
    }


def import_official_drct_detector(
    checkpoint_path_or_dict: Union[str, Path, Dict[str, Any]],
    backbone_name: str = "clip-ViT-L/14",
    embedding_size: int = 1024,
    encoder_module: Optional[nn.Module] = None,
    device: str = "cpu",
) -> Tuple[DRCTNativeReference, Dict[str, Any]]:
    """Import an official upstream DRCT detector at native width for reference evaluation.
    
    Loads official CLIP backbone, 1024-dim projection, and 2-class classification head.
    Exposes single logit difference z = logit_fake - logit_real for 1-logit parity evaluation.
    """
    origin_sha256 = None
    if isinstance(checkpoint_path_or_dict, (str, Path)):
        path = Path(checkpoint_path_or_dict)
        if not path.is_file():
            raise FileNotFoundError(f"Official checkpoint not found: {path}")
        origin_sha256 = _file_sha256(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    else:
        ckpt = checkpoint_path_or_dict

    weights = extract_weights(ckpt)

    # Construct native reference model
    reference_model = DRCTNativeReference(
        backbone_name=backbone_name,
        embedding_size=embedding_size,
        freeze_encoder=False,
        encoder_module=encoder_module,
    )

    # Remap upstream ContrastiveModels state dict keys:
    # Upstream structure:
    #   model.model.visual.* -> visual encoder
    #   model.model.* (text) -> text transformer (explicitly excluded)
    #   model.fc.*           -> Linear(768/512, 1024) [upstream contrastive projection]
    #   fc.*                 -> Linear(1024, 2)       [upstream classifier head]
    has_visual_submod = hasattr(reference_model.encoder, 'visual')
    visual_prefix = 'encoder.visual.' if has_visual_submod else 'encoder.'

    mapped_weights = {}
    for k, v in weights.items():
        if any(k.startswith(p) for p in TEXT_KEYS_PREFIXES):
            # Deliberately ignore text-only tensors from upstream full CLIP wrapper
            continue
        if k.startswith('model.fc.'):
            mapped_k = k.replace('model.fc.', 'projection.')
            mapped_weights[mapped_k] = v
        elif k.startswith('model.model.visual.'):
            mapped_k = k.replace('model.model.visual.', visual_prefix)
            mapped_weights[mapped_k] = v
        elif k.startswith('visual.'):
            mapped_k = k.replace('visual.', visual_prefix)
            mapped_weights[mapped_k] = v
        elif k.startswith('model.model.'):
            mapped_k = k.replace('model.model.', 'encoder.')
            mapped_weights[mapped_k] = v
        elif k.startswith('fc.'):
            mapped_weights[k] = v
        elif k.startswith('projection.'):
            mapped_weights[k] = v
        elif k.startswith('encoder.'):
            mapped_weights[k] = v
        else:
            mapped_weights[k] = v

    # Check projection dimension
    if 'projection.weight' in mapped_weights:
        actual_proj_dim = mapped_weights['projection.weight'].shape[0]
        if actual_proj_dim != embedding_size:
            raise ValueError(
                f"Official projection output dimension is {actual_proj_dim}, "
                f"expected embedding_size={embedding_size}."
            )
    else:
        raise ValueError("Official detector checkpoint missing required 'projection.weight'.")

    # Check classifier dimension (must be 2 classes for upstream official model)
    if 'fc.weight' in mapped_weights:
        actual_classes = mapped_weights['fc.weight'].shape[0]
        if actual_classes != 2:
            raise ValueError(
                f"Official detector expected 2 output classes (real=0, fake=1), "
                f"got {actual_classes} classes."
            )
    else:
        raise ValueError("Official detector checkpoint missing required 'fc.weight'.")

    missing, unexpected = reference_model.load_state_dict(mapped_weights, strict=False)
    # Reject incomplete detector if ANY required destination parameter is missing
    if missing:
        raise ValueError(f"Incomplete official detector checkpoint: missing required weights {missing[:10]}")

    unexplained_unexpected = [
        k for k in unexpected
        if not any(k.startswith(p) for p in TEXT_KEYS_PREFIXES)
    ]
    if unexplained_unexpected:
        raise ValueError(f"Unexpected keys in official detector checkpoint: {unexplained_unexpected[:10]}")

    metadata = {
        'model_type': 'official_reference',
        'origin_sha256': origin_sha256,
        'embedding_size': embedding_size,
        'num_classes': 2,
        'label_mapping': {'real': 0, 'fake': 1},
        'missing_keys': missing,
        'unexpected_keys': unexpected,
        'preprocessing': get_drct_preprocessing_metadata(),
    }

    reference_model.eval()
    return reference_model, metadata


def initialize_drct128_from_pretrained_encoder(
    model: DRCT128Model,
    encoder_checkpoint: Union[str, Path, Dict[str, Any]],
    device: str = "cpu",
) -> Dict[str, Any]:
    """Initialize a new DRCT128 adaptation model from pretrained encoder weights.
    
    Loads ONLY the visual encoder parameters. The 128D projection and classifier
    remain freshly initialized and require adaptation training.
    
    Discards upstream projection and classifier heads without truncating or slicing them.
    Maps source prefixes to match target model encoder structure (with or without .visual wrapper).
    """
    origin_sha256 = None
    if isinstance(encoder_checkpoint, (str, Path)):
        path = Path(encoder_checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"Pretrained encoder checkpoint not found: {path}")
        origin_sha256 = _file_sha256(path)
        ckpt = torch.load(path, map_location=device, weights_only=False)
    else:
        ckpt = encoder_checkpoint

    weights = extract_weights(ckpt)

    target_encoder = model.encoder
    target_keys = set(target_encoder.state_dict().keys())
    target_expects_visual = any(k.startswith('visual.') for k in target_keys)

    # Discard head keys and text keys; filter strictly for visual encoder parameters
    encoder_weights = {}
    for k, v in weights.items():
        if any(k.startswith(p) for p in TEXT_KEYS_PREFIXES):
            continue
        # Discard upstream classification / projection head weights
        if k.startswith(('model.fc.', 'fc.', 'projection.', 'head.', 'classifier.')):
            continue

        # Strip common outer prefixes
        clean_k = k
        for pfx in ('model.model.visual.', 'model.model.', 'encoder.visual.', 'encoder.', 'visual_model.'):
            if clean_k.startswith(pfx):
                clean_k = clean_k.removeprefix(pfx)
                # If prefix ended in .visual., prepend visual. if target expects it
                if 'visual.' in pfx and target_expects_visual:
                    clean_k = f"visual.{clean_k}"
                break

        # Adapt visual. prefix to match target encoder structure
        if target_expects_visual and not clean_k.startswith('visual.'):
            candidate = f"visual.{clean_k}"
            if candidate in target_keys:
                clean_k = candidate
        elif not target_expects_visual and clean_k.startswith('visual.'):
            candidate = clean_k.removeprefix('visual.')
            if candidate in target_keys:
                clean_k = candidate

        if clean_k in target_keys:
            encoder_weights[clean_k] = v

    missing = sorted(list(target_keys - set(encoder_weights.keys())))
    if missing:
        raise ValueError(f"Incomplete pretrained encoder checkpoint: missing required weights {missing[:10]}")

    target_encoder.load_state_dict(encoder_weights, strict=True)

    metadata = {
        'status': 'initialized_for_adaptation_training',
        'trained_detector': False,
        'origin_sha256': origin_sha256,
        'embed_dim': 128,
        'encoder_missing_keys': [],
        'encoder_unexpected_keys': [],
        'head_status': 'freshly_initialized_requires_training',
        'preprocessing': get_drct_preprocessing_metadata(),
    }
    return metadata
