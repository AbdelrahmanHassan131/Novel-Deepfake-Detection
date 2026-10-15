"""Load frozen expert embeddings without silently falling back to random weights."""
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
import torch.nn as nn
from config.protocol import PREPROCESSING_KEYS


def extract_weights(checkpoint):
    for key in ('model_state_dict', 'model', 'state_dict'):
        if key in checkpoint:
            checkpoint = checkpoint[key]
            break
    return {k.removeprefix('module.'): v for k, v in checkpoint.items()}


def validate_expert_checkpoint(checkpoint, expected_type: str, expected_embed_dim: int = 128, opt=None, allow_legacy: bool = False):
    """
    Validate an expert checkpoint's weights, architecture, label mapping, and preprocessing protocol.
    Enforces strict metadata presence under the fresh-training protocol unless allow_legacy=True.
    Shared across both load_expert and inspect_expert_checkpoint.
    """
    weights = extract_weights(checkpoint)
    dim = getattr(opt, 'embed_dim', expected_embed_dim) if opt is not None else expected_embed_dim
    allow_legacy = allow_legacy or (opt is not None and getattr(opt, 'allow_legacy_checkpoint', False))

    protocol = checkpoint.get('protocol')
    if not protocol:
        if not allow_legacy:
            raise ValueError(
                f"Fresh training protocol requires verified metadata. {expected_type} expert checkpoint "
                "is missing 'protocol' dictionary. For legacy checkpoints, set allow_legacy_checkpoint=True."
            )
        protocol = {}

    saved_opts = protocol.get('options', {})
    if not saved_opts and not allow_legacy:
        raise ValueError(
            f"Fresh training protocol requires verified metadata. {expected_type} expert checkpoint "
            "is missing 'protocol.options'. For legacy checkpoints, set allow_legacy_checkpoint=True."
        )

    # 1. Architecture dispatch & weights validation
    saved_arch = saved_opts.get('arch') or checkpoint.get('model_name')
    is_drct = False

    if expected_type == 'rgb':
        # Check if DRCT-128
        if saved_arch == 'DRCT_128' or (saved_arch is None and any(k.startswith('projection.') for k in weights)):
            is_drct = True
            if saved_arch and saved_arch != 'DRCT_128':
                raise ValueError(f"RGB expert claimed invalid architecture '{saved_arch}', expected 'DRCT_128'")
            proj_key = None
            for pk in ('projection.0.weight', 'projection.weight'):
                if pk in weights:
                    proj_key = pk
                    break
            if proj_key is None:
                raise ValueError("DRCT expert checkpoint missing 'projection.0.weight' embedding layer")
            actual_dim = weights[proj_key].shape[0]
            if actual_dim != dim:
                raise ValueError(f"RGB expert embedding dim mismatch: expected {dim}, got {actual_dim} (DRCT expert must have a {dim}-D embedding checkpoint)")
        else:
            # Wang / ResNet RGB path
            valid_rgb = ('Wang2020_128', 'Wang2020Raw', 'res50', 'resnet50')
            if saved_arch and saved_arch not in valid_rgb and not any(r in str(saved_arch).lower() for r in ('wang', 'res50', 'resnet')):
                raise ValueError(f"RGB expert claimed invalid architecture '{saved_arch}', expected one of {valid_rgb}")
            if 'fc.0.weight' not in weights:
                raise ValueError("RGB expert checkpoint missing 'fc.0.weight' embedding layer")
            actual_dim = weights['fc.0.weight'].shape[0]
            if actual_dim != dim:
                raise ValueError(f"RGB expert embedding dim mismatch: expected {dim}, got {actual_dim} (RGB expert must have a {dim}-D embedding checkpoint)")
    elif expected_type == 'wavelet':
        valid_wav = ('WolterWavelet2021_128', 'WolterWavelet2021Raw', 'wavelet')
        if saved_arch and saved_arch not in valid_wav and not any(w in str(saved_arch).lower() for w in ('wolter', 'wavelet')):
            raise ValueError(f"Wavelet expert claimed invalid architecture '{saved_arch}', expected one of {valid_wav}")
        if 'classifier.0.weight' not in weights:
            raise ValueError("Wavelet expert checkpoint missing 'classifier.0.weight' layer")
        actual_dim = weights['classifier.0.weight'].shape[0]
        if actual_dim != dim:
            raise ValueError(f"Wavelet expert embedding dim mismatch: expected {dim}, got {actual_dim} (Wavelet expert must have a {dim}-D embedding checkpoint)")
    else:
        raise ValueError(f"Unknown expert type: {expected_type}")

    if not saved_arch and not allow_legacy:
        raise ValueError(f"Fresh training protocol requires claimed architecture in {expected_type} expert checkpoint.")

    # 2. Label mapping validation
    label_mapping = protocol.get('label_mapping')
    if label_mapping is not None:
        if label_mapping.get('real') != 0 or label_mapping.get('fake') != 1:
            raise ValueError(f"Incompatible label mapping in expert checkpoint: {label_mapping}. Expected real=0, fake=1.")
    elif not allow_legacy:
        raise ValueError(f"Fresh training protocol requires canonical label mapping in {expected_type} expert checkpoint.")

    # 3. Preprocessing protocol validation
    if opt is not None and saved_opts:
        if is_drct:
            keys_to_check = ('drct_backbone', 'drct_preprocessing_version')
        else:
            keys_to_check = PREPROCESSING_KEYS
        for key in keys_to_check:
            if key in saved_opts and hasattr(opt, key):
                saved_val = saved_opts[key]
                opt_val = getattr(opt, key)
                if opt_val is not None and saved_val != opt_val:
                    raise ValueError(
                        f"{expected_type} expert preprocessing mismatch for {key}: checkpoint had {saved_val}, current opt has {opt_val}"
                    )


class DRCT128FeatureExtractor(nn.Module):
    """Frozen DRCT-128 feature extractor for MHA / TwoStream fusion."""

    def __init__(
        self,
        encoder: nn.Module,
        projection: nn.Module,
        backbone_name: str = 'dummy',
        projection_activation: Optional[str] = None,
        projection_dropout: float = 0.0,
        preprocessing_version: str = 'drct_clip_v1',
        input_size: int = 224,
    ):
        super().__init__()
        self.encoder = encoder
        self.projection = projection
        self.backbone_name = backbone_name
        self.projection_activation = projection_activation
        self.projection_dropout = projection_dropout
        self.preprocessing_version = preprocessing_version
        self.input_size = input_size
        self.embed_dim = 128
        self.arch = 'DRCT_128'
        self.label_mapping = {'real': 0, 'fake': 1}

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.encoder(x)
        return self.projection(feat)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def load_expert(opt, name, device):
    embedded = getattr(opt, '_expert_state_dicts', {}).get(name + '_model')
    embedded_meta = getattr(opt, '_expert_metadata', {}).get(name + '_model', {})
    path = getattr(opt, name + '_model_path', None)

    checkpoint = None
    saved_opts = {}
    if embedded is None:
        if not path or not Path(path).is_file():
            raise FileNotFoundError(f'{name} expert checkpoint is required: {path!r}')
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        validate_expert_checkpoint(checkpoint, expected_type=name, expected_embed_dim=getattr(opt, 'embed_dim', 128), opt=opt)
        weights = extract_weights(checkpoint)
        saved_opts = checkpoint.get('protocol', {}).get('options', {})
    else:
        weights = embedded

    dim = getattr(opt, 'embed_dim', 128)

    if name == 'rgb':
        is_drct = (
            embedded_meta.get('arch') == 'DRCT_128'
            or saved_opts.get('arch') == 'DRCT_128'
            or getattr(opt, 'rgb_expert_arch', None) == 'DRCT_128'
            or (embedded is not None and any(k.startswith('projection.') for k in embedded))
            or (checkpoint is not None and checkpoint.get('model_name') == 'DRCT_128')
        )

        if is_drct:
            backbone_name = (
                embedded_meta.get('backbone_name')
                or saved_opts.get('drct_backbone')
                or getattr(opt, 'drct_backbone', 'dummy')
            )
            act = (
                embedded_meta.get('projection_activation')
                or saved_opts.get('drct_projection_activation')
                or getattr(opt, 'drct_projection_activation', None)
            )
            drop = (
                embedded_meta.get('projection_dropout')
                if embedded_meta.get('projection_dropout') is not None
                else saved_opts.get('drct_projection_dropout', getattr(opt, 'drct_projection_dropout', 0.0))
            )
            prep_ver = embedded_meta.get('preprocessing_version') or saved_opts.get('drct_preprocessing_version', 'drct_clip_v1')
            in_sz = embedded_meta.get('input_size') or saved_opts.get('cropSize', 224)

            from models.drct.backbone import build_drct_backbone
            from models.drct.drct_model import build_drct_projection

            encoder, in_dim = build_drct_backbone(
                name=backbone_name,
                pretrained=(embedded is None and not any(k.startswith('encoder.') for k in weights)),
                freeze=True,
                allow_download=False,
            )
            projection = build_drct_projection(
                in_dim=in_dim,
                embed_dim=dim,
                activation=act,
                dropout=drop,
            )

            model = DRCT128FeatureExtractor(
                encoder=encoder,
                projection=projection,
                backbone_name=backbone_name,
                projection_activation=act,
                projection_dropout=drop,
                preprocessing_version=prep_ver,
                input_size=in_sz,
            )
            extractor_weights = {k: v for k, v in weights.items() if not k.startswith('fc.')}
            # Strictly load all weights without bypassing missing encoder keys
            try:
                model.load_state_dict(extractor_weights, strict=True)
            except Exception as err:
                raise ValueError(f"Failed to load DRCT_128 model weights: {err}") from err
        else:
            from models.shared.resnet import resnet50
            model = resnet50(pretrained=False, num_classes=1)
            if 'fc.0.weight' not in weights or weights['fc.0.weight'].shape[0] != dim:
                raise ValueError(f'RGB expert must have a {dim}-D embedding checkpoint')
            # External expert files contain the complete training head.  Fusion
            # checkpoints deliberately embed the already-truncated feature
            # extractor, so reconstruct the corresponding form before strict load.
            layers = [nn.Linear(model.fc.in_features, dim), nn.ReLU()]
            if embedded is None:
                layers += [nn.Dropout(0.5), nn.Linear(dim, 1)]
            model.fc = nn.Sequential(*layers)
            model.load_state_dict(weights, strict=True)
            model.fc = nn.Sequential(*list(model.fc.children())[:2])

    elif name == 'wavelet':
        from models.wolter2021.wavelet_cnn import WaveletPacketCNN128
        model = WaveletPacketCNN128(3 * 4 ** getattr(opt, 'wavelet_level', 3), embed_dim=dim)
        if embedded is not None:
            model.classifier = nn.Sequential(*list(model.classifier.children())[:2])
        model.load_state_dict(weights, strict=True)
        model.classifier = nn.Sequential(*list(model.classifier.children())[:2])
    else:
        raise ValueError(name)

    model.requires_grad_(False)
    return model.to(device).eval()

