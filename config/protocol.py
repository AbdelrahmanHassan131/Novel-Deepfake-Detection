"""Options persisted in checkpoints and experiment configurations."""
DATA_PROTOCOL = dict(manifest=None, manifest_split='train', val_manifest=None,
                     val_manifest_split='dev', audit_hashes=False, allow_folder_training=False)
AUGMENTATION_PROTOCOL = dict(noise_prob=0.0, noise_std=[0.0, 3.0], downscale_prob=0.0,
                             downscale_range=[0.5, 1.0])
LEGACY_PREPROCESSING_KEYS = (
    'cropSize', 'loadSize', 'no_crop', 'no_resize', 'rz_interp',
    'wavelet_type', 'wavelet_level', 'wavelet_mode', 'use_log_packets', 'wavelet_log_mode',
)
DRCT_PREPROCESSING_KEYS = (
    'drct_backbone', 'drct_preprocessing_version',
)
PREPROCESSING_KEYS = LEGACY_PREPROCESSING_KEYS


def get_preprocessing_keys_for_arch(arch: str = None):
    """Return required preprocessing keys scoped to the model architecture."""
    if arch == 'DRCT_128':
        return DRCT_PREPROCESSING_KEYS
    return LEGACY_PREPROCESSING_KEYS




def checkpoint_metadata(model):
    from data.manifest import sha256
    from pathlib import Path
    opt_obj = getattr(model, 'opt', None)
    if opt_obj is not None:
        raw_items = vars(opt_obj).items() if hasattr(opt_obj, '__dict__') and len(vars(opt_obj)) > 0 else [(k, getattr(opt_obj, k)) for k in dir(opt_obj) if not k.startswith('_')]
        options = {k: v for k, v in raw_items
                   if isinstance(v, (str, int, float, bool, list, dict, type(None))) and not k.startswith('_') and not callable(v)}
    else:
        options = {}
    manifests = {}
    for key in ('manifest', 'val_manifest'):
        path = options.get(key)
        if path and Path(path).is_file():
            manifests[key] = {'path': str(Path(path).resolve()), 'sha256': sha256(path)}
    experts = {}
    for name in ('rgb', 'wavelet', 'xception', 'convnext'):
        path = options.get(name + '_model_path')
        attr_name = name + '_model'
        expert_info = {}
        if path and Path(path).is_file():
            expert_info['path'] = str(Path(path).resolve())
            expert_info['sha256'] = sha256(path)
        if hasattr(model, attr_name):
            m = getattr(model, attr_name)
            expert_info['arch'] = getattr(m, 'arch', 'DRCT_128' if hasattr(m, 'encoder') else ('Wang2020_128' if name == 'rgb' else 'WolterWavelet2021_128'))
            if hasattr(m, 'backbone_name'):
                expert_info['backbone_name'] = m.backbone_name
        if expert_info:
            experts[name] = expert_info
    meta = {
        'format_version': 2,
        'label_mapping': {'real': 0, 'fake': 1},
        'options': options,
        'manifests': manifests,
        'experts': experts,
    }
    if hasattr(model, 'head_class_name'):
        meta['head_class'] = model.head_class_name
        meta['head_params'] = getattr(model, 'head_param_count', 0)
        meta['head_trainable_params'] = getattr(model, 'head_trainable_param_count', 0)
    elif hasattr(model, 'model') and hasattr(model.model, '__class__'):
        meta['head_class'] = model.model.__class__.__name__
        try:
            meta['head_params'] = sum(p.numel() for p in model.model.parameters())
            meta['head_trainable_params'] = sum(p.numel() for p in model.model.parameters() if p.requires_grad)
        except Exception:
            pass
    return meta

