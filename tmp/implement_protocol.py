from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    text = p.read_text(encoding='utf-8')
    if old not in text:
        raise RuntimeError(f'Missing edit anchor: {path}: {old[:80]}')
    p.write_text(text.replace(old, new), encoding='utf-8')


p = Path('data/builders/dataset_factory.py')
s = p.read_text(encoding='utf-8')
s = s[:s.index('def get_dataset(opt):')] + '''def get_dataset(opt):
    """Read the binary root once; nested source folders retain outer labels."""
    return dataset_folder(opt, opt.dataroot)


def get_mha_dataset(opt):
    return FusionDataset(opt, opt.dataroot)
'''
p.write_text(s, encoding='utf-8')

replace('config/configuration.py', 'from config.defaults import (', 'from config.protocol import DATA_PROTOCOL, AUGMENTATION_PROTOCOL\n\nfrom config.defaults import (')
replace('config/configuration.py', "        self.val_split = defaults['val_split']", "        self.val_split = defaults['val_split']\n        for key, value in DATA_PROTOCOL.items():\n            setattr(self, key, kwargs.get(key, value))")
replace('config/configuration.py', "        self.data_aug = defaults['data_aug']", "        self.data_aug = defaults['data_aug']\n        for key, value in AUGMENTATION_PROTOCOL.items():\n            setattr(self, key, kwargs.get(key, value))")
replace('config/compatibility.py', 'import argparse', 'import argparse\nfrom config.protocol import DATA_PROTOCOL, AUGMENTATION_PROTOCOL')
replace('config/compatibility.py', "        dataroot=_get(opt, 'dataroot', './dataset/'),", "        **{k: _get(opt, k, v) for k, v in DATA_PROTOCOL.items()},\n        dataroot=_get(opt, 'dataroot', './dataset/'),")
replace('config/compatibility.py', "        blur_prob=_get(opt, 'blur_prob', 0.0),", "        **{k: _get(opt, k, v) for k, v in AUGMENTATION_PROTOCOL.items()},\n        blur_prob=_get(opt, 'blur_prob', 0.0),")
replace('config/compatibility.py', '    opt.val_root = config.data.val_root', '    opt.val_root = config.data.val_root\n    for key in DATA_PROTOCOL:\n        setattr(opt, key, getattr(config.data, key))\n    for key in AUGMENTATION_PROTOCOL:\n        setattr(opt, key, getattr(config.augmentation, key))')
replace('config/compatibility.py', "optimizer=_get(opt, 'optim', 'adam')", "optimizer=_get(opt, 'optim', _get(opt, 'optimizer', 'adam'))")

replace('train.py', "parser.add_argument('--pretrained', action='store_true'", "parser.add_argument('--pretrained', action=argparse.BooleanOptionalAction")
replace('train.py', "parser.add_argument('--use_log_packets', action='store_true'", "parser.add_argument('--use_log_packets', action=argparse.BooleanOptionalAction")
replace('train.py', "default='cross_attention', choices=['cross_attention', 'self_attention', 'concat']", "default='token_attention', choices=['token_attention', 'gated', 'concat', 'cross_attention', 'self_attention']")
replace('train.py', '    # Parse args\n', '''    parser.add_argument('--manifest', help='CSV with explicit labels and source groups')
    parser.add_argument('--manifest_split', default='train')
    parser.add_argument('--val_manifest')
    parser.add_argument('--val_manifest_split', default='dev')
    parser.add_argument('--audit_hashes', action='store_true')
    parser.add_argument('--allow_folder_training', action='store_true', help='Smoke/legacy experiments only: group independence cannot be audited')
    parser.add_argument('--noise_prob', type=float, default=0.0)
    parser.add_argument('--noise_std', type=float, nargs=2, default=[0.0, 3.0], help='Noise sigma in 0-255 pixel units')
    parser.add_argument('--downscale_prob', type=float, default=0.0)
    parser.add_argument('--downscale_range', type=float, nargs=2, default=[0.5, 1.0])
    # Parse args
''')
replace('train.py', '    # Instantiate runtime early', '''    from data.manifest import read_manifest, audit_rows, PROTECTED_SPLITS
    if not opt_clean.manifest and not opt_clean.allow_folder_training:
        raise ValueError('Use --manifest for auditable training, or --allow_folder_training for a smoke experiment.')
    if opt_clean.manifest:
        if opt_clean.manifest_split in PROTECTED_SPLITS or opt_clean.val_manifest_split in PROTECTED_SPLITS:
            raise ValueError('Test splits must not be used for training or checkpoint selection.')
        if opt_clean.manifest_split == opt_clean.val_manifest_split:
            raise ValueError('Training and development split names must differ.')
        if not opt_clean.val_manifest:
            opt_clean.val_manifest = opt_clean.manifest
        rows = read_manifest(opt_clean.manifest, opt_clean.dataroot, opt_clean.manifest_split)
        dev_rows = read_manifest(opt_clean.val_manifest, opt_clean.val_root or opt_clean.dataroot, opt_clean.val_manifest_split)
        report = audit_rows(rows + dev_rows, hash_files=opt_clean.audit_hashes)
        if not report['passed']:
            raise ValueError('Dataset audit failed: ' + '; '.join(report['errors'][:20]))
        if {r['label'] for r in rows} != {0, 1} or {r['label'] for r in dev_rows} != {0, 1}:
            raise ValueError('Training and development each require real and fake samples.')
        print('Dataset audit:', report)
    # Seed BEFORE constructing datasets and model weights, not just the training loop.
    if opt_clean.seed is not None:
        seed_everything(opt_clean.seed, deterministic=opt_clean.deterministic)

    # Instantiate runtime early''')
replace('train.py', '    if val_root and os.path.exists(val_root):', '    if opt_clean.val_manifest or (val_root and os.path.exists(val_root)):')
replace('train.py', '        val_opt.dataroot = val_root', '        val_opt.dataroot = val_root or opt_clean.dataroot\n        val_opt.manifest = opt_clean.val_manifest\n        val_opt.manifest_split = opt_clean.val_manifest_split\n        val_opt.class_bal = False')

# Device selection and variable-size embeddings for expert ablations.
for p in Path('models').rglob('trainer*.py'):
    s = p.read_text(encoding='utf-8')
    s = s.replace('self.model.to(opt.gpu_ids[0])', 'self.model.to(self.device)')
    s = s.replace("map_location='cuda'", 'map_location=self.device')
    s = s.replace('pretrained_flag = self.isTrain and not opt.continue_train', "pretrained_flag = self.isTrain and not opt.continue_train and getattr(opt, 'pretrained', True)")
    p.write_text(s, encoding='utf-8')
replace('models/wang2020_128/trainer.py', 'nn.Linear(2048, 128)', "nn.Linear(2048, getattr(opt, 'embed_dim', 128))")
replace('models/wang2020_128/trainer.py', 'nn.Linear(128, 1)', "nn.Linear(getattr(opt, 'embed_dim', 128), 1)")
replace('models/wolter2021/trainer_128.py', '                num_classes=1', "                num_classes=1, embed_dim=getattr(opt, 'embed_dim', 128)")
replace('models/wolter2021/wavelet_cnn.py', 'def __init__(self, input_channels, num_classes=1):', 'def __init__(self, input_channels, num_classes=1, embed_dim=128):')
replace('models/wolter2021/wavelet_cnn.py', 'nn.Linear(512, 128)', 'nn.Linear(512, embed_dim)')
replace('models/wolter2021/wavelet_cnn.py', 'nn.Linear(128, num_classes)', 'nn.Linear(embed_dim, num_classes)')
replace('models/wolter2021/wavelet_cnn.py', '        x = self.pool4(x)', '        # Level-4 packets can be 1x1 after three pools. Preserve them.\n        if min(x.shape[-2:]) >= 2:\n            x = self.pool4(x)')

replace('training/checkpoint_manager.py', "        return state\n", "        from config.protocol import checkpoint_metadata\n        state['protocol'] = checkpoint_metadata(self.model)\n        state['expert_state_dicts'] = {name: getattr(self.model, name).state_dict() for name in ('rgb_model', 'wavelet_model', 'xception_model', 'convnext_model') if hasattr(self.model, name)}\n        return state\n")
