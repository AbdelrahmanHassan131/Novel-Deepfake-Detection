from pathlib import Path


def replace(path, old, new):
    p = Path(path)
    s = p.read_text(encoding='utf-8')
    if old not in s:
        raise RuntimeError(f'Missing anchor {path}: {old[:60]}')
    p.write_text(s.replace(old, new), encoding='utf-8')


replace('evaluate.py', '    return parser.parse_args()', '''    parser.add_argument('--manifest', help='Evaluation manifest; explicit labels and source groups')
    parser.add_argument('--split', default='external_test')
    parser.add_argument('--legacy_config', help='Original preprocessing/options and label_mapping for checkpoints without metadata')
    parser.add_argument('--threshold_file', help='Threshold calibrated on development predictions from this checkpoint')
    parser.add_argument('--bootstrap', type=int, default=200)
    return parser.parse_args()''')
replace('evaluate.py', '    overrides = {}', "    overrides = dict(manifest=args.manifest, manifest_split=args.split, legacy_config=args.legacy_config, threshold_file=args.threshold_file, bootstrap=args.bootstrap)")
replace('evaluation/evaluator.py', '        # 2. Build Dataloader', '''        threshold = 0.5
        threshold_file = (opt_overrides or {}).get('threshold_file')
        if threshold_file:
            import json
            with open(threshold_file, encoding='utf-8') as stream:
                calibration = json.load(stream)
            if calibration['checkpoint_sha256'] != metadata.get('checkpoint_sha256'):
                raise ValueError('Threshold was selected for a different checkpoint')
            if set(calibration.get('source_splits', [])) - {'dev', 'val', 'external_dev'}:
                raise ValueError('Threshold must come from development data')
            threshold = float(calibration['threshold'])
        metadata['decision_threshold_source'] = threshold_file or 'fixed 0.5; no test calibration'
        # 2. Build Dataloader''')
replace('evaluation/evaluator.py', 'ClassificationMetrics(threshold=0.5)', 'ClassificationMetrics(threshold=threshold)')
replace('evaluation/evaluator.py', '        # 5. Performance Profiling', '''        from evaluation.generalization import export_predictions, summarize
        import json
        records = dataloader.dataset.records
        inference_result.predictions = (inference_result.probabilities >= threshold).astype(float)
        export_predictions(os.path.join(self.output_dir, 'predictions.csv'), records,
                           inference_result.probabilities, metadata.get('checkpoint_sha256'))
        generalization = summarize(records, inference_result.probabilities, threshold,
                                  repeats=(opt_overrides or {}).get('bootstrap', 200))
        generalization['checkpoint'] = metadata
        with open(os.path.join(self.output_dir, 'generalization_report.json'), 'w', encoding='utf-8') as stream:
            json.dump(generalization, stream, indent=2, allow_nan=False)
        metrics['balanced_accuracy'] = generalization['overall']['balanced_accuracy']
        # 5. Performance Profiling''')
# Correct confusion matrix visualization threshold.
replace('evaluation/evaluator.py', "                title=f'{detected_arch} Confusion Matrix'", "                title=f'{detected_arch} Confusion Matrix', threshold=threshold")
replace('evaluation/visualization/gradcam.py', "label_text = 'REAL' if labels[i] == 1 else 'FAKE'", "label_text = 'REAL' if labels[i] == 0 else 'FAKE'")
replace('models/wang2020/trainer.py', 'resnet50(pretrained=True)', "resnet50(pretrained=getattr(opt, 'pretrained', True))")
# AUC selection avoids class-imbalance-driven epoch selection.
replace('training/hooks/checkpoint_hook.py', 'current_metric = result.accuracy', 'current_metric = result.auc')
# Match backend packet count across wavelet-level ablations.
for name in ('models/wolter2021/trainer_128.py', 'models/wolter2021/trainer_raw.py'):
    p = Path(name)
    s = p.read_text(encoding='utf-8').replace('input_tensor.shape[1] == 192', 'input_tensor.shape[1] == 3 * 4 ** self.wavelet_level')
    p.write_text(s, encoding='utf-8')
# Four-expert legacy cross-attention remains evaluation-only; its existing
# four-token self-attention is suitable for a separately controlled extension.
p = Path('models/mha_wwxc/trainer.py')
s = p.read_text(encoding='utf-8')
anchor = "        self.embed_dim ="
i = s.index(anchor)
s = s[:i] + '''        if getattr(opt, 'fusion_type', 'token_attention') == 'token_attention':
            opt.fusion_type = 'self_attention'
        if getattr(opt, 'fusion_type', '') == 'cross_attention' and not getattr(opt, '_loading_checkpoint', False):
            raise ValueError('Single-key WWXC cross-attention is legacy-only; use self_attention.')
''' + s[i:]
p.write_text(s, encoding='utf-8')
