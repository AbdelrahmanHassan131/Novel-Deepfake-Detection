"""Reproducible corruptions and attacks applied to the same raw image in all streams."""
import argparse
import copy
from io import BytesIO
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageFilter
import torch
from torch.utils.data import DataLoader
from data.datasets.rgb_dataset import RGBDataset
from evaluation.checkpoint_loader import CheckpointLoader
from evaluation.generalization import summarize, export_predictions
from evaluation.pixel_pipeline import PixelDetector, denormalize, projected_attack


def corrupt(pixels, name):
    if name.startswith('noise'):
        sigma = float(name.split('_')[1]) / 255
        return (pixels + torch.randn_like(pixels) * sigma).clamp(0, 1)
    outputs = []
    for sample in pixels:
        image = Image.fromarray((sample.detach().cpu().permute(1, 2, 0).numpy() * 255).round().astype('uint8'))
        if name.startswith('jpeg_'):
            buffer = BytesIO()
            image.save(buffer, format='JPEG', quality=int(name.split('_')[1]))
            buffer.seek(0)
            image = Image.open(buffer).convert('RGB')
        elif name.startswith('blur_'):
            image = image.filter(ImageFilter.GaussianBlur(float(name.split('_')[1])))
        elif name.startswith('resize_'):
            size = image.size
            scale = float(name.split('_')[1])
            image = image.resize(tuple(max(1, int(s * scale)) for s in size), Image.Resampling.BILINEAR).resize(size, Image.Resampling.BILINEAR)
        else:
            raise ValueError(f'Unknown corruption {name}')
        outputs.append(torch.from_numpy(np.array(image).copy()).permute(2, 0, 1).float() / 255)
    return torch.stack(outputs).to(pixels.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--arch')
    parser.add_argument('--val_root', '--dataroot', dest='dataroot', required=True)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--split', default='external_test')
    parser.add_argument('--legacy_config')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--output_dir', default='evaluation_results/robustness')
    parser.add_argument('--conditions', nargs='+', default=['clean', 'jpeg_75', 'jpeg_30', 'blur_1', 'noise_3', 'resize_0.5', 'fgsm', 'pgd'])
    parser.add_argument('--eps', type=float, default=4/255, help='L-infinity budget in raw [0,1] pixel units')
    parser.add_argument('--steps', type=int, default=10)
    parser.add_argument('--step_size', type=float, default=1/255)
    parser.add_argument('--restarts', type=int, default=2)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--threshold_file')
    parser.add_argument('--bootstrap', type=int, default=200)
    for expert in ('rgb', 'wavelet', 'xception', 'convnext'):
        parser.add_argument('--' + expert + '_model_path')
    args = parser.parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    overrides = dict(dataroot=args.dataroot, legacy_config=args.legacy_config,
                     manifest=args.manifest, manifest_split=args.split)
    overrides.update({key: value for key, value in vars(args).items() if key.endswith('_model_path') and value})
    loader = CheckpointLoader(args.checkpoint, args.arch, args.device)
    model = loader.load(overrides)
    model.eval()
    detector = PixelDetector(model)
    opt = copy.copy(model.opt)
    opt.isTrain = False
    dataset = RGBDataset(opt, args.dataroot)
    batches = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    threshold = .5
    if args.threshold_file:
        calibration = json.loads(Path(args.threshold_file).read_text(encoding='utf-8'))
        if calibration['checkpoint_sha256'] != loader.metadata['checkpoint_sha256']:
            raise ValueError('Threshold checkpoint mismatch')
        if not calibration.get('source_splits') or set(calibration['source_splits']) - {'dev', 'val', 'external_dev'}:
            raise ValueError('Calibration must come from development data')
        threshold = calibration['threshold']
    root = Path(args.output_dir)
    root.mkdir(parents=True, exist_ok=True)
    report = {'checkpoint': loader.metadata, 'settings': vars(args),
              'pixel_pipeline': 'deterministic resize/crop -> raw corruption/attack -> normalization and recomputed wavelets -> all experts',
              'conditions': {}}
    for condition in dict.fromkeys(['clean'] + args.conditions):
        scores = []
        print(f'Evaluating {condition}', flush=True)
        for batch, labels in batches:
            pixels = denormalize(batch.to(model.device)).clamp(0, 1)
            if condition in ('fgsm', 'pgd'):
                pixels = projected_attack(detector, pixels, labels.to(model.device), eps=args.eps,
                    steps=1 if condition == 'fgsm' else args.steps,
                    step_size=args.eps if condition == 'fgsm' else args.step_size,
                    random_start=condition == 'pgd', restarts=1 if condition == 'fgsm' else args.restarts)
            elif condition != 'clean':
                pixels = corrupt(pixels, condition)
            with torch.no_grad():
                scores.extend(detector(pixels).sigmoid().cpu().tolist())
        report['conditions'][condition] = summarize(dataset.records, scores, threshold, args.bootstrap, args.seed)
        export_predictions(root / f'{condition}_predictions.csv', dataset.records, scores, loader.metadata['checkpoint_sha256'])
        (root / 'robustness_report.json').write_text(json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
    print(root / 'robustness_report.json')


if __name__ == '__main__':
    main()

