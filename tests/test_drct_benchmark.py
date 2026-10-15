"""Unit tests for DRCT-128 throughput benchmark and observability tool."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
import torch

from tools.benchmark_drct import (
    SyntheticDataLoader,
    run_single_benchmark,
    parse_args,
    main,
)


def test_synthetic_dataloader():
    """Verify synthetic dataloader produces correct batch shapes and label balances."""
    batch_size = 8
    loader = SyntheticDataLoader(batch_size=batch_size, steps=3, image_size=224)
    assert len(loader) == 3

    for images, labels in loader:
        assert images.shape == (batch_size, 3, 224, 224)
        assert labels.shape == (batch_size,)
        # Check both real (0) and fake (1) labels exist
        assert set(labels.tolist()).issubset({0.0, 1.0})
        assert (labels == 0).sum() == 4
        assert (labels == 1).sum() == 4


def test_run_single_benchmark_cpu():
    """Verify run_single_benchmark runs end-to-end on CPU with dummy backbone."""
    opt = SimpleNamespace(
        drct_backbone='dummy',
        manifest=None,
        dataroot=None,
        batch_size=4,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        use_amp=False,
        amp_dtype='fp16',
        drct_alpha=0.3,
        drct_pos_margin=0.0,
        drct_neg_margin=1.0,
        embed_dim=128,
    )

    device = torch.device('cpu')
    res = run_single_benchmark(
        opt=opt,
        device=device,
        warmup_steps=1,
        measure_steps=2,
        synthetic=True,
    )

    assert res['status'] == 'completed'
    assert res['batch_size'] == 4
    assert res['measured_steps'] == 2
    assert res['measured_samples'] == 8
    assert res['images_per_sec'] > 0
    assert res['contrastive_pool_size'] == 4
    assert res['contrastive_pairs_per_step'] == 6  # 4 * 3 / 2
    assert 'contrastive_pair_note' in res
    assert 'avg_step_time_ms' in res
    assert 'avg_data_wait_ms' in res


def test_benchmark_cli_and_json_export(tmp_path, monkeypatch):
    """Verify CLI main entry point runs with --synthetic and writes valid JSON report."""
    report_file = tmp_path / "test_report.json"
    test_args = [
        "benchmark_drct.py",
        "--synthetic",
        "--drct_backbone", "dummy",
        "--device", "cpu",
        "--batch_size", "4",
        "--num_workers", "0",
        "--warmup_steps", "1",
        "--measure_steps", "2",
        "--no-use_amp",
        "--output", str(report_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert report_file.is_file()
    data = json.loads(report_file.read_text(encoding='utf-8'))
    assert data['benchmark'] == 'drct_128_throughput'
    assert 'system_info' in data
    assert 'colab_recommendations' in data
    assert len(data['results']) == 1
    assert data['results'][0]['batch_size'] == 4
    assert data['results'][0]['images_per_sec'] > 0


def test_benchmark_sweep_mode(tmp_path, monkeypatch):
    """Verify benchmark sweep mode iterates through candidate batch sizes."""
    report_file = tmp_path / "sweep_report.json"
    test_args = [
        "benchmark_drct.py",
        "--sweep",
        "--sweep_batch_sizes", "2,4",
        "--sweep_workers", "0",
        "--synthetic",
        "--drct_backbone", "dummy",
        "--device", "cpu",
        "--warmup_steps", "1",
        "--measure_steps", "2",
        "--no-use_amp",
        "--output", str(report_file),
    ]
    monkeypatch.setattr(sys, "argv", test_args)

    main()

    assert report_file.is_file()
    data = json.loads(report_file.read_text(encoding='utf-8'))
    assert len(data['results']) == 2
    assert data['results'][0]['batch_size'] == 2
    assert data['results'][1]['batch_size'] == 4


def test_benchmark_gradient_accumulation():
    """Verify gradient accumulation correctly aggregates steps and scales effective batch size."""
    opt = SimpleNamespace(
        drct_backbone='dummy',
        manifest=None,
        dataroot=None,
        batch_size=4,
        grad_accum_steps=2,
        num_workers=0,
        prefetch_factor=2,
        persistent_workers=False,
        pin_memory=False,
        use_amp=False,
        amp_dtype='fp16',
        drct_alpha=0.3,
        drct_pos_margin=0.0,
        drct_neg_margin=1.0,
        embed_dim=128,
    )

    device = torch.device('cpu')
    res = run_single_benchmark(
        opt=opt,
        device=device,
        warmup_steps=1,
        measure_steps=4,
        synthetic=True,
    )

    assert res['status'] == 'completed'
    assert res['data_source'] == 'synthetic'
    assert res['batch_size'] == 4
    assert res['grad_accum_steps'] == 2
    assert res['effective_batch_size'] == 8
    # 4 measured steps with accum_steps=2 yields exactly 2 optimizer steps
    assert res['optimizer_steps'] == 2


def test_benchmark_requires_explicit_mode(tmp_path, monkeypatch):
    """Verify benchmark fails fast if neither --manifest nor --synthetic is supplied."""
    opt = SimpleNamespace(
        drct_backbone='dummy',
        manifest=None,
        dataroot=None,
        batch_size=4,
        grad_accum_steps=1,
        num_workers=0,
        pin_memory=False,
        use_amp=False,
        amp_dtype='fp16',
        drct_alpha=0.3,
        drct_pos_margin=0.0,
        drct_neg_margin=1.0,
        embed_dim=128,
    )
    with pytest.raises(ValueError, match="Manifest path must be provided when synthetic=False"):
        run_single_benchmark(opt, device=torch.device('cpu'), synthetic=False)

    test_args = ["benchmark_drct.py", "--drct_backbone", "dummy"]
    monkeypatch.setattr(sys, "argv", test_args)
    with pytest.raises(SystemExit):
        main()


@pytest.mark.parametrize("num_workers", [0, 1])
def test_benchmark_real_manifest_smoke(tmp_path, monkeypatch, num_workers):
    """Verify benchmark runs with a real temporary manifest and images on disk for workers 0 and >0."""
    from PIL import Image
    from data.drct_manifest import DRCTManifest

    img_dir = tmp_path / "images"
    img_dir.mkdir()

    records = []
    for i in range(4):
        p = img_dir / f"img_{i}.png"
        img = Image.new("RGB", (32, 32), color=(i * 40, i * 20, 100))
        img.save(p)
        label = i % 2
        records.append({
            "sample_id": f"s_{i}",
            "parent_id": f"s_{i}",
            "group_id": f"g_{i}",
            "path": str(p),
            "label": label,
            "split": "train",
            "dataset_source": "test",
            "sample_kind": "real_original" if label == 0 else "fake_original",
            "sha256": f"{i:064x}",
        })

    manifest_path = tmp_path / "manifest.csv"
    DRCTManifest(records=records).save(str(manifest_path))

    opt = SimpleNamespace(
        drct_backbone='dummy',
        manifest=str(manifest_path),
        dataroot=str(img_dir),
        batch_size=2,
        grad_accum_steps=1,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        persistent_workers=False,
        pin_memory=False,
        use_amp=False,
        amp_dtype='fp16',
        drct_alpha=0.3,
        drct_pos_margin=0.0,
        drct_neg_margin=1.0,
        embed_dim=128,
    )

    res = run_single_benchmark(
        opt=opt,
        device=torch.device('cpu'),
        warmup_steps=1,
        measure_steps=2,
        synthetic=False,
    )

    assert res['status'] == 'completed'
    assert res['data_source'] == 'manifest'
    assert res['num_workers'] == num_workers
    assert res['measured_steps'] == 2

