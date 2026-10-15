"""Regression coverage for actual audit, preparation, backup and CLI lifecycles."""
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch

from data.drct_manifest import file_sha256, read_drct_manifest, write_drct_manifest
from data.drct_preparation import (
    check_drct_readiness, generate_gate_sidecar, import_precomputed_reconstructions,
    inventory_dataset_directory, prepare_reconstructions,
)
from models.drct.loss import MarginContrastiveLoss
from tools.drct_colab_workflow import (
    copy_stable, create_run, digest, mirror_tree, read_run, restore_preparation,
    run_training, save_preparation,
)

ROOT = Path(__file__).resolve().parents[1]


def image(path, color):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new('RGB', (17, 23), color).save(path)
    return path


def cohort(tmp_path, reconstructions=True):
    dataset = tmp_path / 'dataset'
    for split, label, color in [('train', 'real', 'red'), ('train', 'fake', 'blue'),
                                 ('val', 'real', 'green'), ('val', 'fake', 'yellow')]:
        image(dataset / split / label / f'{label}.png', color)
    base = tmp_path / 'base.csv'
    inventory_dataset_directory(dataset, base)
    rec_dir = tmp_path / 'recs'
    rec_dir.mkdir()
    manifest = tmp_path / 'selected.csv'
    if reconstructions:
        mapping = []
        for row in read_drct_manifest(base):
            if row['split'] != 'train':
                continue
            rec = image(rec_dir / f'{row["sample_id"]}.png', 'orange' if row['label'] == '0' else 'purple')
            mapping.append(dict(parent_id=row['sample_id'], path=rec.name,
                parent_sha256=row['sha256'], sha256=file_sha256(rec),
                rec_model_revision='mock-test-only', rec_settings='{"steps": 1}', rec_seed='42'))
        map_path = tmp_path / 'mapping.csv'
        with map_path.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(mapping[0]))
            writer.writeheader()
            writer.writerows(mapping)
        import_precomputed_reconstructions(base, rec_dir, manifest, reconstruct_fakes=True, mapping_csv=map_path)
    else:
        copy_stable(base, manifest)
    generate_gate_sidecar(manifest, require_hashes=True)
    return dataset, rec_dir, manifest


@pytest.mark.parametrize('batch', [2, 32, 64])
def test_identical_contrastive_vectors_have_zero_loss_and_finite_gradients(batch):
    torch.manual_seed(1)
    emb = torch.randn(1, 128).repeat(batch, 1).requires_grad_()
    loss = MarginContrastiveLoss()(emb, torch.zeros(batch, dtype=torch.long))
    assert loss.item() == 0
    loss.backward()
    assert torch.isfinite(emb.grad).all()
    assert emb.grad.count_nonzero() == 0


def test_hash_audit_failure_returns_nonzero_and_revokes_gate(tmp_path):
    dataset, _, manifest = cohort(tmp_path)
    image(dataset / 'train' / 'real' / 'real.png', 'black')
    result = subprocess.run([sys.executable, '-m', 'data.drct_preparation', 'audit',
        '--manifest', str(manifest), '--hashes', '--require_ready'], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode != 0
    assert 'Content hash mismatch' in result.stderr
    assert not manifest.with_suffix('.verified.json').exists()


def test_serialized_metadata_and_gate_ignore_different_caller_records(tmp_path):
    _, _, manifest = cohort(tmp_path, False)
    records = read_drct_manifest(manifest)
    for row in records:
        row['identity_id'] = 'shared-person'
        row['generator'] = 'actual-generator'
    write_drct_manifest(manifest, records)
    assert read_drct_manifest(manifest)[0]['identity_id'] == 'shared-person'
    with pytest.raises(ValueError, match='identity_id overlaps'):
        generate_gate_sidecar(manifest, records=[dict(records[0], identity_id='unrelated')])


def test_shared_reconstruction_path_rejected(tmp_path):
    dataset, rec_dir, manifest = cohort(tmp_path, False)
    records = read_drct_manifest(manifest)
    original = next(r for r in records if r['split'] == 'train' and r['label'] == '0')
    other = dict(original, sample_id='other', parent_id='other', group_id='other')
    other['path'] = str(image(dataset / 'other' / 'real.png', 'black'))
    other['sha256'] = file_sha256(other['path'])
    records.append(other)
    write_drct_manifest(manifest, records)
    image(rec_dir / 'real_rec.png', 'orange')
    with pytest.raises(ValueError, match='multiple parents'):
        import_precomputed_reconstructions(manifest, rec_dir, tmp_path / 'bad.csv')


def test_full_readiness_requires_fake_pairs_and_verified_links(tmp_path):
    _, _, manifest = cohort(tmp_path)
    assert check_drct_readiness(manifest)['is_ready_for_full_drct']
    records = read_drct_manifest(manifest)
    write_drct_manifest(manifest, [r for r in records if r['sample_kind'] != 'fake_reconstruction'])
    assert not check_drct_readiness(manifest)['is_ready_for_full_drct']
    for r in records:
        if r['sample_kind'].endswith('_reconstruction'):
            r['linkage_basis'] = 'filename_match_unverified'
    write_drct_manifest(manifest, records)
    assert not check_drct_readiness(manifest)['is_ready_for_full_drct']


def test_prepare_without_backend_fails_before_writing(tmp_path):
    _, rec_dir, manifest = cohort(tmp_path, False)
    with pytest.raises(ValueError, match='No reconstruction backend'):
        prepare_reconstructions(manifest, tmp_path / 'result.csv', rec_dir)
    assert not (tmp_path / 'result.csv').exists()


def test_new_session_restore_copies_nested_reconstructions_and_exact_gate(tmp_path):
    _, rec_dir, manifest = cohort(tmp_path)
    nested = image(rec_dir / 'nested' / 'extra.png', 'pink')
    bundle = tmp_path / 'drive' / 'preparation'
    save_preparation(manifest, bundle, rec_dir)
    saved = digest(manifest)
    for p in rec_dir.rglob('*.png'):
        p.unlink()
    manifest.unlink()
    manifest.with_suffix('.verified.json').unlink()
    with pytest.raises(ValueError, match='unchanged-dataset'):
        restore_preparation(bundle, manifest, rec_dir)
    restore_preparation(bundle, manifest, rec_dir, immutable_dataset=True)
    assert nested.is_file()
    assert digest(manifest) == saved
    assert manifest.with_suffix('.verified.json').is_file()
    assert check_drct_readiness(manifest)['is_ready_for_full_drct']


def test_atomic_checkpoint_save_preserves_previous_on_failure(tmp_path, monkeypatch):
    from training.checkpoint_manager import CheckpointManager
    manager = object.__new__(CheckpointManager)
    manager.rank = 0
    manager.save_dir = str(tmp_path)
    manager._build_state = lambda *a: {'test': 1}
    last = tmp_path / 'last.pth'
    last.write_bytes(b'previous-good-generation')
    def fail_save(state, path):
        Path(path).write_bytes(b'incomplete')
        raise OSError('simulated interruption')
    monkeypatch.setattr(torch, 'save', fail_save)
    with pytest.raises(OSError, match='simulated'):
        manager._save('last.pth', 1, None, 1, None)
    assert last.read_bytes() == b'previous-good-generation'
    assert not list(tmp_path.glob('*.tmp'))


def test_restore_rejects_corrupted_reconstruction_without_overwriting_good_copy(tmp_path):
    _, rec_dir, manifest = cohort(tmp_path)
    bundle = tmp_path / 'bundle'
    save_preparation(manifest, bundle, rec_dir)
    original = next(rec_dir.glob('*.png'))
    good_bytes = original.read_bytes()
    (bundle / 'reconstructions' / original.name).write_bytes(b'corrupted-backup')
    with pytest.raises(ValueError, match='differs from audited manifest'):
        restore_preparation(bundle, manifest, rec_dir, immutable_dataset=True)
    assert original.read_bytes() == good_bytes


def test_preparation_rejects_reconstruction_outside_backup_root(tmp_path):
    _, _, manifest = cohort(tmp_path)
    with pytest.raises(ValueError, match='inside the backed-up'):
        save_preparation(manifest, tmp_path / 'bundle', tmp_path / 'wrong-root')


def test_drct_rejects_wrong_clip_input_size_and_nonfinite_loss():
    from config.configuration import Config, ModelConfig, DataConfig
    from config.types import ArchitectureType
    from config.validator import ConfigValidator
    from data.datasets.drct_dataset import DRCTDataset
    config = Config(model=ModelConfig(architecture=ArchitectureType.DRCT_128),
                    data=DataConfig(crop_size=128))
    with pytest.raises(ValueError, match='crop_size=224'):
        ConfigValidator.validate_or_raise(config)
    with pytest.raises(ValueError, match='224-pixel'):
        DRCTDataset(SimpleNamespace(cropSize=128, drct_backbone='clip_vit_l14'), 'unused')
    config.data.crop_size = 224
    config.model.drct_alpha = float('nan')
    with pytest.raises(ValueError, match='drct_alpha'):
        ConfigValidator.validate_or_raise(config)


def test_native_import_ignores_only_qualified_text_embeddings():
    from models.drct.backbone import DummyBackbone
    from models.drct.drct_model import DRCTNativeReference
    from models.drct.checkpoint_io import import_official_drct_detector
    encoder = DummyBackbone(out_features=8)
    native = DRCTNativeReference(encoder_module=encoder, embedding_size=16)
    weights = {}
    for key, value in native.state_dict().items():
        upstream = key.replace('encoder.', 'model.model.visual.').replace('projection.', 'model.fc.')
        weights[upstream] = value.clone()
    weights['model.model.positional_embedding'] = torch.zeros(77, 8)
    weights['model.model.logit_scale'] = torch.tensor(1.)
    restored, _ = import_official_drct_detector(weights, embedding_size=16,
                                               encoder_module=DummyBackbone(out_features=8))
    for key, value in native.state_dict().items():
        assert torch.equal(restored.state_dict()[key], value)
    del weights['fc.bias']
    with pytest.raises(ValueError, match='missing required'):
        import_official_drct_detector(weights, embedding_size=16, encoder_module=DummyBackbone(out_features=8))


def test_failed_training_still_backs_up_completed_artifacts(tmp_path, monkeypatch):
    import tools.drct_colab_workflow as workflow
    _, _, manifest = cohort(tmp_path, False)
    run = create_run(ROOT, manifest, tmp_path / 'runs', tmp_path / 'drive', 'test', ['--epochs', '5'])
    class FailedProcess:
        args = ['mock']
        def __init__(self, *a, **kw):
            exp = Path(run['experiment_dir'])
            exp.mkdir(parents=True)
            (exp / 'metrics.csv').write_text('complete epoch\n')
        def wait(self): return 1
        def poll(self): return 1
    monkeypatch.setattr(workflow.subprocess, 'Popen', FailedProcess)
    # current_commit uses Popen internally too; isolate that read.
    monkeypatch.setattr(workflow, 'current_commit', lambda _: run['commit'])
    with pytest.raises(subprocess.CalledProcessError):
        run_training(run, ROOT, interval_seconds=0.02)
    assert (Path(run['drive_dir']) / 'metrics.csv').is_file()


def test_real_cpu_training_evaluation_calibration_and_fresh_session_resume(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    dataset, _, manifest = cohort(tmp_path)
    args = ['--arch', 'DRCT_128', '--drct_backbone', 'dummy', '--no-pretrained',
            '--drct_alpha', '0.3', '--dataroot', str(dataset), '--val_manifest_split', 'dev',
            '--allow_source_overlap', '--allow_aggregate_sources', '--epochs', '1', '--epochs_decay', '0',
            '--batch_size', '2', '--num_workers', '0', '--seed', '42']
    run = create_run(ROOT, manifest, tmp_path / 'runs', tmp_path / 'drive', 'cpu', args)
    best = run_training(run, ROOT, interval_seconds=0.02)
    assert best.is_file()
    eval_root = tmp_path / 'eval'
    subprocess.run([sys.executable, 'evaluate.py', '--checkpoint', str(best), '--arch', 'DRCT_128',
        '--manifest', str(manifest), '--split', 'dev', '--batch_size', '2', '--device', 'cpu',
        '--no_plots', '--no_tsne', '--no_gradcam', '--no_profiling', '--output_dir', str(eval_root)],
        cwd=ROOT, check=True, capture_output=True, text=True)
    predictions = eval_root / 'DRCT_128' / 'predictions.csv'
    threshold = best.parent / 'threshold_logit.json'
    subprocess.run([sys.executable, 'analyze_predictions.py', 'calibrate', '--predictions', str(predictions),
        '--score_space', 'logit', '--eval_precision', 'fp32', '--output', str(threshold)],
        cwd=ROOT, check=True, capture_output=True, text=True)
    artifact = json.loads(threshold.read_text())
    assert artifact['checkpoint_sha256'] == digest(best)
    assert artifact['score_space'] == 'logit'
    assert artifact['source_splits'] == ['dev']
    experiment = Path(run['experiment_dir'])
    mirror_tree(experiment, run['drive_dir'])
    last_before = torch.load(experiment / 'checkpoints' / 'last.pth', weights_only=False)
    # This path was generated within this test's isolated tmp_path.
    assert experiment.is_relative_to(tmp_path)
    shutil.rmtree(experiment)
    restored_run = read_run(run['drive_dir'])
    run_training(restored_run, ROOT, resume=True, interval_seconds=0.02)
    last_after = torch.load(experiment / 'checkpoints' / 'last.pth', weights_only=False)
    assert last_after['epoch'] == last_before['epoch'] == 1  # never default to 10,000 epochs
    assert restored_run['args'] == run['args']
    changed = dict(run, manifest_sha256='changed')
    with pytest.raises(ValueError, match='identity changed'):
        run_training(changed, ROOT, resume=True)


def test_colab_cells_compile_and_have_no_obsolete_gate_flags():
    guide = (ROOT / 'COLAB_DRCT_128_TRAINING.md').read_text(encoding='utf-8')
    for index, cell in enumerate(re.findall(r'```python\n(.*?)```', guide, flags=re.S)):
        compile(cell, f'Colab cell {index}', 'exec')
    assert '--generate_gate' not in guide
    assert '.csv.verified.json' not in guide
