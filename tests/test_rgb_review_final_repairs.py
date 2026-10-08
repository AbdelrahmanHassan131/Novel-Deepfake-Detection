"""Regression coverage for the second RGB review. NOT RUN locally."""
import copy
import json
from types import SimpleNamespace

import pytest
import torch

from config.compatibility import config_from_opt, config_to_opt
from data.manifest import select_representative_dev_cohort
from data.transforms.augmentations import resolve_augmentation_recipe
from evaluation.generalization import calibrate, export_predictions
from training.checkpoint_manager import CheckpointManager


def test_actual_cli_preserves_rgb_policy_and_explicit_zero(monkeypatch, tmp_path):
    import train
    monkeypatch.setattr('sys.argv', ['train.py', '--dataroot', str(tmp_path),
        '--arch', 'Wang2020_128', '--rgb_head_type', 'linear', '--rgb_dropout', '0',
        '--fine_tune_policy', 'head_only', '--bn_policy', 'frozen',
        '--backbone_lr_mult', '0', '--decay_bias_norm', '--aug_recipe', 'rgb_v1',
        '--jpg_prob', '0', '--blur_prob', '0', '--optim', 'adamw'])
    options = train.parse_args()
    config = config_from_opt(resolve_augmentation_recipe(options))
    restored = config_to_opt(config)
    expected = dict(rgb_head_type='linear', rgb_dropout=0.0, fine_tune_policy='head_only',
                    bn_policy='frozen', backbone_lr_mult=0.0, decay_bias_norm=True)
    for key, value in expected.items():
        assert getattr(restored, key) == value
        assert json.loads(json.dumps(config.to_dict()))['model'][key] == value
    assert restored.jpg_prob == restored.blur_prob == 0.0
    assert restored.optim == 'adamw'


def test_recipe_copy_and_repeat_resolution_preserve_user_values():
    options = SimpleNamespace(aug_recipe='rgb_v1', jpg_prob=0.0,
                              _explicit_options=['jpg_prob'])
    resolved = resolve_augmentation_recipe(options, in_place=False)
    assert not hasattr(options, 'blur_prob')
    assert resolved.blur_prob == 0.5 and resolved.jpg_prob == 0.0
    assert vars(resolve_augmentation_recipe(resolved)) == vars(resolved)


def test_old_optimizer_resume_preserves_moments_and_group_hyperparameters(tmp_path):
    net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    old = torch.optim.Adam(net.parameters(), lr=0.00003, weight_decay=0.001)
    net(torch.ones(1, 2)).sum().backward()
    old.step()
    state = copy.deepcopy(old.state_dict())
    wrapper = SimpleNamespace(model=net, opt=SimpleNamespace(fine_tune_policy='full', backbone_lr_mult=1.0),
                              optimizer=torch.optim.Adam([
                                  {'params': list(net[0].parameters())},
                                  {'params': list(net[1].parameters())}], lr=0.1))
    manager = CheckpointManager(str(tmp_path), wrapper)
    manager._restore_or_migrate_optimizer(state, net)
    assert len(wrapper.optimizer.param_groups) == 1
    assert wrapper.optimizer.param_groups[0]['lr'] == 0.00003
    assert wrapper.optimizer.param_groups[0]['weight_decay'] == 0.001
    for param, old_state in old.state.items():
        assert torch.equal(wrapper.optimizer.state[param]['exp_avg'], old_state['exp_avg'])


def test_source_quota_is_a_cap_and_components_stay_together():
    rows = [dict(sample_id=f's{i}', path=f'/image{i}.jpg', label=i % 2, split='dev',
                 dataset_source='a' if i < 8 else 'b', group_id=f'g{i}', sha256=f'h{i}')
            for i in range(16)]
    rows[1]['sha256'] = rows[0]['sha256']  # linked cross-class records (audit should reject)
    with pytest.raises(ValueError):
        select_representative_dev_cohort(rows, target_size=16)
    rows[1]['sha256'] = 'h1'
    rows[2]['group_id'] = rows[0]['group_id']
    selected, report = select_representative_dev_cohort(rows, target_size=8, source_quotas={'a': 2})
    again, _ = select_representative_dev_cohort(list(reversed(rows)), target_size=8, source_quotas={'a': 2})
    assert {r['sample_id'] for r in selected} == {r['sample_id'] for r in again}
    assert report['source_counts'].get('a', 0) <= 2
    ids = {r['sample_id'] for r in selected}
    assert ('s0' in ids) == ('s2' in ids)


def test_calibration_cannot_certify_missing_precision_or_references(tmp_path):
    rows = [dict(sample_id=f'c{i}', path=f'c{i}.png', label=i, split='dev_calibration',
                 group_id=f'cg{i}', sha256=f'ch{i}') for i in range(2)]
    predictions = tmp_path / 'predictions.csv'
    threshold = tmp_path / 'threshold.json'
    export_predictions(predictions, rows, [0.1, 0.9], checkpoint_hash='checkpoint')
    with pytest.raises(ValueError, match='no recorded precision'):
        calibrate(predictions, threshold, eval_precision='fp32')
    with pytest.raises(FileNotFoundError):
        calibrate(predictions, threshold, train_manifest=tmp_path / 'missing.csv')
    assert calibrate(predictions, threshold)['independence_status'] == 'unverified_dev_calibration'


def test_resume_rejects_changed_finetuning_even_when_shapes_match(tmp_path):
    wrapper = SimpleNamespace(model=torch.nn.Linear(2, 1),
                              opt=SimpleNamespace(fine_tune_policy='full'))
    checkpoint = dict(protocol=dict(label_mapping={'real': 0, 'fake': 1},
                                   options={'fine_tune_policy': 'head_only'}))
    with pytest.raises(ValueError, match='fine_tune_policy'):
        CheckpointManager(str(tmp_path), wrapper)._validate_protocol(checkpoint)


def test_unverified_image_groups_require_explicit_engineering_mode():
    from training.validator import verify_source_readiness
    train = [dict(label=i, dataset_source='train_source', grouping_basis='image_level_unverified')
             for i in (0, 1)]
    dev = [dict(label=i, dataset_source='dev_source', grouping_basis='image_level_unverified')
           for i in (0, 1)]
    with pytest.raises(ValueError, match='video/identity independence'):
        verify_source_readiness(train, dev, monitor_metric='auc')
    verify_source_readiness(train, dev, monitor_metric='auc',
                            allow_aggregate_sources=True, allow_source_overlap=True)
