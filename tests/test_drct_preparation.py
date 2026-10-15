"""Unit tests for DRCT data preparation, parent linkage, and readiness audit."""
import csv
import json
import os
from pathlib import Path
import pytest
from PIL import Image

from data.drct_manifest import (
    compute_reconstruction_cache_key,
    read_drct_manifest,
    validate_drct_records,
    write_drct_manifest,
)
from data.drct_preparation import (
    check_drct_readiness,
    inventory_source_manifest,
    prepare_reconstructions,
)


def mock_reconstruct(src_path: Path, dst_path: Path, settings: dict) -> bool:
    """Mock reconstruction function creating a synthetic image."""
    dst_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.new('RGB', (64, 64), color=(128, 64, 32))
    img.save(dst_path)
    return True


def test_parent_linkage_and_opposite_labels(tmp_path):
    """Test that reconstructed real sample receives label 1 while authentic parent keeps label 0."""
    src_csv = tmp_path / "source.csv"
    img_real = tmp_path / "real_01.png"
    img_fake = tmp_path / "fake_01.png"
    Image.new('RGB', (64, 64), color=(10, 20, 30)).save(img_real)
    Image.new('RGB', (64, 64), color=(40, 50, 60)).save(img_fake)

    with open(src_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'path', 'label', 'split', 'group_id', 'dataset_source'])
        writer.writeheader()
        writer.writerow({'sample_id': 'real_01', 'path': str(img_real), 'label': 0, 'split': 'train', 'group_id': 'grp_real', 'dataset_source': 'coco'})
        writer.writerow({'sample_id': 'fake_01', 'path': str(img_fake), 'label': 1, 'split': 'train', 'group_id': 'grp_fake', 'dataset_source': 'sd21'})

    inv_csv = tmp_path / "drct_inventory.csv"
    inventory_source_manifest(src_csv, inv_csv)

    rec_dir = tmp_path / "reconstructions"
    out_csv = tmp_path / "drct_prepared.csv"
    res = prepare_reconstructions(
        inventory_manifest_path=inv_csv,
        output_manifest_path=out_csv,
        reconstruction_output_dir=rec_dir,
        target_split='train',
        reconstruction_fn=mock_reconstruct,
    )

    assert res['generated_count'] == 1

    records = read_drct_manifest(out_csv)
    by_id = {r['sample_id']: r for r in records}

    # Parent real original: label 0
    assert int(by_id['real_01']['label']) == 0
    assert by_id['real_01']['sample_kind'] == 'real_original'

    # Reconstructed real: label 1
    recs = [r for r in records if r.get('parent_id') == 'real_01' and r['sample_id'] != 'real_01']
    assert len(recs) == 1
    rec_sample = recs[0]
    assert int(rec_sample['label']) == 1, "Reconstructed real must have label 1!"
    assert rec_sample['sample_kind'] == 'real_reconstruction'
    assert rec_sample['group_id'] == 'grp_real', "Reconstruction must inherit parent group_id!"
    assert rec_sample['split'] == 'train', "Reconstruction must inherit parent split!"


def test_interrupted_preparation_resumes_without_recomputing(tmp_path):
    """Test that existing valid reconstruction files are reused on resume."""
    src_csv = tmp_path / "source.csv"
    img_real = tmp_path / "real_01.png"
    Image.new('RGB', (64, 64), color=(1, 2, 3)).save(img_real)

    with open(src_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['sample_id', 'path', 'label', 'split', 'group_id', 'dataset_source'])
        writer.writeheader()
        writer.writerow({'sample_id': 'real_01', 'path': str(img_real), 'label': 0, 'split': 'train', 'group_id': 'grp_real', 'dataset_source': 'coco'})

    inv_csv = tmp_path / "drct_inventory.csv"
    inventory_source_manifest(src_csv, inv_csv)

    rec_dir = tmp_path / "reconstructions"
    out_csv = tmp_path / "drct_prepared.csv"

    # Run 1: generate
    call_counter = [0]
    def counting_reconstruct(src, dst, settings):
        call_counter[0] += 1
        return mock_reconstruct(src, dst, settings)

    res1 = prepare_reconstructions(inv_csv, out_csv, rec_dir, target_split='train', reconstruction_fn=counting_reconstruct)
    assert res1['generated_count'] == 1
    assert call_counter[0] == 1

    # Run 2: resume should reuse without calling generator
    res2 = prepare_reconstructions(out_csv, out_csv, rec_dir, target_split='train', reconstruction_fn=counting_reconstruct)
    assert res2['reused_count'] == 1
    assert res2['generated_count'] == 0
    assert call_counter[0] == 1, "Generator was unexpectedly re-run on completed valid output!"


def test_deterministic_cache_key_and_invalidation():
    """Verify deterministic cache keys and invalidation when parent hash or settings change."""
    k1 = compute_reconstruction_cache_key('p1', 'hash_aaa', 'sd21', {'steps': 20}, seed=42)
    k2 = compute_reconstruction_cache_key('p1', 'hash_aaa', 'sd21', {'steps': 20}, seed=42)
    assert k1 == k2, "Cache key is not deterministic!"

    # Invalidation on changed seed
    k3 = compute_reconstruction_cache_key('p1', 'hash_aaa', 'sd21', {'steps': 20}, seed=43)
    assert k1 != k3

    # Invalidation on changed parent hash
    k4 = compute_reconstruction_cache_key('p1', 'hash_bbb', 'sd21', {'steps': 20}, seed=42)
    assert k1 != k4

    # Invalidation on changed model revision
    k5 = compute_reconstruction_cache_key('p1', 'hash_aaa', 'sdxl', {'steps': 20}, seed=42)
    assert k1 != k5


def test_preservation_of_protected_splits_and_cross_split_rejection():
    """Verify that reconstructions cannot cross splits and external eval parents are rejected."""
    bad_records = [
        # Parent in external_test
        {
            'sample_id': 'parent_test',
            'parent_id': 'parent_test',
            'path': 'dummy.png',
            'label': 0,
            'split': 'external_test',
            'sample_kind': 'real_original',
            'group_id': 'grp1',
            'dataset_source': 'test_set',
            'sha256': '111',
            'rec_model_revision': 'none',
            'rec_settings': '{}',
            'rec_seed': '0',
        },
        # Derivative attempting to be trained in train split
        {
            'sample_id': 'rec_test',
            'parent_id': 'parent_test',
            'path': 'rec.png',
            'label': 1,
            'split': 'train',
            'sample_kind': 'real_reconstruction',
            'group_id': 'grp1',
            'dataset_source': 'test_set',
            'sha256': '222',
            'rec_model_revision': 'sd21',
            'rec_settings': '{}',
            'rec_seed': '42',
        }
    ]

    val = validate_drct_records(bad_records)
    assert not val['passed']
    assert any("split ('train') does not match parent split ('external_test')" in e for e in val['errors'])


def test_drct_readiness_audit(tmp_path):
    """Test auditing readiness for full DRCT vs RGB control."""
    manifest = tmp_path / "drct_audit.csv"

    # Only originals (no reconstructions)
    records = [
        {'sample_id': 'r1', 'parent_id': 'r1', 'path': 'r1.png', 'label': 0, 'split': 'train', 'sample_kind': 'real_original', 'group_id': 'g1', 'dataset_source': 's', 'sha256': '1', 'rec_model_revision': 'none', 'rec_settings': '{}', 'rec_seed': '0'},
        {'sample_id': 'f1', 'parent_id': 'f1', 'path': 'f1.png', 'label': 1, 'split': 'train', 'sample_kind': 'fake_original', 'group_id': 'g2', 'dataset_source': 's', 'sha256': '2', 'rec_model_revision': 'none', 'rec_settings': '{}', 'rec_seed': '0'},
    ]
    write_drct_manifest(manifest, records)

    audit1 = check_drct_readiness(manifest, split='train')
    assert audit1['is_ready_for_rgb_control'] is True
    assert audit1['is_ready_for_full_drct'] is False
    assert audit1['missing_real_reconstructions'] == 1

    # Add matching reconstruction with actual file on disk
    rec_img = tmp_path / "rec_r1.png"
    Image.new('RGB', (32, 32), color=(0, 0, 0)).save(rec_img)
    records.append(
        {'sample_id': 'rec_r1', 'parent_id': 'r1', 'path': str(rec_img), 'label': 1, 'split': 'train', 'sample_kind': 'real_reconstruction', 'group_id': 'g1', 'dataset_source': 's', 'sha256': '3', 'rec_model_revision': 'sd', 'rec_settings': '{}', 'rec_seed': '42'}
    )
    write_drct_manifest(manifest, records)

    audit2 = check_drct_readiness(manifest, split='train')
    assert audit2['is_ready_for_full_drct'] is False
    assert audit2['missing_real_reconstructions'] == 0
    assert audit2['missing_fake_reconstructions'] == 1
    records[-1]['linkage_basis'] = 'generator_callback'
    fake_rec_img = tmp_path / 'rec_f1.png'
    Image.new('RGB', (32, 32), color=(20, 30, 40)).save(fake_rec_img)
    records.append(dict(sample_id='rec_f1', parent_id='f1', path=str(fake_rec_img), label=1,
                        split='train', sample_kind='fake_reconstruction', group_id='g2',
                        dataset_source='s', sha256='4', linkage_basis='generator_callback'))
    write_drct_manifest(manifest, records)
    assert check_drct_readiness(manifest)['is_ready_for_full_drct'] is True


def test_contradictory_hash_rejection():
    """Verify that records with identical content sha256 but opposing labels are rejected."""
    records = [
        {'sample_id': 's1', 'parent_id': 's1', 'path': 's1.png', 'label': 0, 'split': 'train', 'sample_kind': 'real_original', 'group_id': 'g1', 'dataset_source': 's', 'sha256': 'deadbeef'*8, 'rec_model_revision': 'none', 'rec_settings': '{}', 'rec_seed': '0'},
        {'sample_id': 's2', 'parent_id': 's2', 'path': 's2.png', 'label': 1, 'split': 'train', 'sample_kind': 'fake_original', 'group_id': 'g2', 'dataset_source': 's', 'sha256': 'deadbeef'*8, 'rec_model_revision': 'none', 'rec_settings': '{}', 'rec_seed': '0'},
    ]
    val = validate_drct_records(records)
    assert not val['passed']
    assert any("contradictory labels" in err.lower() for err in val['errors'])


def test_cross_split_group_rejection():
    """Verify that records sharing a group_id across splits are rejected."""
    records = [
        {'sample_id': 's1', 'parent_id': 's1', 'path': 's1.png', 'label': 0, 'split': 'train', 'sample_kind': 'real_original', 'group_id': 'group_alpha', 'dataset_source': 's', 'sha256': '1'*64, 'rec_model_revision': 'none', 'rec_settings': '{}', 'rec_seed': '0'},
        {'sample_id': 's2', 'parent_id': 's2', 'path': 's2.png', 'label': 0, 'split': 'val', 'sample_kind': 'real_original', 'group_id': 'group_alpha', 'dataset_source': 's', 'sha256': '2'*64, 'rec_model_revision': 'none', 'rec_settings': '{}', 'rec_seed': '0'},
    ]
    val = validate_drct_records(records)
    assert not val['passed']
    assert any("spans multiple splits" in err.lower() for err in val['errors'])


def test_inventory_dataset_directory(tmp_path):
    """Verify inventory_dataset_directory discovers real and fake folders and writes valid inventory."""
    from data.drct_preparation import inventory_dataset_directory

    dataset_root = tmp_path / "dataset"
    (dataset_root / "0_real").mkdir(parents=True)
    (dataset_root / "1_fake").mkdir(parents=True)

    Image.new('RGB', (16, 16), color=(255, 0, 0)).save(dataset_root / "0_real" / "r1.png")
    Image.new('RGB', (16, 16), color=(0, 255, 0)).save(dataset_root / "1_fake" / "f1.png")

    out_csv = tmp_path / "inv.csv"
    res = inventory_dataset_directory(
        dataset_root=dataset_root,
        output_manifest_path=out_csv,
        split='train',
        source_name='synthetic_test',
    )
    assert res['total_samples'] == 2
    assert res['real_count'] == 1
    assert res['fake_count'] == 1

    records = read_drct_manifest(out_csv)
    assert len(records) == 2
    assert {r['sample_kind'] for r in records} == {'real_original', 'fake_original'}


def test_import_precomputed_reconstructions(tmp_path):
    """Verify import_precomputed_reconstructions matches precomputed files by stem or key."""
    from data.drct_preparation import inventory_dataset_directory, import_precomputed_reconstructions

    dataset_root = tmp_path / "dataset"
    (dataset_root / "0_real").mkdir(parents=True)
    (dataset_root / "1_fake").mkdir(parents=True)

    Image.new('RGB', (16, 16), color=(255, 10, 20)).save(dataset_root / "0_real" / "sample_a.png")
    Image.new('RGB', (16, 16), color=(10, 255, 20)).save(dataset_root / "1_fake" / "sample_b.png")

    inv_csv = tmp_path / "inv.csv"
    inventory_dataset_directory(dataset_root=dataset_root, output_manifest_path=inv_csv, split='train')

    # Precomputed reconstructions folder containing sample_a.png
    precomputed_dir = tmp_path / "precomputed"
    precomputed_dir.mkdir()
    Image.new('RGB', (16, 16), color=(20, 30, 255)).save(precomputed_dir / "sample_a.png")

    out_csv = tmp_path / "prepared.csv"
    res = import_precomputed_reconstructions(
        inventory_manifest_path=inv_csv,
        reconstruction_dir=precomputed_dir,
        output_manifest_path=out_csv,
        target_split='train',
    )
    assert res['matched_count'] == 1
    records = read_drct_manifest(out_csv)
    assert len(records) == 3  # 1 real, 1 fake, 1 real_reconstruction
    rec_record = next(r for r in records if r['sample_kind'] == 'real_reconstruction')
    assert int(rec_record['label']) == 1
    assert rec_record['parent_id'] == 'train_sample_a'

