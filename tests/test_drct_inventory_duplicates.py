"""Exact-duplicate cleanup excludes training rows without altering evaluation."""
import json
import shutil

from PIL import Image
import pytest

from data.drct_manifest import read_drct_manifest, file_sha256, validate_drct_records
from data.drct_preparation import inventory_dataset_directory


def dataset(root):
    for split, label, color in [('train', 'real', 'red'), ('train', 'fake', 'blue'),
                                ('val', 'real', 'green'), ('val', 'fake', 'yellow')]:
        path = root / split / label / 'image.png'
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', (16, 16), color).save(path)
    return root


def test_default_still_rejects_overlap_and_records_report(tmp_path):
    root = dataset(tmp_path / 'data')
    shutil.copyfile(root / 'val/real/image.png', root / 'train/real/duplicate.png')
    manifest = tmp_path / 'selected.csv'
    with pytest.raises(ValueError, match='multiple splits'):
        inventory_dataset_directory(root, manifest)
    assert not manifest.exists()
    assert not manifest.with_suffix('.verified.json').exists()
    report = json.loads(manifest.with_suffix('.inventory_report.json').read_text())
    assert report['status'] == 'failed_validation'
    assert report['cross_split_content_hashes'] == 1


def test_cleanup_removes_all_training_copies_preserving_files_and_dev(tmp_path):
    root = dataset(tmp_path / 'data')
    for name in ('duplicate.png', 'duplicate_again.png'):
        shutil.copyfile(root / 'val/real/image.png', root / 'train/real' / name)
    before = {str(p): file_sha256(p) for p in root.rglob('*.png')}
    manifest = tmp_path / 'selected.csv'
    result = inventory_dataset_directory(root, manifest, duplicate_policy='drop_train')
    rows = read_drct_manifest(manifest)
    assert result['scanned_samples'] == 6
    assert result['total_inventoried'] == 4
    assert result['excluded_training_samples'] == 2
    assert validate_drct_records(rows)['passed']
    assert sum(r['split'] == 'dev' for r in rows) == 2
    assert len(read_drct_manifest(result['excluded_rows_csv'])) == 2
    assert {str(p): file_sha256(p) for p in root.rglob('*.png')} == before
    report = json.loads(manifest.with_suffix('.inventory_report.json').read_text())
    assert report['status'] == 'success'
    assert report['manifest_sha256'] == file_sha256(manifest)
    assert manifest.with_suffix('.verified.json').is_file()
    old_bytes = manifest.read_bytes()
    inventory_dataset_directory(root, manifest, duplicate_policy='drop_train')
    assert manifest.read_bytes() == old_bytes


def test_contradictory_labels_are_not_silently_removed(tmp_path):
    root = dataset(tmp_path / 'data')
    shutil.copyfile(root / 'val/real/image.png', root / 'train/fake/conflict.png')
    manifest = tmp_path / 'selected.csv'
    with pytest.raises(ValueError, match='Contradictory labels'):
        inventory_dataset_directory(root, manifest, duplicate_policy='drop_train')
    report = json.loads(manifest.with_suffix('.inventory_report.json').read_text())
    assert report['excluded_training_samples'] == 0
    assert not manifest.exists()


def test_cleanup_must_not_empty_training_class(tmp_path):
    root = dataset(tmp_path / 'data')
    shutil.copyfile(root / 'val/real/image.png', root / 'train/real/image.png')
    with pytest.raises(ValueError, match='must retain both'):
        inventory_dataset_directory(root, tmp_path / 'selected.csv', duplicate_policy='drop_train')


def test_never_choose_between_two_evaluation_splits(tmp_path):
    root = dataset(tmp_path / 'data')
    shutil.copytree(root / 'val', root / 'test')
    with pytest.raises(ValueError, match='multiple splits'):
        inventory_dataset_directory(root, tmp_path / 'selected.csv',
                                    splits=['train', 'dev', 'test'], duplicate_policy='drop_train')


def test_invalid_policy_fails_before_inventory(tmp_path):
    with pytest.raises(ValueError, match='duplicate_policy'):
        inventory_dataset_directory(tmp_path, tmp_path / 'selected.csv', duplicate_policy='ignore')
