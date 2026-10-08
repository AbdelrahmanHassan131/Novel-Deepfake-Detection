"""
Unit tests for data feeding, startup optimization, and staging.
STATUS: NOT RUN (Static code phase test suite)
"""

import argparse
import csv
import hashlib
import os
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
from torch.utils.data import Dataset, DataLoader

from config.configuration import Config
from config.compatibility import config_from_opt, config_to_opt
from config.validator import ConfigValidator, ConfigurationError
from data.manifest import read_manifest, write_manifest, clear_manifest_cache, _MANIFEST_CACHE
from data.loaders.dataloader_factory import _worker_init_fn, create_dataloader
from training.runtime.distributed_runtime import DistributedRuntime
from tools.stage_files_locally import check_disk_space, stage_dataset


class DummyDataset(Dataset):
    def __init__(self, size=10):
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        return torch.randn(3, 32, 32), torch.tensor(idx % 2)


class TestConfigDataFeeding(unittest.TestCase):
    def test_config_attributes_and_conversion(self):
        opt = argparse.Namespace(
            dataroot='./dummy',
            batch_size=32,
            val_batch_size=64,
            num_workers=4,
            val_num_workers=2,
            pin_memory=True,
            prefetch_factor=3,
            persistent_workers=True,
            channels_last=True,
        )
        cfg = config_from_opt(opt)
        self.assertEqual(cfg.data.batch_size, 32)
        self.assertEqual(cfg.data.val_batch_size, 64)
        self.assertEqual(cfg.data.val_num_workers, 2)
        self.assertEqual(cfg.runtime.num_workers, 4)
        self.assertTrue(cfg.runtime.pin_memory)
        self.assertEqual(cfg.runtime.prefetch_factor, 3)
        self.assertTrue(cfg.runtime.persistent_workers)
        self.assertTrue(cfg.runtime.channels_last)

        # Convert back
        opt_back = config_to_opt(cfg)
        self.assertEqual(opt_back.batch_size, 32)
        self.assertEqual(opt_back.val_batch_size, 64)
        self.assertEqual(opt_back.val_num_workers, 2)
        self.assertEqual(opt_back.num_workers, 4)
        self.assertTrue(opt_back.pin_memory)
        self.assertEqual(opt_back.prefetch_factor, 3)
        self.assertTrue(opt_back.persistent_workers)
        self.assertTrue(opt_back.channels_last)

    def test_validator_detects_invalid_values(self):
        cfg = Config.from_defaults()
        cfg.data.val_batch_size = -5
        cfg.data.val_num_workers = -1
        cfg.runtime.prefetch_factor = 0

        report = ConfigValidator.validate(cfg)
        self.assertFalse(report.is_valid)
        error_msgs = " ".join(report.errors)
        self.assertIn("val_batch_size", error_msgs)
        self.assertIn("val_num_workers", error_msgs)
        self.assertIn("prefetch_factor", error_msgs)


class TestDataLoaderZeroWorkersAndValidation(unittest.TestCase):
    @patch('data.loaders.dataloader_factory.get_dataset')
    def test_dataloader_zero_workers_safety(self, mock_get_dataset):
        mock_get_dataset.return_value = DummyDataset(10)
        opt = argparse.Namespace(
            dataroot='./dummy',
            batch_size=4,
            val_batch_size=8,
            num_workers=0,
            val_num_workers=0,
            isTrain=False,
            serial_batches=True,
            class_bal=False,
            pin_memory=False,
            prefetch_factor=2,
            persistent_workers=True,
        )
        loader = create_dataloader(opt)
        self.assertEqual(loader.batch_size, 8)
        self.assertEqual(loader.num_workers, 0)
        # In PyTorch, if num_workers=0, persistent_workers must be False
        self.assertFalse(loader.persistent_workers)

    @patch('data.loaders.dataloader_factory.get_dataset')
    def test_dataloader_train_vs_val_separation(self, mock_get_dataset):
        mock_get_dataset.return_value = DummyDataset(10)
        opt_train = argparse.Namespace(
            dataroot='./dummy',
            batch_size=16,
            val_batch_size=32,
            num_workers=4,
            val_num_workers=2,
            isTrain=True,
            serial_batches=True,
            class_bal=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
        loader_train = create_dataloader(opt_train)
        self.assertEqual(loader_train.batch_size, 16)
        self.assertEqual(loader_train.num_workers, 4)

        opt_val = argparse.Namespace(
            dataroot='./dummy',
            batch_size=16,
            val_batch_size=32,
            num_workers=4,
            val_num_workers=2,
            isTrain=False,
            serial_batches=True,
            class_bal=False,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False,
        )
        loader_val = create_dataloader(opt_val)
        self.assertEqual(loader_val.batch_size, 32)
        self.assertEqual(loader_val.num_workers, 2)


class TestDistributedLoaderWrap(unittest.TestCase):
    def test_wrap_loader_zero_workers(self):
        dataset = DummyDataset(10)
        loader = DataLoader(dataset, batch_size=2, num_workers=0)
        runtime = DistributedRuntime(argparse.Namespace(seed=42))
        # Mock DDP active
        runtime._world_size = 2
        runtime._rank = 0
        runtime._local_rank = 0
        runtime._opt = argparse.Namespace(seed=42)

        wrapped = runtime.wrap_loader(loader, is_train=False)
        self.assertEqual(wrapped.num_workers, 0)
        self.assertFalse(wrapped.persistent_workers)


class TestManifestOptimization(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.manifest_path = Path(self.tmpdir) / 'test_manifest.csv'
        clear_manifest_cache()

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)
        clear_manifest_cache()

    def test_manifest_cache_and_no_probing(self):
        # Create a manifest with fake non-existent paths
        rows = [
            {'sample_id': 's1', 'path': '/non/existent/path/1.png', 'label': '0', 'split': 'train', 'dataset_source': 'ds1', 'group_id': 'g1'},
            {'sample_id': 's2', 'path': '/non/existent/path/2.png', 'label': '1', 'split': 'train', 'dataset_source': 'ds1', 'group_id': 'g2'},
        ]
        write_manifest(self.manifest_path, rows)

        # check_files=False should succeed even though files do not exist
        loaded = read_manifest(self.manifest_path, check_files=False)
        self.assertEqual(len(loaded), 2)
        self.assertGreater(len(_MANIFEST_CACHE), 0)

        # Call again: should hit cache
        loaded_cached = read_manifest(self.manifest_path, check_files=False)
        self.assertEqual(len(loaded_cached), 2)

        # Clear cache
        clear_manifest_cache()
        self.assertEqual(len(_MANIFEST_CACHE), 0)


class TestStagingUtility(unittest.TestCase):
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.src_dir = Path(self.tmpdir) / 'src'
        self.dest_dir = Path(self.tmpdir) / 'dest'
        self.src_dir.mkdir()
        self.manifest_path = Path(self.tmpdir) / 'manifest.csv'
        self.output_manifest = Path(self.tmpdir) / 'staged_manifest.csv'

        # Create dummy source files
        self.rows = []
        for i in range(3):
            file_p = self.src_dir / f"img_{i}.png"
            content = f"content_{i}".encode('utf-8')
            file_p.write_bytes(content)
            digest = hashlib.sha256(content).hexdigest()
            self.rows.append({
                'sample_id': f'sample_{i}',
                'path': str(file_p),
                'label': str(i % 2),
                'split': 'train',
                'dataset_source': 'test_src',
                'group_id': f'g_{i}',
                'sha256': digest,
            })
        write_manifest(self.manifest_path, self.rows)

    def tearDown(self):
        shutil.rmtree(self.tmpdir, ignore_errors=True)

    def test_check_disk_space_headroom(self):
        with patch('shutil.disk_usage') as mock_usage:
            # 100MB free, requiring 100MB * 1.2 = 120MB -> should raise
            mock_usage.return_value = MagicMock(free=100 * 1024 * 1024)
            with self.assertRaises(RuntimeError):
                check_disk_space(self.dest_dir, 100 * 1024 * 1024, headroom_ratio=1.2)

            # 200MB free, requiring 100MB * 1.2 = 120MB -> should pass
            mock_usage.return_value = MagicMock(free=200 * 1024 * 1024)
            free = check_disk_space(self.dest_dir, 100 * 1024 * 1024, headroom_ratio=1.2)
            self.assertEqual(free, 200 * 1024 * 1024)

    def test_stage_dataset_execution_and_verification(self):
        res = stage_dataset(
            manifest_path=self.manifest_path,
            dest_dir=self.dest_dir,
            output_manifest=self.output_manifest,
            verify_digest=True,
            dry_run=False,
        )
        self.assertEqual(res['status'], 'success')
        self.assertEqual(res['sample_count'], 3)
        self.assertTrue(self.output_manifest.exists())

        staged_rows = read_manifest(self.output_manifest, check_files=True)
        self.assertEqual(len(staged_rows), 3)
        for r in staged_rows:
            self.assertTrue(Path(r['path']).exists())
            self.assertTrue(str(self.dest_dir) in r['path'])


if __name__ == '__main__':
    unittest.main()
