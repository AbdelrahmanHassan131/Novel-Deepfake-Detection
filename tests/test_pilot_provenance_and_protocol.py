# STATUS: NOT RUN (runtime tests deferred; static syntax check only)
"""
Tests for Source Provenance, Metadata Coverage, Image Profiling, and Pilot Protocol.

Verifies:
1. Metadata-only source coverage report generation
2. Detection of single-class sources and collection shortcuts (e.g. diffgan)
3. Synthetic placeholder group identification (independent:<hash>)
4. Partition conflict detection across splits
5. Deterministic, group-aware representative dev cohort selection
6. Bounded image profiler and packaging shortcut hypothesis formulation
"""

import tempfile
import unittest
from pathlib import Path

from PIL import Image

from data.manifest import (
    generate_source_coverage_report,
    select_representative_dev_cohort,
    write_manifest,
)
from tools.profile_image_cohort import inspect_single_image, generate_hypotheses


class TestPilotProvenanceAndProtocol(unittest.TestCase):

    def test_source_coverage_report_detection(self):
        """Verifies source counts, generator coverage, single-class warnings, and partition conflicts."""
        rows = [
            {'sample_id': 's1', 'path': 'p1.jpg', 'label': 0, 'split': 'train', 'dataset_source': 'celeb_df', 'generator': 'authentic', 'group_id': 'vid_1'},
            {'sample_id': 's2', 'path': 'p2.jpg', 'label': 1, 'split': 'train', 'dataset_source': 'celeb_df', 'generator': 'deepfake', 'group_id': 'vid_1'},
            {'sample_id': 's3', 'path': 'p3.jpg', 'label': 1, 'split': 'train', 'dataset_source': 'diffgan', 'generator': 'DDPM', 'group_id': 'independent:abc'},
            {'sample_id': 's4', 'path': 'p4.jpg', 'label': 1, 'split': 'dev', 'dataset_source': 'diffgan', 'generator': 'DDPM', 'group_id': 'independent:def'},
            {'sample_id': 's5', 'path': 'p5.jpg', 'label': 0, 'split': 'dev', 'dataset_source': 'single_source', 'generator': 'authentic', 'group_id': 'g1'},
            # Cross-split conflict test
            {'sample_id': 's6', 'path': 'p6.jpg', 'label': 0, 'split': 'train', 'dataset_source': 'celeb_df', 'generator': 'authentic', 'group_id': 'cross_vid'},
            {'sample_id': 's7', 'path': 'p7.jpg', 'label': 1, 'split': 'dev', 'dataset_source': 'celeb_df', 'generator': 'deepfake', 'group_id': 'cross_vid'},
        ]

        report = generate_source_coverage_report(rows)
        self.assertEqual(report['total_samples'], 7)
        self.assertEqual(report['synthetic_placeholder_groups'], 2)

        # Celeb_df has both labels, diffgan has only label 1 (fake), single_source has only label 0 (real)
        single_class_names = {item['source'] for item in report['single_class_sources']}
        self.assertIn('diffgan', single_class_names)
        self.assertIn('single_source', single_class_names)

        # Flagged collection names
        collections = {item['collection_name'] for item in report['collections_identified']}
        self.assertIn('diffgan', collections)

        # Partition conflict detected for cross_vid
        self.assertGreater(report['partition_conflicts_count'], 0)
        self.assertTrue(any('cross_vid' in c for c in report['partition_conflicts']))

    def test_representative_dev_cohort_selection(self):
        """Verifies deterministic selection of group-aware representative dev subset."""
        dev_rows = []
        # Create 10 real groups of 5 images each = 50 real images
        for g in range(10):
            for i in range(5):
                dev_rows.append({
                    'sample_id': f'r_{g}_{i}',
                    'path': f'r_{g}_{i}.png',
                    'label': 0,
                    'split': 'dev',
                    'dataset_source': 'source_A',
                    'group_id': f'real_group_{g}',
                })
        # Create 10 fake groups of 5 images each = 50 fake images
        for g in range(10):
            for i in range(5):
                dev_rows.append({
                    'sample_id': f'f_{g}_{i}',
                    'path': f'f_{g}_{i}.png',
                    'label': 1,
                    'split': 'dev',
                    'dataset_source': 'source_B',
                    'group_id': f'fake_group_{g}',
                })

        # Request balanced subset of 30 images (15 real, 15 fake)
        selected, summary = select_representative_dev_cohort(
            dev_rows,
            target_size=30,
            seed=42,
            target_real=15,
            target_fake=15
        )

        self.assertEqual(len(selected), 30)
        self.assertEqual(summary['real_count'], 15)
        self.assertEqual(summary['fake_count'], 15)
        self.assertIn('cohort_sha256_digest', summary)

        # Connected groups must not be broken
        selected_real_groups = {r['group_id'] for r in selected if r['label'] == 0}
        self.assertEqual(len(selected_real_groups), 3)  # 3 groups * 5 images = 15 images

    def test_image_inspection_and_hypotheses(self):
        """Verifies image profile attributes and packaging hypothesis generation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            img_path = Path(tmpdir) / 'test_sample.png'
            Image.new('RGB', (320, 240), color=(100, 150, 200)).save(img_path)

            info = inspect_single_image(img_path)
            self.assertTrue(info['decode_success'])
            self.assertEqual(info['format'], 'PNG')
            self.assertEqual(info['width'], 320)
            self.assertEqual(info['height'], 240)
            self.assertAlmostEqual(info['aspect_ratio'], round(320 / 240, 4))
            self.assertEqual(info['color_mode'], 'RGB')
            self.assertGreater(info['file_size_bytes'], 0)

            # Test hypothesis formulation on artificial imbalance
            class_stats = {
                0: {'format_percentages': {'JPEG': 95.0, 'PNG': 5.0}, 'mean_resolution': [1024, 1024], 'mean_aspect_ratio': 1.0},
                1: {'format_percentages': {'JPEG': 10.0, 'PNG': 90.0}, 'mean_resolution': [256, 256], 'mean_aspect_ratio': 1.0},
            }
            hypotheses = generate_hypotheses(class_stats, {})
            self.assertGreater(len(hypotheses), 0)
            self.assertTrue(any('Format Packaging Shortcut' in h for h in hypotheses))
            self.assertTrue(any('Resolution Shortcut' in h for h in hypotheses))


if __name__ == '__main__':
    unittest.main()
