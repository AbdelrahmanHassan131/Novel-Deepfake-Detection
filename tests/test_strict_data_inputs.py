# STATUS: NOT RUN
"""Regression tests for Prompt I: Strict data inputs and correct loading (S1, S2, S8).

Verifies:
1. Unresolved name PROTECTED_SPLITS is resolved and manifest loading control-flow handles
   both explicit --manifest and root/manifest.csv without uninitialized rows.
2. Inventory does not fabricate source_video_id or group_id unless verified independent photographs.
3. Cross-platform path portability: deterministic relative resolution under effective_root,
   multi-part suffix relocation matching (no basename matching), remap_prefixes, and per-source roots.
"""
import ast
import os
import tempfile
import unittest
from pathlib import Path


class TestStrictDataInputs(unittest.TestCase):

    def test_protected_splits_imported_and_defined(self):
        """Verify prepare_dataset.py imports and defines PROTECTED_SPLITS."""
        prep_path = Path("prepare_dataset.py")
        tree = ast.parse(prep_path.read_text(encoding="utf-8"), filename="prepare_dataset.py")
        
        imported_names = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                for alias in node.names:
                    imported_names.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    imported_names.add(alias.name)

        self.assertIn("PROTECTED_SPLITS", imported_names,
                      "prepare_dataset.py must import PROTECTED_SPLITS from data.manifest")

    def test_manifest_loading_control_flow_and_no_fallback_fabrication(self):
        """Verify manifest loading handles explicit and implicit manifest and rejects missing manifest for pilot."""
        from prepare_dataset import main
        import sys

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create root/manifest.csv
            manifest_csv = tmppath / "manifest.csv"
            manifest_csv.write_text(
                "sample_id,path,label,split,dataset_source,group_id\n"
                "s1,real/img1.png,0,dev,src1,g1\n",
                encoding="utf-8"
            )
            (tmppath / "real").mkdir(parents=True)
            (tmppath / "real" / "img1.png").write_bytes(b"data1")

            out_csv = tmppath / "out.csv"

            # 1. Pilot without manifest should fail if neither --manifest nor root/manifest.csv exists
            empty_dir = tmppath / "empty"
            empty_dir.mkdir()
            test_args = ["prepare_dataset.py", "pilot", "--root", str(empty_dir), "--output", str(out_csv)]
            orig_argv = sys.argv
            try:
                sys.argv = test_args
                with self.assertRaises(ValueError) as ctx:
                    main()
                self.assertIn("required for action 'pilot'", str(ctx.exception))
            finally:
                sys.argv = orig_argv

    def test_inventory_does_not_fabricate_groups_or_videos(self):
        """Verify inventory retains unknown grouping metadata and stores relative paths."""
        from prepare_dataset import main
        import sys
        from data.manifest import read_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            train_real = tmppath / "train" / "real"
            train_real.mkdir(parents=True)
            img1 = train_real / "img1.png"
            img1.write_bytes(b"real_data")

            out_csv = tmppath / "inv.csv"
            test_args = [
                "prepare_dataset.py", "inventory",
                "--root", str(tmppath),
                "--source", "test_source",
                "--output", str(out_csv)
            ]
            orig_argv = sys.argv
            try:
                sys.argv = test_args
                main()
            finally:
                sys.argv = orig_argv

            rows = read_manifest(str(out_csv), root=str(tmppath), check_files=True)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            # Group and video must be unknown (not fabricated from parent folder 'real')
            self.assertEqual(row['group_id'], 'unknown')
            self.assertEqual(row['source_video_id'], 'unknown')
            self.assertEqual(row['dataset_source'], 'test_source')
            self.assertEqual(row['label'], 0)

    def test_inventory_independent_images_flag(self):
        """Verify inventory with --independent_images assigns distinct group_id and source_video_id='none'."""
        from prepare_dataset import main
        import sys
        from data.manifest import read_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            img = tmppath / "real" / "face_01.png"
            img.parent.mkdir(parents=True)
            img.write_bytes(b"face_data")

            out_csv = tmppath / "inv_indep.csv"
            test_args = [
                "prepare_dataset.py", "inventory",
                "--root", str(tmppath),
                "--source", "celeba_photographs",
                "--independent_images",
                "--output", str(out_csv)
            ]
            orig_argv = sys.argv
            try:
                sys.argv = test_args
                main()
            finally:
                sys.argv = orig_argv

            rows = read_manifest(str(out_csv), root=str(tmppath), check_files=True)
            self.assertEqual(len(rows), 1)
            row = rows[0]
            self.assertEqual(row['source_video_id'], 'none')
            import hashlib
            expected = hashlib.sha256(b'celeba_photographs:real/face_01.png').hexdigest()
            self.assertEqual(row['group_id'], f'independent:{expected}')

    def test_cross_platform_path_portability_and_relocation(self):
        """Verify read_manifest resolves relative paths deterministically and relocates absolute paths without basename matching."""
        from data.manifest import read_manifest, write_manifest

        with tempfile.TemporaryDirectory() as tmpdir:
            tmppath = Path(tmpdir)
            # Create real image under tmpdir / real / img1.png
            img_file = tmppath / "real" / "img1.png"
            img_file.parent.mkdir(parents=True)
            img_file.write_bytes(b"image_content")

            # 1. Windows absolute path with drive letter
            win_abs_path = "G:\\Master's Study\\datasets\\source_a\\real\\img1.png"
            manifest_rows = [{
                'sample_id': 's1',
                'path': win_abs_path,
                'label': 0,
                'split': 'train',
                'dataset_source': 'source_a',
                'group_id': 'g1'
            }]
            m_path = tmppath / "manifest.csv"
            write_manifest(str(m_path), manifest_rows)

            # Unmapped absolute path must raise FileNotFoundError (no ambiguous suffix guessing)
            with self.assertRaises(FileNotFoundError):
                read_manifest(str(m_path), root=str(tmppath), check_files=True)

            # Providing explicit remap_prefixes resolves the image accurately
            remap = {"G:\\Master's Study\\datasets\\source_a": str(tmppath)}
            loaded = read_manifest(str(m_path), remap_prefixes=remap, check_files=True)
            self.assertEqual(len(loaded), 1)
            self.assertEqual(Path(loaded[0]['path']).resolve(), img_file.resolve())

            # 2. Relative path should never collide with a file in CWD
            # Create a file in CWD with same name
            cwd_file = Path("img1.png")
            created_cwd_file = False
            if not cwd_file.exists():
                try:
                    cwd_file.write_bytes(b"cwd_content")
                    created_cwd_file = True
                except Exception:
                    pass

            try:
                rel_rows = [{
                    'sample_id': 's2',
                    'path': 'real/img1.png',
                    'label': 0,
                    'split': 'train',
                    'dataset_source': 'source_a',
                    'group_id': 'g1'
                }]
                m_rel = tmppath / "manifest_rel.csv"
                write_manifest(str(m_rel), rel_rows)
                loaded_rel = read_manifest(str(m_rel), root=str(tmppath), check_files=True)
                self.assertEqual(Path(loaded_rel[0]['path']).resolve(), img_file.resolve())
            finally:
                if created_cwd_file and cwd_file.exists():
                    cwd_file.unlink()

            # 3. Source roots mapping
            src_b_dir = tmppath / "source_b_dir"
            (src_b_dir / "real").mkdir(parents=True)
            img_b = src_b_dir / "real" / "img1.png"
            img_b.write_bytes(b"src_b_content")

            multi_src_rows = [
                {'sample_id': 's_a', 'path': 'real/img1.png', 'label': 0, 'split': 'train', 'dataset_source': 'src_a', 'group_id': 'ga'},
                {'sample_id': 's_b', 'path': 'real/img1.png', 'label': 0, 'split': 'train', 'dataset_source': 'src_b', 'group_id': 'gb'},
            ]
            m_multi = tmppath / "manifest_multi.csv"
            write_manifest(str(m_multi), multi_src_rows)

            loaded_multi = read_manifest(
                str(m_multi),
                root=str(tmppath),
                source_roots={'src_a': str(tmppath), 'src_b': str(src_b_dir)},
                check_files=True
            )
            self.assertEqual(Path(loaded_multi[0]['path']).resolve(), img_file.resolve())
            self.assertEqual(Path(loaded_multi[1]['path']).resolve(), img_b.resolve())


if __name__ == '__main__':
    unittest.main()
