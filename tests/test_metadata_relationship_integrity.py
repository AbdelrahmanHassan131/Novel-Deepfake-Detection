# STATUS: NOT RUN
"""
Deferred regression tests for Prompt B (R6, R7):
1. Independent photographs and synthesis images with absent video IDs must NOT merge into giant components.
2. Manipulations with two original videos (FF++) must connect BOTH originals into the same component.
3. Manipulations linking two identities (Celeb-DF) must connect BOTH identities into the same component.
4. Bridge row removed by frame capping must NOT break the component relationship between linked entities.
"""

import unittest
from pathlib import Path
from data.manifest import known, audit_rows, extract_entity_tokens
from data.adapters import adapt_celeba, adapt_ffpp, adapt_celeb_df, adapt_diffface
from prepare_dataset import connected_components, build_pilot_100k, apply_frame_caps


class TestMetadataRelationshipIntegrity(unittest.TestCase):
    def test_independent_photographs_and_synthesis_not_merged(self):
        """
        Verify that missing-video markers ('none', 'n/a_photograph', 'n/a_synthesis')
        do not join unrelated images into a single giant connected component.
        """
        raw_photos = [
            {'sample_id': f'img_{i}', 'path': f'/data/celeba/{i:06d}.jpg', 'label': '0', 'split': 'unassigned'}
            for i in range(10)
        ]
        adapted_photos = [adapt_celeba(r) for r in raw_photos]

        # In R6 bug, all 10 photos had source_video_id='n/a_photograph' and merged into 1 component.
        # Now, each photo has source_video_id='none' (or distinct), so each must be its own component.
        comp_photos = connected_components(adapted_photos, require_groups=True)
        self.assertEqual(len(comp_photos), 10, "Photographs without shared identities must not merge.")

        raw_synth = [
            {'sample_id': f'synth_{i}', 'path': f'/data/diffface/ADM/fake_{i:04d}.png', 'label': '1', 'split': 'unassigned'}
            for i in range(5)
        ]
        adapted_synth = [adapt_diffface(r) for r in raw_synth]
        comp_synth = connected_components(adapted_synth, require_groups=True)
        self.assertEqual(len(comp_synth), 5, "Synthesis images without shared source must not merge.")

        # Also verify audit_rows does not report 'source_video_id overlaps' when split across train and dev
        adapted_photos[0]['split'] = 'train'
        adapted_photos[1]['split'] = 'dev'
        report = audit_rows(adapted_photos[:2], require_groups=True)
        overlap_errors = [e for e in report['errors'] if 'overlaps' in e]
        self.assertEqual(overlap_errors, [], "Missing video marker must not cause cross-split overlap errors.")

    def test_manipulation_with_both_originals_connected(self):
        """
        Verify that a FaceForensics++ manipulated video (e.g. 042_056)
        connects BOTH original videos (042 AND 056) into the exact same component.
        """
        orig_1 = {
            'sample_id': 'ffpp_042_000',
            'path': '/data/ffpp/original_sequences/youtube/c23/videos/042/frame_000.png',
            'split': 'unassigned'
        }
        orig_2 = {
            'sample_id': 'ffpp_056_000',
            'path': '/data/ffpp/original_sequences/youtube/c23/videos/056/frame_000.png',
            'split': 'unassigned'
        }
        manip = {
            'sample_id': 'ffpp_042_056_000',
            'path': '/data/ffpp/manipulated_sequences/Deepfakes/c23/videos/042_056/frame_000.png',
            'split': 'unassigned'
        }

        adapted_rows = [adapt_ffpp(orig_1), adapt_ffpp(orig_2), adapt_ffpp(manip)]

        # Check multi-entity original_id
        manip_origs = extract_entity_tokens('original_id', adapted_rows[2]['original_id'])
        self.assertIn('ffpp_042', manip_origs)
        self.assertIn('ffpp_056', manip_origs)

        comp_map = connected_components(adapted_rows, require_groups=True)
        # All 3 records must be unified into exactly ONE connected component
        self.assertEqual(len(comp_map), 1, "Manipulation must connect both original videos into one component.")
        comp_indices = list(comp_map.values())[0]
        self.assertEqual(sorted(comp_indices), [0, 1, 2])

    def test_two_identities_celeb_df_connected(self):
        """
        Verify that a Celeb-DF manipulated video with two donor identities (id0_id1_0000)
        connects pristine videos of id0 AND id1 into the exact same component.
        """
        orig_a = {
            'sample_id': 'celebdf_real_a',
            'path': '/data/celebdf/Celeb-real/id0_0000/000.png',
            'split': 'unassigned'
        }
        orig_b = {
            'sample_id': 'celebdf_real_b',
            'path': '/data/celebdf/Celeb-real/id1_0000/000.png',
            'split': 'unassigned'
        }
        fake_ab = {
            'sample_id': 'celebdf_fake_ab',
            'path': '/data/celebdf/Celeb-synthesis/id0_id1_0000/000.png',
            'split': 'unassigned'
        }

        adapted = [adapt_celeb_df(orig_a), adapt_celeb_df(orig_b), adapt_celeb_df(fake_ab)]
        comp_map = connected_components(adapted, require_groups=True)

        self.assertEqual(len(comp_map), 1, "Celeb-DF pair must unite both donor identities into one component.")
        comp_indices = list(comp_map.values())[0]
        self.assertEqual(sorted(comp_indices), [0, 1, 2])

    def test_bridge_row_removed_by_capping_preserves_relationship(self):
        """
        Verify that when connected components are computed on the full pool first,
        even if a bridge row linking two entities is capped or dropped, the retained
        frames retain the shared group_id and do not split across partitions.
        """
        # Entity 1: Video 1 with 10 frames
        v1_rows = [
            {'sample_id': f'v1_{i}', 'path': f'/data/ffpp/042/{i:03d}.png', 'group_id': 'ffpp_single_042',
             'dataset_source': 'FaceForensics++', 'label': 0, 'source_video_id': 'ffpp_042', 'split': 'unassigned'}
            for i in range(10)
        ]
        # Entity 2: Video 2 with 10 frames
        v2_rows = [
            {'sample_id': f'v2_{i}', 'path': f'/data/ffpp/056/{i:03d}.png', 'group_id': 'ffpp_single_056',
             'dataset_source': 'FaceForensics++', 'label': 0, 'source_video_id': 'ffpp_056', 'split': 'unassigned'}
            for i in range(10)
        ]
        # Bridge: Manipulated video with 1 frame linking 042 and 056
        bridge_row = [{
            'sample_id': 'bridge_manip',
            'path': '/data/ffpp/042_056/000.png',
            'group_id': 'ffpp_pair_042_056',
            'dataset_source': 'FaceForensics++',
            'label': 1,
            'source_video_id': 'ffpp_042_056',
            'original_id': 'ffpp_042,ffpp_056',
            'split': 'unassigned'
        }]

        all_pool = v1_rows + v2_rows + bridge_row

        # Build pilot with frame_cap=2.
        # Connected components on full pool will link v1 and v2 via bridge_row before any capping.
        capped_out, report = build_pilot_100k(
            all_pool,
            target_real=2,
            target_fake=1,
            frame_cap=2,
            dev_ratio=0.5,
            test_ratio=0.0,
            seed=42,
            require_groups=True
        )

        # Check group_ids of all rows from v1 and v2: they must all share the same component group_id
        v1_group_ids = {r['group_id'] for r in capped_out if 'v1_' in r['sample_id']}
        v2_group_ids = {r['group_id'] for r in capped_out if 'v2_' in r['sample_id']}
        self.assertEqual(v1_group_ids, v2_group_ids, "v1 and v2 must share the unified component group_id.")

        # Check splits: since they belong to the same component, they must never be split across train and dev
        all_splits = {r['split'] for r in capped_out}
        # If in train, all are in train; if in dev, all are in dev
        v1_splits = {r['split'] for r in capped_out if 'v1_' in r['sample_id']}
        v2_splits = {r['split'] for r in capped_out if 'v2_' in r['sample_id']}
        self.assertEqual(v1_splits, v2_splits, "Connected components must never be split across different partitions.")


if __name__ == '__main__':
    unittest.main()
