"""Inventory, grouped splitting, pilot data preparation, and leakage safeguards.

Never invent source identities or assign independent groups to conceal missing metadata.
Ensures connected originals and manipulations stay together in the same split.
"""
import argparse
from collections import defaultdict, Counter
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from PIL import Image

import re
from data.manifest import read_manifest, write_manifest, audit_rows, GROUP_FIELDS, UNKNOWN, PROTECTED_SPLITS, known, sha256, extract_entity_tokens
from data.adapters import apply_adapter, ADAPTERS
from data.hash_cache import HashCache, SQLiteHashCache


def connected_components(rows: List[Dict[str, Any]], require_groups: bool = True) -> Dict[str, List[int]]:
    """Union-Find across all identity/original/video/hash links.

    Every sample connected by any link is assigned to the exact same component.
    """
    parent = list(range(len(rows)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, row in enumerate(rows):
        if require_groups and not known(row.get('group_id')):
            raise ValueError(
                f"Missing group_id for sample {row.get('sample_id')}; "
                "fill group_id from original metadata before splitting. "
                "Do not treat frames as independent."
            )
        for key in GROUP_FIELDS + ('sha256',):
            val = row.get(key)
            tokens = extract_entity_tokens(key, val)
            for token_val in tokens:
                token = (key, token_val)
                if token in seen:
                    parent[find(i)] = find(seen[token])
                else:
                    seen[token] = i

    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)

    # Compute stable hash ID for each component based on canonical entity tokens
    hashed_groups = {}
    for indices in groups.values():
        entity_tokens = set()
        for i in indices:
            for key in GROUP_FIELDS:
                val = rows[i].get(key)
                for t in extract_entity_tokens(key, val):
                    entity_tokens.add(f"{key}:{t}")
        if entity_tokens:
            canonical_anchor = min(entity_tokens)
        else:
            canonical_anchor = f"sample:{min(rows[i]['sample_id'] for i in indices)}"
        group_hash = "connected:" + hashlib.sha256(canonical_anchor.encode()).hexdigest()
        hashed_groups[group_hash] = indices
    return hashed_groups


def _frame_sort_key(row: Dict[str, Any]) -> Tuple[int, int, str]:
    """Sort frames by verified temporal/frame order extracted from filename or sample_id."""
    p = str(row.get('path', ''))
    match = re.search(r'(\d+)(?:\.[a-zA-Z0-9]+)?$', Path(p).name)
    if match:
        return (0, int(match.group(1)), p)
    s_id = str(row.get('sample_id', ''))
    match2 = re.search(r'(\d+)$', s_id)
    if match2:
        return (0, int(match2.group(1)), p)
    return (1, 0, p)


def apply_frame_caps(rows: List[Dict[str, Any]], frame_cap: int = 30, seed: int = 42) -> List[Dict[str, Any]]:
    """Cap well-spaced frames per source video in verified temporal order."""
    if frame_cap <= 0:
        return rows

    video_buckets = defaultdict(list)
    uncapped = []

    for i, row in enumerate(rows):
        vid = row.get('source_video_id')
        if known(vid):
            video_buckets[(row.get('dataset_source', 'unknown'), vid)].append(row)
        else:
            uncapped.append(row)

    capped_rows = list(uncapped)

    for (src, vid), v_rows in sorted(video_buckets.items()):
        v_rows.sort(key=_frame_sort_key)
        if len(v_rows) <= frame_cap:
            capped_rows.extend(v_rows)
        else:
            # Deterministic spaced selection to maintain temporal coverage
            indices = np.linspace(0, len(v_rows) - 1, frame_cap, dtype=int)
            selected = [v_rows[idx] for idx in indices]
            capped_rows.extend(selected)

    return capped_rows


def build_pilot_100k(
    rows: List[Dict[str, Any]],
    target_real: int = 50000,
    target_fake: int = 50000,
    frame_cap: int = 30,
    source_quotas: Optional[Dict[str, int]] = None,
    generator_quotas: Optional[Dict[str, int]] = None,
    dev_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
    holdout_source: Optional[str] = None,
    holdout_generator: Optional[str] = None,
    require_groups: bool = True,
    allow_shortfall: bool = False,
    allow_holdout_override: bool = False,
    train_size: Optional[str] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Select training rows referencing the original images; never copy image files.

    train_size is an even total (balanced class targets) or 'all'. None preserves
    the legacy per-class targets. 'all' includes every eligible training row,
    disables frame caps, and rejects source/generator quotas. Dev/test and source
    holdouts remain excluded. Selection resolves relationships on the full pool.

    Guarantees:
      - 50K real / 50K fake training targets (or honest shortage reporting if insufficient).
      - Frame caps applied per video to avoid video dominance.
      - Connected groups (identities, originals, fakes, recompressions) never split across partitions.
      - Source-by-class summaries generated to expose confounding.
    """
    full_training = False
    if train_size is not None:
        size_text = str(train_size).strip().lower()
        full_training = size_text == 'all'
        if full_training:
            if source_quotas or generator_quotas:
                raise ValueError("--train_size all cannot be combined with source/generator quotas")
            frame_cap = None
        else:
            if not size_text.isdecimal() or int(size_text) <= 0 or int(size_text) % 2:
                raise ValueError("train_size must be a positive even image count (e.g. 100000) or 'all'")
            target_real = target_fake = int(size_text) // 2

    # 0. Validate ratios
    if dev_ratio < 0.0 or test_ratio < 0.0 or (dev_ratio + test_ratio) >= 1.0:
        raise ValueError(
            f"Invalid ratios: dev_ratio={dev_ratio}, test_ratio={test_ratio}. "
            "Ratios must be non-negative and sum to < 1.0"
        )

    # Copy quotas to avoid mutating caller dictionaries
    rem_source_quotas = dict(source_quotas.copy()) if source_quotas is not None else None
    rem_gen_quotas = dict(generator_quotas.copy()) if generator_quotas is not None else None

    # 1. Resolve connected components on the full pool first
    # Building the relationship graph before capping or sampling guarantees that bridge rows
    # linking two entities are never dropped prior to graph unification.
    comp_map = connected_components(rows, require_groups=require_groups)

    # Assign persistent group_id to all rows in the pool
    for group_id, indices in comp_map.items():
        for idx in indices:
            rows[idx]['group_id'] = group_id

    # 2. Resolve component-level split constraints on the full pool BEFORE capping
    holdout_indices = set()
    dev_indices = set()
    test_indices = set()
    candidate_train_groups = []

    for group_id, indices in sorted(comp_map.items()):
        # Check pre-assigned splits across ALL rows in this component
        existing_fixed = {
            rows[i].get('split') for i in indices
            if known(rows[i].get('split')) and rows[i].get('split') not in {'unassigned', 'pool'}
        }

        # Check for conflicting fixed partitions
        if len(existing_fixed) > 1:
            examples = [
                {key: rows[i].get(key) for key in
                 ('sample_id', 'path', 'split', 'source_video_id', 'identity_id', 'original_id', 'sha256')}
                for split in sorted(existing_fixed)
                for i in [next(j for j in indices if rows[j].get('split') == split)]
            ]
            raise ValueError(
                f"Conflicting fixed partition assignments in connected component '{group_id}': {sorted(existing_fixed)}. "
                "Connected original/manipulated records cannot span multiple splits. "
                f"Representative records: {json.dumps(examples, ensure_ascii=False)}. "
                "Inspect the shared group/identity/original/video/hash metadata; do not bypass the overlap check."
            )

        # Check holdout source/generator rules across ALL rows in this connected component
        is_external_holdout = any(
            (holdout_source and rows[i].get('dataset_source') == holdout_source) or
            (holdout_generator and rows[i].get('generator') == holdout_generator)
            for i in indices
        )

        if is_external_holdout:
            if 'train' in existing_fixed:
                if not allow_holdout_override:
                    raise ValueError(
                        f"Conflicting fixed partition in component '{group_id}': Component is pre-assigned to 'train' "
                        f"but matches holdout rule (holdout_source='{holdout_source}', holdout_generator='{holdout_generator}'). "
                        "Held-out sources/generators cannot be included in the training partition. "
                        "Exclude held-out source records before splitting or pass allow_holdout_override=True."
                    )
                else:
                    for i in indices:
                        if rows[i].get('split') not in {'dev', 'test', 'internal_test', 'external_test'}:
                            rows[i]['split'] = 'external_dev'
                        holdout_indices.add(i)
                    continue
            else:
                for i in indices:
                    if rows[i].get('split') not in {'dev', 'test', 'internal_test', 'external_test'}:
                        rows[i]['split'] = 'external_dev'
                    holdout_indices.add(i)
                continue

        if len(existing_fixed) == 1:
            fixed_split = next(iter(existing_fixed))
            for i in indices:
                rows[i]['split'] = fixed_split

            if fixed_split in PROTECTED_SPLITS or fixed_split in {'external_dev', 'external_test'}:
                for i in indices:
                    holdout_indices.add(i)
            elif fixed_split == 'dev':
                for i in indices:
                    dev_indices.add(i)
            elif fixed_split in {'test', 'internal_test'}:
                for i in indices:
                    test_indices.add(i)
            elif fixed_split == 'train':
                # Pre-assigned train component: eligible for training quota
                candidate_train_groups.append((group_id, indices))
            continue

        # Freeze dev and internal_test partitions using stable partition hash
        part_u = int(hashlib.sha256(f"partition:{group_id}".encode()).hexdigest()[:12], 16) / 16**12
        if dev_ratio > 0 and part_u < dev_ratio:
            for i in indices:
                rows[i]['split'] = 'dev'
                dev_indices.add(i)
        elif test_ratio > 0 and part_u < (dev_ratio + test_ratio):
            for i in indices:
                rows[i]['split'] = 'internal_test'
                test_indices.add(i)
        else:
            for i in indices:
                rows[i]['split'] = 'unassigned'
            candidate_train_groups.append((group_id, indices))

    # 3. Apply frame caps per verified video inside candidate training components
    eligible_train_groups = []
    for group_id, indices in candidate_train_groups:
        comp_rows = [rows[i] for i in indices]
        if frame_cap is not None and frame_cap > 0:
            selected_comp = apply_frame_caps(comp_rows, frame_cap=frame_cap, seed=seed)
        else:
            selected_comp = comp_rows

        u_key = hashlib.sha256(f"{seed}:{group_id}".encode()).hexdigest()
        real_count = sum(1 for r in selected_comp if r['label'] == 0)
        fake_count = sum(1 for r in selected_comp if r['label'] == 1)
        src_counts = Counter(r.get('dataset_source', 'unknown') for r in selected_comp)
        gen_counts = Counter(r.get('generator', 'unknown') for r in selected_comp)

        eligible_train_groups.append({
            'group_id': group_id,
            'selected_rows': selected_comp,
            'u_key': u_key,
            'real_count': real_count,
            'fake_count': fake_count,
            'src_counts': src_counts,
            'gen_counts': gen_counts,
        })

    # Sort eligible training groups deterministically by training seed
    eligible_train_groups.sort(key=lambda g: g['u_key'])

    if full_training:
        # No class balancing/downsampling in full mode: retain natural counts.
        target_real = sum(g['real_count'] for g in eligible_train_groups)
        target_fake = sum(g['fake_count'] for g in eligible_train_groups)
        if target_real == 0 or target_fake == 0:
            raise ValueError("Full training pool must contain both real and fake eligible images")

    # 4. Fill training quotas exactly without exceeding targets or source/generator caps
    cur_real = 0
    cur_fake = 0
    selected_train_rows = []

    for g in eligible_train_groups:
        comp_rows = g['selected_rows']
        real_cnt = g['real_count']
        fake_cnt = g['fake_count']

        can_fit_real = (real_cnt == 0) or (cur_real + real_cnt <= target_real)
        can_fit_fake = (fake_cnt == 0) or (cur_fake + fake_cnt <= target_fake)

        if not (can_fit_real and can_fit_fake):
            continue

        quota_ok = True
        if rem_source_quotas:
            for s, cnt in g['src_counts'].items():
                if s in rem_source_quotas and rem_source_quotas[s] < cnt:
                    quota_ok = False
                    break
        if quota_ok and rem_gen_quotas:
            for gen, cnt in g['gen_counts'].items():
                if gen in rem_gen_quotas and rem_gen_quotas[gen] < cnt:
                    quota_ok = False
                    break

        if quota_ok and (cur_real < target_real or cur_fake < target_fake):
            for r in comp_rows:
                r['split'] = 'train'
                selected_train_rows.append(r)
            cur_real += real_cnt
            cur_fake += fake_cnt

            if rem_source_quotas:
                for s, cnt in g['src_counts'].items():
                    if s in rem_source_quotas:
                        rem_source_quotas[s] -= cnt
            if rem_gen_quotas:
                for gen, cnt in g['gen_counts'].items():
                    if gen in rem_gen_quotas:
                        rem_gen_quotas[gen] -= cnt

    if dev_ratio > 0 and len(dev_indices) == 0:
        raise ValueError(
            "Dev split has 0 samples. Insufficient independent validation coverage; "
            "adjust dev_ratio or dataset pool."
        )

    # 5. Compile output pool: full evaluation membership + selected training rows
    output_rows = [rows[i] for i in sorted(holdout_indices | dev_indices | test_indices)] + selected_train_rows
    output_rows.sort(key=lambda r: r['sample_id'])

    # Compile honest shortage & confounding report
    real_shortage = max(0, target_real - cur_real)
    fake_shortage = max(0, target_fake - cur_fake)

    if (real_shortage > 0 or fake_shortage > 0) and not allow_shortfall:
        raise ValueError(
            f"Pilot selection shortfall: requested {target_real} real and {target_fake} fake samples, "
            f"but only found {cur_real} real and {cur_fake} fake eligible samples "
            f"(shortage: real={real_shortage}, fake={fake_shortage}). "
            "To accept a smaller balanced pool (e.g. for smoke testing or limited datasets), "
            "declare smaller --target_real/--target_fake or specify --allow_shortfall."
        )

    shortage_report = {
        'target_real': target_real,
        'selected_real': cur_real,
        'real_shortage': real_shortage,
        'target_fake': target_fake,
        'selected_fake': cur_fake,
        'fake_shortage': fake_shortage,
        'allow_shortfall': allow_shortfall,
        'total_training_samples': cur_real + cur_fake,
        'dev_samples': len(dev_indices),
        'internal_test_samples': len(test_indices),
        'external_holdout_samples': len(holdout_indices),
        'unique_connected_groups': len(comp_map),
        'frame_cap_applied': frame_cap,
    }

    # Audit source-by-class distributions
    source_class = defaultdict(Counter)
    split_counts = Counter()
    splits_dict = {}
    for r in output_rows:
        s = r['split']
        if s not in splits_dict:
            splits_dict[s] = {
                'real_count': 0,
                'fake_count': 0,
                'total_count': 0,
                'frame_cap': frame_cap if (s == 'train' and frame_cap is not None) else 'None',
                '_groups': set(),
            }
        if r['label'] == 0:
            splits_dict[s]['real_count'] += 1
        elif r['label'] == 1:
            splits_dict[s]['fake_count'] += 1
        splits_dict[s]['total_count'] += 1
        splits_dict[s]['_groups'].add(r.get('group_id'))

        source_class[(r['split'], r.get('dataset_source', 'unknown'))][r['label']] += 1
        split_counts[r['split']] += 1

    for s, data in splits_dict.items():
        data['group_count'] = len(data.pop('_groups'))

    confounding_warnings = []
    for (split, src), counts in sorted(source_class.items()):
        if len(counts) == 1:
            label_name = 'fake' if 1 in counts else 'real'
            confounding_warnings.append(
                f"Split '{split}' / Source '{src}' is exclusively {label_name} ({counts[list(counts.keys())[0]]} samples). "
                "Detector may learn source shortcuts rather than facial manipulation artifacts."
            )

    report = {
        'status': 'complete',
        'training_selection': {
            'mode': 'all' if full_training else 'balanced_subset',
            'requested_train_size': 'all' if full_training else target_real + target_fake,
            'selected_train_size': cur_real + cur_fake,
            'selection_order': 'seeded connected-group order; not filename order',
            'images_copied': False,
            'evaluation_membership': 'preserved from full pool',
        },
        'shortage_summary': shortage_report,
        'splits': splits_dict,
        'split_sample_counts': dict(split_counts),
        'source_class_distribution': [
            {'split': s, 'source': src, 'real': c[0], 'fake': c[1]}
            for (s, src), c in sorted(source_class.items())
        ],
        'confounding_warnings': confounding_warnings,
    }
    return output_rows, report


def grouped_split(rows, seed=42, holdout_source=None, holdout_generator=None):
    """Legacy connected-group splitting preserving existing behavior."""
    comp_map = connected_components(rows, require_groups=True)
    for group_hash, indices in comp_map.items():
        u = int(hashlib.sha256(f'{seed}:{group_hash}'.encode()).hexdigest()[:12], 16) / 16**12
        split = 'train' if u < .8 else ('dev' if u < .9 else 'internal_test')
        if any((holdout_source and rows[i]['dataset_source'] == holdout_source) or
               (holdout_generator and rows[i].get('generator') == holdout_generator) for i in indices):
            split = 'external_dev'
        for i in indices:
            rows[i]['group_id'] = group_hash
            rows[i]['split'] = split
    return rows


def near_duplicates(rows, distance=4):
    """64-bit dHash candidate audit with block indexing; manual confirmation needed."""
    if not 0 <= distance <= 8:
        raise ValueError('dHash distance must be between 0 and 8')
    buckets, suspects = defaultdict(list), []
    parts = distance + 1
    boundaries = np.linspace(0, 64, parts + 1, dtype=int)
    hashes = []
    for i, row in enumerate(rows):
        with Image.open(row['path']) as image:
            grey = np.asarray(image.convert('L').resize((9, 8)))
        bits = (grey[:, 1:] > grey[:, :-1]).flatten()
        value = int(''.join('1' if bit else '0' for bit in bits), 2)
        candidates = set()
        for part, (start, end) in enumerate(zip(boundaries[:-1], boundaries[1:])):
            token = (part, (value >> int(start)) & ((1 << int(end-start)) - 1))
            candidates.update(buckets[token])
            buckets[token].append(i)
        for j in candidates:
            if rows[j]['split'] != row['split'] and (value ^ hashes[j]).bit_count() <= distance:
                suspects.append({'first': rows[j]['sample_id'], 'second': row['sample_id'],
                                 'distance': (value ^ hashes[j]).bit_count()})
        hashes.append(value)
    return {'algorithm': '64-bit dHash', 'distance': distance, 'cross_split_candidates': suspects,
            'note': 'Candidate matches need review; no matches does not prove absence of every near-duplicate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', nargs='?', default='pilot', choices=['inventory', 'adapt', 'split', 'pilot', 'audit'],
                        help="Action to execute: inventory, adapt, split, pilot, or audit (default: pilot)")
    parser.add_argument('--root', '--source_dir', dest='root', help='Dataset root directory or source folder')
    parser.add_argument('--manifest', help='Input manifest CSV for adapt/split/pilot/audit')
    parser.add_argument('--output', '--output_manifest', dest='output', required=True, help='Output path')
    parser.add_argument('--source', help='Verified dataset name for inventory')
    parser.add_argument('--adapter', choices=list(ADAPTERS.keys()), help='Metadata adapter to apply')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--hashes', action='store_true')
    parser.add_argument('--hash_cache', help='Path to persistent hash cache JSON file')
    parser.add_argument('--near_duplicates', action='store_true')
    parser.add_argument('--holdout_source')
    parser.add_argument('--holdout_generator')
    parser.add_argument('--train_size', help="Training images from the same pool: even count such as 100000, 200000, 400000, or 'all'. No images copied. All disables frame caps; dev/test remain excluded.")
    parser.add_argument('--target_real', type=int, default=None)
    parser.add_argument('--target_fake', type=int, default=None)
    parser.add_argument('--allow_shortfall', action='store_true', default=False,
                        help='Allow pilot selection to succeed even if target_real or target_fake counts are not met')
    parser.add_argument('--allow_holdout_override', action='store_true', default=False,
                        help='Allow holdout source/generator to override pre-assigned train splits instead of failing')
    parser.add_argument('--frame_cap', type=int, default=30)
    parser.add_argument('--dev_ratio', type=float, default=0.1, help='Fraction of pool reserved for development split')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Fraction of pool reserved for internal test split')
    parser.add_argument('--require_groups', action='store_true', default=True, help='Require auditable group metadata')
    parser.add_argument('--quotas_file', help='JSON file containing source/generator quotas')
    parser.add_argument('--independent_images', action='store_true', default=False,
                        help='Mark images as independent groups/videos (only valid for verified independent still photographs)')
    parser.add_argument('--remap_path_prefix', type=str, nargs='*', default=[],
                        help='Old and new path prefixes separated by = (e.g. C:/old=/new/path)')
    parser.add_argument('--source_roots', type=str,
                        help='JSON string or path to JSON file mapping dataset_source names to root directory paths')
    args = parser.parse_args()
    if args.train_size is not None:
        if args.action != 'pilot':
            parser.error('--train_size is supported only for the pilot selection action')
        if args.target_real is not None or args.target_fake is not None:
            parser.error('Choose --train_size OR --target_real/--target_fake, not both')
        size_text = args.train_size.strip().strip("'\"").lower().replace('_', '')
        if size_text != 'all' and (not size_text.isdecimal() or int(size_text) <= 0 or int(size_text) % 2):
            parser.error('--train_size must be a positive even image count or all')
        args.train_size = size_text

    root = Path(args.root).resolve() if args.root else None
    hash_cache = (SQLiteHashCache(args.hash_cache) if Path(args.hash_cache).suffix.lower() in {'.sqlite', '.db'}
                  else HashCache(args.hash_cache)) if args.hash_cache and args.hashes else None

    if args.action == 'inventory':
        if not root:
            raise ValueError('--root or --source_dir required for inventory')
        if not args.source:
            raise ValueError('--source is required; do not label several repositories as one source')
        rows = []
        print(f'Inventory: scanning directory entries under {root}; this may take time for a large pool...', flush=True)
        for image in sorted(root.rglob('*')):
            if image.suffix.lower() not in {'.png', '.jpg', '.jpeg', '.bmp', '.webp'}:
                continue
            rel_path = image.relative_to(root).as_posix()
            parts = image.relative_to(root).parts
            lbl = None
            split_hint = 'unassigned'

            # 1. Flat layout: root / real / img.png
            if parts[0] in {'real', 'fake', '0_real', '1_fake'}:
                lbl = int(parts[0] in {'fake', '1_fake'})
            # 2. Split layout: root / train / real / img.png or root / val / fake / img.png
            elif len(parts) > 1 and parts[1] in {'real', 'fake', '0_real', '1_fake'}:
                lbl = int(parts[1] in {'fake', '1_fake'})
                p0_lower = parts[0].lower()
                split_hint = 'dev' if p0_lower in {'val', 'dev', 'validation'} else ('train' if p0_lower == 'train' else 'internal_test')
            else:
                # 3. Search path parts for unambiguous class indicator
                parts_lower = [p.lower() for p in parts]
                if 'fake' in parts_lower or '1_fake' in parts_lower:
                    lbl = 1
                elif 'real' in parts_lower or '0_real' in parts_lower:
                    lbl = 0

            if lbl is None:
                raise ValueError(
                    f'Cannot infer binary label for {image}; expected class folder (real/fake) '
                    'directly under root or under a split directory (e.g. train/real, val/fake).'
                )

            # Both identifiers include the source and FULL relative path (including
            # split, class, directories, and extension). Stems alone collide across
            # e.g. train/real/0001.png and val/fake/0001.jpg.
            identity_digest = hashlib.sha256(f"{args.source}:{rel_path}".encode()).hexdigest()

            # Never fabricate video/group IDs unless explicit independent photographs
            if args.independent_images:
                vid_id = 'none'
                grp_id = f"independent:{identity_digest}"
            else:
                vid_id = 'unknown'
                grp_id = 'unknown'

            file_hash = 'unknown'
            if args.hashes:
                file_hash = hash_cache.get_or_compute(str(image)) if hash_cache else sha256(image)

            rows.append(dict(
                sample_id=identity_digest[:16],
                path=rel_path,
                label=lbl,
                split=split_hint,
                dataset_source=args.source,
                group_id=grp_id,
                source_video_id=vid_id,
                identity_id='unknown',
                original_id='unknown',
                generator='authentic' if lbl == 0 else 'manipulated',
                sha256=file_hash,
            ))
            if len(rows) % 50000 == 0:
                print(f'Inventory progress: {len(rows):,} images recorded...', flush=True)
        if hash_cache:
            hash_cache.close()
        if not rows:
            raise ValueError('No images found')
        write_manifest(args.output, rows)
        print('Inventory written. Fill verified group/generator metadata before splitting.')
        return

    # Deterministic manifest loading path for adapt/split/pilot/audit
    manifest_path = args.manifest
    if not manifest_path and root and (root / 'manifest.csv').is_file():
        manifest_path = str(root / 'manifest.csv')

    if not manifest_path:
        raise ValueError(
            f"--manifest (or an existing manifest.csv under --root) is required for action '{args.action}'. "
            "Run 'inventory' and metadata enrichment first, or specify an existing manifest."
        )

    remap_prefixes = {}
    if args.remap_path_prefix:
        for item in args.remap_path_prefix:
            if '=' in item:
                old_p, new_p = item.split('=', 1)
                remap_prefixes[old_p] = new_p

    source_roots = None
    if args.source_roots:
        if Path(args.source_roots).is_file():
            source_roots = json.loads(Path(args.source_roots).read_text(encoding='utf-8'))
        else:
            source_roots = json.loads(args.source_roots)

    if args.action == 'pilot' and Path(args.output).resolve() == Path(manifest_path).resolve():
        raise ValueError('Selection output must differ from the pool manifest; keep the full pool reusable.')
    rows = read_manifest(manifest_path, root=root, remap_prefixes=remap_prefixes, source_roots=source_roots, progress_every=50000)

    if args.action == 'adapt':
        if not args.adapter:
            raise ValueError('--adapter is required for adapt action')
        for r in rows:
            apply_adapter(r, args.adapter)
        write_manifest(args.output, rows)
        print(f"Applied adapter '{args.adapter}' to {len(rows)} samples. Written to {args.output}")
        return

    if args.action == 'split':
        if any(r['split'] not in {'unassigned', 'pool'} for r in rows):
            raise ValueError('Split only an unassigned/pool manifest; preserve existing final holdouts')
        rows = grouped_split(rows, args.seed, args.holdout_source, args.holdout_generator)
        write_manifest(args.output, rows)
        return

    if args.action == 'pilot':
        quotas = {}
        if args.quotas_file:
            quotas = json.loads(Path(args.quotas_file).read_text(encoding='utf-8'))
        rows, report = build_pilot_100k(
            rows,
            target_real=args.target_real if args.target_real is not None else 50000,
            target_fake=args.target_fake if args.target_fake is not None else 50000,
            frame_cap=args.frame_cap,
            source_quotas=quotas.get('sources'),
            generator_quotas=quotas.get('generators'),
            dev_ratio=args.dev_ratio,
            test_ratio=args.test_ratio,
            seed=args.seed,
            holdout_source=args.holdout_source,
            holdout_generator=args.holdout_generator,
            require_groups=args.require_groups,
            allow_shortfall=args.allow_shortfall,
            allow_holdout_override=args.allow_holdout_override,
            train_size=args.train_size,
        )
        write_manifest(args.output, rows)
        report_path = Path(args.output).with_suffix('.report.json')
        report_path.write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f"Pilot manifest written to {args.output}")
        print(f"Pilot summary report written to {report_path}")
        print(json.dumps(report['shortage_summary'], indent=2))
        return

    try:
        report = audit_rows(rows, hash_files=args.hashes, hash_cache=hash_cache, progress_every=5000)
    finally:
        if hash_cache:
            hash_cache.close()
    if args.hashes:
        write_manifest(manifest_path, rows)

    manifest_digest = hashlib.sha256(Path(manifest_path).read_bytes()).hexdigest()
    report['manifest_sha256'] = manifest_digest

    if args.near_duplicates:
        report['near_duplicate_audit'] = near_duplicates(rows)
        report['near_duplicates_verified'] = False
        if report['near_duplicate_audit']['cross_split_candidates']:
            report['passed'] = False
            report['errors'].append('Cross-split near-duplicate candidates require review')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key != 'errors'}, indent=2), flush=True)
    if report['errors']:
        print(f"Audit failed with {len(report['errors']):,} issues; full details saved to {args.output}.", flush=True)
        print(json.dumps(report['errors'][:10], indent=2), flush=True)

    gate_path = Path(manifest_path).with_suffix('.verified.json')
    if report['passed']:
        gate_data = {
            'verified': True,
            'manifest_path': str(Path(manifest_path).resolve()),
            'manifest_sha256': manifest_digest,
            'audit_report': str(Path(args.output).resolve()),
            'samples': len(rows),
            'unique_groups': report.get('unique_groups'),
            'hashes_verified': report.get('hashes_verified', False),
            'classes': sorted(list({int(r['label']) for r in rows})),
            'splits': sorted(list({r['split'] for r in rows})),
        }
        gate_path.write_text(json.dumps(gate_data, indent=2), encoding='utf-8')
        print(f"Preparation success gate written to: {gate_path}")
    else:
        if gate_path.is_file():
            gate_path.unlink()
            print(f"Stale preparation gate removed: {gate_path}")
        raise SystemExit(1)


if __name__ == '__main__':
    main()
