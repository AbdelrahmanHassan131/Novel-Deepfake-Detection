"""DRCT Data Preparation: Inventory, Resumable Reconstruction Pipeline, and Readiness Audit.

Provides command-line and programmatic tools to:
1. Inventory source datasets (manifests or directory structures) into auditable DRCT base manifests.
2. Resumably generate or import reconstruction pairs with strict parent-child provenance and atomic writes.
3. Audit DRCT readiness and emit canonical manifest-digest-bound verification gates (.verified.json).
"""
import argparse
import csv
import hashlib
import json
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Union
from PIL import Image

from data.drct_manifest import (
    compute_reconstruction_cache_key,
    file_sha256,
    read_drct_manifest,
    validate_drct_records,
    write_drct_manifest,
    DRCT_REQUIRED_COLUMNS,
)


def generate_gate_sidecar(
    manifest_path: Union[str, Path],
    records: Optional[List[Dict[str, Any]]] = None,
    require_hashes: bool = False,
) -> Path:
    """Run canonical audit and write a manifest-digest-bound .verified.json gate sidecar."""
    m_path = Path(manifest_path).resolve()
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest not found for gate generation: {m_path}")

    # Audit the serialized artifact, never a possibly different caller list.
    gate_path = m_path.with_suffix('.verified.json')
    try:
        records = read_drct_manifest(m_path)
        validation = validate_drct_records(records)
        if not validation['passed'] or not records:
            raise ValueError(f"Invalid DRCT manifest: {validation['errors'][:10]}")
        for row in records:
            if not Path(row['path']).is_file():
                raise FileNotFoundError(f"Image missing: {row['path']}")
        from data.manifest import audit_rows
        report = audit_rows(records, hash_files=require_hashes)
        if not report['passed']:
            raise ValueError(f"Manifest gate audit failed: {'; '.join(report['errors'][:10])}")
        if require_hashes:
            # audit_rows fills missing hashes; persist them before binding the gate.
            write_drct_manifest(m_path, records)
    except Exception:
        if gate_path.is_file():
            gate_path.unlink()
        raise

    raw_bytes = m_path.read_bytes()
    manifest_digest = hashlib.sha256(raw_bytes).hexdigest()

    gate_path = m_path.with_suffix('.verified.json')
    unique_groups = report.get('unique_groups')
    if unique_groups is None:
        unique_groups = len({r.get('group_id') or r.get('sample_id') for r in records})

    gate_data = {
        'verified': True,
        'manifest_path': str(m_path),
        'manifest_sha256': manifest_digest,
        'audit_report': 'canonical_inline_audit',
        'samples': len(records),
        'unique_groups': unique_groups,
        'hashes_verified': require_hashes,
        'classes': sorted(list({int(r['label']) for r in records})),
        'splits': sorted(list({str(r.get('split', 'train')) for r in records})),
    }
    temporary = gate_path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(gate_data, indent=2), encoding='utf-8')
    os.replace(temporary, gate_path)
    return gate_path


def inventory_dataset_directory(
    dataset_root: Union[str, Path],
    output_manifest_path: Union[str, Path],
    split: Optional[str] = None,
    splits: Optional[List[str]] = None,
    source_name: Optional[str] = None,
    map_val_to_dev: bool = True,
    generate_gate: bool = True,
    duplicate_policy: str = 'error',
) -> Dict[str, Any]:
    """Inventory a standard dataset root into a DRCT base manifest.
    
    Standardizes split names to ['train', 'dev'] (or explicit splits),
    scans class folders recursively, produces collision-safe sample IDs,
    verifies non-empty binary coverage, and writes the verification gate.
    With duplicate_policy='drop_train', exclude training rows whose exact
    content also exists in evaluation, only when all copies agree on label.
    Image files and evaluation rows are never deleted or relabeled.
    """
    root = Path(dataset_root).resolve()
    out_path = Path(output_manifest_path).resolve()
    if duplicate_policy not in ('error', 'drop_train'):
        raise ValueError("duplicate_policy must be 'error' or 'drop_train'")
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")

    if split is not None:
        target_splits = [split]
    elif splits is not None:
        target_splits = splits
    else:
        target_splits = ['train', 'dev']

    src_name = source_name or str(root.name)
    records: List[Dict[str, Any]] = []

    real_aliases = ('real', '0_real', '0')
    fake_aliases = ('fake', '1_fake', '1')

    seen_sids: Set[str] = set()

    for s in target_splits:
        # Resolve split directory on disk: support 'dev' mapped to 'val' if 'dev' does not exist
        split_dir = root / s
        if not split_dir.is_dir():
            if s == 'dev' and map_val_to_dev and (root / 'val').is_dir():
                split_dir = root / 'val'
            elif s == 'val' and (root / 'dev').is_dir():
                split_dir = root / 'dev'
            elif len(target_splits) == 1 and any((root / a).is_dir() for a in real_aliases + fake_aliases):
                split_dir = root
            else:
                raise FileNotFoundError(
                    f"Requested split directory '{s}' not found in dataset root '{root}'. "
                    f"Available folders: {[d.name for d in root.iterdir() if d.is_dir()]}"
                )

        split_records_count = 0
        for aliases, label, kind in [(real_aliases, 0, 'real_original'), (fake_aliases, 1, 'fake_original')]:
            cls_found = False
            for alias in aliases:
                cls_dir = split_dir / alias
                if not cls_dir.is_dir():
                    continue
                cls_found = True
                for img_path in sorted(cls_dir.rglob('*.*')):
                    if img_path.suffix.lower() not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp'):
                        continue
                    f_hash = file_sha256(img_path)
                    # Use split_stem, and disambiguate with label and hash if duplicate stems exist across classes/extensions
                    base_sid = f"{s}_{img_path.stem}"
                    if base_sid in seen_sids:
                        path_id = hashlib.sha256(str(img_path.relative_to(root)).encode()).hexdigest()[:16]
                        sid = f"{s}_{label}_{img_path.stem}_{path_id}"
                    else:
                        sid = base_sid
                    seen_sids.add(sid)

                    rec = {
                        'sample_id': sid,
                        'parent_id': sid,
                        'path': str(img_path.resolve()),
                        'label': label,
                        'split': s,
                        'sample_kind': kind,
                        'group_id': sid,
                        'dataset_source': src_name,
                        'sha256': f_hash,
                        'rec_model_revision': 'none',
                        'rec_settings': '{}',
                        'rec_seed': '0',
                        'grouping_basis': 'image_level_unverified',
                    }
                    records.append(rec)
                    split_records_count += 1
                    if len(records) % 10000 == 0:
                        print(f'[InventoryDir] Hashed {len(records):,} images...', flush=True)
                break

        if split_records_count == 0:
            raise ValueError(f"Requested split '{s}' contains 0 valid images in {split_dir}.")

    scanned_count = len(records)
    excluded = []
    hashes = {}
    for row in records:
        hashes.setdefault(row['sha256'], []).append(row)
    overlapping = {h: rows for h, rows in hashes.items()
                   if len({r['split'] for r in rows}) > 1}
    if duplicate_policy == 'drop_train':
        removable_hashes = {h for h, rows in overlapping.items()
                            if len({r['label'] for r in rows}) == 1}
        kept = []
        for row in records:
            if row['split'] == 'train' and row['sha256'] in removable_hashes:
                excluded.append(dict(row, exclusion_reason='exact_duplicate_in_evaluation'))
            else:
                kept.append(row)
        records = kept

    # Always write an exclusion trail, even if remaining conflicts prevent use.
    excluded_path = out_path.with_suffix('.excluded_duplicates.csv')
    write_drct_manifest(excluded_path, excluded)
    report_path = out_path.with_suffix('.inventory_report.json')
    counts = {}
    for row in records:
        key = (row['split'], int(row['label']))
        counts[key] = counts.get(key, 0) + 1
    cleanup_report = {
        'status': 'failed_validation', 'duplicate_policy': duplicate_policy,
        'scanned_samples': scanned_count, 'retained_samples': len(records),
        'cross_split_content_hashes': len(overlapping),
        'excluded_training_samples': len(excluded),
        'evaluation_rows_preserved': True, 'image_files_changed': False,
        'excluded_rows_csv': str(excluded_path),
        'counts': [dict(split=s, label=l, count=n) for (s, l), n in sorted(counts.items())],
    }
    # Refuse empty splits, contradictory labels, or remaining evaluation overlap.
    val_res = validate_drct_records(records)
    for s in target_splits:
        labels = {label for (row_split, label) in counts if row_split == s}
        if labels != {0, 1}:
            val_res['errors'].append(f"Split '{s}' must retain both real (0) and fake (1) images; found {labels}.")
            val_res['passed'] = False
    cleanup_report['errors'] = val_res['errors']
    report_path.write_text(json.dumps(cleanup_report, indent=2), encoding='utf-8')
    if not val_res['passed']:
        raise ValueError(f"Directory inventory validation failed: {'; '.join(val_res['errors'][:10])}. "
                         f"Report: {report_path}. For same-label train/evaluation duplicates, explicitly use duplicate_policy='drop_train'.")

    write_drct_manifest(out_path, records)

    if generate_gate:
        generate_gate_sidecar(out_path, records, require_hashes=False)

    cleanup_report['status'] = 'success'
    cleanup_report['manifest_sha256'] = file_sha256(out_path)
    report_path.write_text(json.dumps(cleanup_report, indent=2), encoding='utf-8')
    print(f'[InventoryDir] Retained {len(records):,}; excluded {len(excluded):,} training duplicates. Report: {report_path}', flush=True)

    real_count = sum(1 for r in records if r['label'] == 0)
    fake_count = sum(1 for r in records if r['label'] == 1)
    return {
        'status': 'success',
        'output_manifest': str(out_path),
        'total_inventoried': len(records),
        'total_samples': len(records),
        'real_count': real_count,
        'fake_count': fake_count,
        'splits': sorted(list(set(r['split'] for r in records))),
        'scanned_samples': scanned_count,
        'excluded_training_samples': len(excluded),
        'inventory_report': str(report_path),
        'excluded_rows_csv': str(excluded_path),
    }


def inventory_source_manifest(
    source_manifest_path: Union[str, Path],
    output_manifest_path: Union[str, Path],
    root: Optional[Union[str, Path]] = None,
    generate_gate: bool = True,
) -> Dict[str, Any]:
    """Convert an existing audited CSV manifest into a DRCT base manifest preserving provenance."""
    src_path = Path(source_manifest_path).resolve()
    out_path = Path(output_manifest_path).resolve()

    if not src_path.is_file():
        raise FileNotFoundError(f"Source manifest not found: {src_path}")

    records = []
    with open(src_path, mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            sid = row.get('sample_id') or Path(row['path']).stem
            lbl = int(row['label'])
            kind = 'real_original' if lbl == 0 else 'fake_original'
            
            f_hash = row.get('sha256') or ''
            resolved_img = Path(row['path'])
            if not resolved_img.is_absolute() and root is not None:
                resolved_img = Path(root) / resolved_img
            if not f_hash and resolved_img.is_file():
                f_hash = file_sha256(resolved_img)

            rec = {
                'sample_id': sid,
                'parent_id': sid,
                'path': str(resolved_img) if resolved_img.is_file() else row['path'],
                'label': lbl,
                'split': row.get('split', 'train'),
                'sample_kind': kind,
                'group_id': row.get('group_id') or sid,
                'dataset_source': row.get('dataset_source', 'source'),
                'sha256': f_hash,
                'rec_model_revision': 'none',
                'rec_settings': '{}',
                'rec_seed': '0',
            }
            # Preserve original extended metadata if present
            for extra_key in ('source_video_id', 'identity_id', 'original_id', 'generator'):
                if extra_key in row:
                    rec[extra_key] = row[extra_key]

            records.append(rec)

    val_res = validate_drct_records(records)
    if not val_res['passed']:
        raise ValueError(f"Inventory validation failed: {'; '.join(val_res['errors'][:10])}")

    write_drct_manifest(out_path, records)

    if generate_gate:
        generate_gate_sidecar(out_path, records, require_hashes=False)

    return {
        'status': 'success',
        'output_manifest': str(out_path),
        'total_inventoried': len(records),
    }


def import_precomputed_reconstructions(
    inventory_manifest_path: Union[str, Path],
    reconstruction_dir: Union[str, Path],
    output_manifest_path: Union[str, Path],
    target_split: str = 'train',
    rec_model_revision: str = 'precomputed_v1',
    rec_settings: Optional[Dict[str, Any]] = None,
    rec_seed: int = 0,
    reconstruct_fakes: bool = False,
    generate_gate: bool = True,
    mapping_csv: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """Import precomputed reconstructed images matching base inventory records.
    
    Supports both real and fake reconstructions, verifies image files on disk,
    prevents ambiguous parent-child linkages, and emits the verification gate.
    """
    inv_path = Path(inventory_manifest_path).resolve()
    rec_dir = Path(reconstruction_dir).resolve()
    out_manifest = Path(output_manifest_path).resolve()

    if not rec_dir.is_dir():
        raise FileNotFoundError(f"Reconstruction directory not found: {rec_dir}")

    records = read_drct_manifest(inv_path)
    existing_by_id = {r['sample_id']: dict(r) for r in records}
    explicit = {}
    if mapping_csv is not None:
        with open(mapping_csv, newline='', encoding='utf-8-sig') as stream:
            for item in csv.DictReader(stream):
                required = ('parent_id', 'path', 'parent_sha256', 'sha256', 'rec_model_revision', 'rec_settings', 'rec_seed')
                if any(not item.get(k) for k in required):
                    raise ValueError(f"Mapping requires nonempty columns: {required}")
                pid = item['parent_id']
                if pid in explicit or pid not in existing_by_id:
                    raise ValueError(f"Duplicate or unknown mapping parent: {pid}")
                parent = existing_by_id[pid]
                if parent['split'] != target_split or not parent['sample_kind'].endswith('_original'):
                    raise ValueError(f"Mapping parent is not an original in {target_split}: {pid}")
                if item['parent_sha256'] != parent.get('sha256') or file_sha256(parent['path']) != item['parent_sha256']:
                    raise ValueError(f"Mapping parent hash mismatch: {pid}")
                if not isinstance(json.loads(item['rec_settings']), dict):
                    raise ValueError('rec_settings must be a JSON object')
                int(item['rec_seed'])
                explicit[pid] = item

    # Index files in rec_dir by exact relative path, filename, and clean stem
    rec_files_by_stem: Dict[str, List[Path]] = {}
    for f in rec_dir.rglob('*.*'):
        if f.suffix.lower() in ('.png', '.jpg', '.jpeg', '.webp'):
            rec_files_by_stem.setdefault(f.name.lower(), []).append(f)
            rec_files_by_stem.setdefault(f.stem.lower(), []).append(f)
            clean_stem = f.stem.lower().replace('_rec', '').replace('-rec', '')
            if clean_stem != f.stem.lower():
                rec_files_by_stem.setdefault(clean_stem, []).append(f)

    settings_str = json.dumps(rec_settings or {}, sort_keys=True)
    matched_count = 0
    missing_count = 0
    ambiguous_count = 0
    used_paths = {}

    for r in records:
        if r.get('split') != target_split:
            continue
        kind = r.get('sample_kind')
        if kind != 'real_original' and not (reconstruct_fakes and kind == 'fake_original'):
            continue

        pid = r['sample_id']
        parent_path = Path(r['path'])
        parent_stem = parent_path.stem.lower()

        candidates: List[Path] = []
        for key in (() if mapping_csv else (pid.lower(), f"{pid.lower()}_rec", parent_path.name.lower(), parent_stem, f"{parent_stem}_rec")):
            if key in rec_files_by_stem:
                for c in rec_files_by_stem[key]:
                    if c not in candidates:
                        candidates.append(c)
        if pid in explicit:
            candidate = Path(explicit[pid]['path'])
            candidates = [candidate if candidate.is_absolute() else rec_dir / candidate]

        if not candidates:
            missing_count += 1
            continue

        if len(candidates) > 1:
            # Check if all candidates point to the same file or distinct files
            distinct = {c.resolve() for c in candidates}
            if len(distinct) > 1:
                ambiguous_count += 1
                raise ValueError(
                    f"Ambiguous reconstruction match for parent '{pid}': "
                    f"matched multiple candidates: {[str(c) for c in distinct]}"
                )

        rec_file = candidates[0]
        file_key = str(rec_file.resolve())
        if file_key in used_paths and used_paths[file_key] != pid:
            raise ValueError(f"Reconstruction {rec_file} matched multiple parents: {used_paths[file_key]}, {pid}. Supply an explicit mapping.")
        used_paths[file_key] = pid
        if not rec_file.is_file() or rec_file.stat().st_size == 0:
            missing_count += 1
            continue

        with Image.open(rec_file) as img:
            img.verify()
        rec_hash = file_sha256(rec_file)
        if pid in explicit and explicit[pid]['sha256'] != rec_hash:
            raise ValueError(f"Mapping reconstruction hash mismatch: {pid}")
        rec_id = f"rec_{pid}_{rec_hash[:8]}"
        rec_kind = 'real_reconstruction' if kind == 'real_original' else 'fake_reconstruction'
        rec_row = {
            'sample_id': rec_id,
            'parent_id': pid,
            'path': str(rec_file.resolve()),
            'label': 1,
            'split': r['split'],
            'sample_kind': rec_kind,
            'group_id': r['group_id'],
            'dataset_source': r['dataset_source'],
            'sha256': rec_hash,
            'rec_model_revision': explicit.get(pid, {}).get('rec_model_revision', rec_model_revision),
            'rec_settings': explicit.get(pid, {}).get('rec_settings', settings_str),
            'rec_seed': explicit.get(pid, {}).get('rec_seed', str(rec_seed)),
            'parent_sha256': r.get('sha256', ''),
            'linkage_basis': 'explicit_mapping' if mapping_csv else 'filename_match_unverified',
        }
        for field in ('identity_id', 'source_video_id', 'original_id', 'generator', 'grouping_basis'):
            if field in r:
                rec_row[field] = r[field]
        existing_by_id[rec_id] = rec_row
        matched_count += 1

    all_records = list(existing_by_id.values())
    val_res = validate_drct_records(all_records)
    if not val_res['passed']:
        raise ValueError(f"Imported reconstruction validation failed: {'; '.join(val_res['errors'][:10])}")

    write_drct_manifest(out_manifest, all_records)

    if generate_gate:
        generate_gate_sidecar(out_manifest, all_records, require_hashes=False)

    return {
        'status': 'success',
        'matched_count': matched_count,
        'missing_count': missing_count,
        'ambiguous_count': ambiguous_count,
        'total_records': len(all_records),
        'output_manifest': str(out_manifest),
    }


def prepare_reconstructions(
    inventory_manifest_path: Union[str, Path],
    output_manifest_path: Union[str, Path],
    reconstruction_output_dir: Union[str, Path],
    target_split: str = 'train',
    rec_model_revision: str = 'sd21_unclip_v1',
    rec_settings: Optional[Dict[str, Any]] = None,
    rec_seed: int = 42,
    reconstruction_fn: Optional[Callable[[Path, Path, Dict[str, Any]], bool]] = None,
    reconstruct_fakes: bool = False,
    dry_run: bool = False,
    generate_gate: bool = True,
) -> Dict[str, Any]:
    """Generate or import reconstructions for DRCT training resumably with atomic writes."""
    inv_path = Path(inventory_manifest_path).resolve()
    out_manifest = Path(output_manifest_path).resolve()
    rec_dir = Path(reconstruction_output_dir).resolve()
    rec_dir.mkdir(parents=True, exist_ok=True)

    if rec_settings is None:
        rec_settings = {'steps': 20, 'guidance_scale': 7.5}
    settings_str = json.dumps(rec_settings, sort_keys=True)

    records = read_drct_manifest(inv_path)
    if reconstruction_fn is None and not dry_run:
        raise ValueError('No reconstruction backend supplied. Use import_rec --mapping_csv for precomputed images; the CLI does not generate diffusion images.')
    existing_records_by_id = {r['sample_id']: dict(r) for r in records}

    reused_count = 0
    generated_count = 0
    missing_count = 0

    for r in records:
        if r.get('split') != target_split:
            continue
        kind = r.get('sample_kind')
        if kind != 'real_original' and not (reconstruct_fakes and kind == 'fake_original'):
            continue

        pid = r['sample_id']
        parent_hash = r.get('sha256') or pid
        if not dry_run and file_sha256(r['path']) != parent_hash:
            raise ValueError(f"Parent content changed before reconstruction: {pid}")
        cache_key = compute_reconstruction_cache_key(
            parent_id=pid,
            parent_sha256=parent_hash,
            rec_model_revision=rec_model_revision,
            rec_settings=rec_settings,
            rec_seed=rec_seed,
        )

        rec_filename = f"{cache_key[:16]}_rec.png"
        rec_filepath = rec_dir / rec_filename
        rec_sample_id = f"rec_{pid}_{cache_key[:8]}"

        file_valid = False
        if rec_filepath.is_file() and rec_filepath.stat().st_size > 0:
            try:
                with Image.open(rec_filepath) as img:
                    img.verify()
                rec_hash = file_sha256(rec_filepath)
                previous = existing_records_by_id.get(rec_sample_id)
                if previous and previous.get('sha256') != rec_hash:
                    raise ValueError(f"Cached reconstruction hash mismatch: {rec_filepath}")
                file_valid = True
                reused_count += 1
            except Exception:
                file_valid = False

        if not file_valid:
            if dry_run:
                missing_count += 1
                continue
            if reconstruction_fn is not None:
                src_path = Path(r['path'])
                tmp_filepath = rec_filepath.with_suffix(f".tmp_{os.getpid()}_{cache_key[:6]}.png")
                fn_params = {**rec_settings, 'seed': rec_seed}
                success = reconstruction_fn(src_path, tmp_filepath, fn_params)
                if success and tmp_filepath.is_file() and tmp_filepath.stat().st_size > 0:
                    try:
                        with Image.open(tmp_filepath) as img:
                            img.verify()
                        os.replace(tmp_filepath, rec_filepath)
                        rec_hash = file_sha256(rec_filepath)
                        generated_count += 1
                        file_valid = True
                    except Exception:
                        if tmp_filepath.exists():
                            tmp_filepath.unlink()
                        missing_count += 1
                        continue
                else:
                    if tmp_filepath.exists():
                        tmp_filepath.unlink()
                    missing_count += 1
                    continue
            else:
                missing_count += 1
                continue

        rec_kind = 'real_reconstruction' if kind == 'real_original' else 'fake_reconstruction'
        rec_row = {
            'sample_id': rec_sample_id,
            'parent_id': pid,
            'path': str(rec_filepath.resolve()),
            'label': 1,
            'split': r['split'],
            'sample_kind': rec_kind,
            'group_id': r['group_id'],
            'dataset_source': r['dataset_source'],
            'sha256': rec_hash,
            'rec_model_revision': rec_model_revision,
            'rec_settings': settings_str,
            'rec_seed': str(rec_seed),
            'parent_sha256': r.get('sha256', ''),
            'linkage_basis': 'generator_callback',
        }
        for field in ('identity_id', 'source_video_id', 'original_id', 'generator', 'grouping_basis'):
            if field in r:
                rec_row[field] = r[field]
        existing_records_by_id[rec_sample_id] = rec_row

    all_records = list(existing_records_by_id.values())
    val_res = validate_drct_records(all_records)
    if not val_res['passed']:
        raise ValueError(f"Reconstruction validation failed: {'; '.join(val_res['errors'][:10])}")

    if not dry_run:
        write_drct_manifest(out_manifest, all_records)
        if generate_gate:
            generate_gate_sidecar(out_manifest, all_records, require_hashes=False)

    return {
        'status': 'success',
        'reused_count': reused_count,
        'generated_count': generated_count,
        'missing_count': missing_count,
        'total_records': len(all_records),
        'output_manifest': str(out_manifest),
    }


def check_drct_readiness(
    manifest_path: Union[str, Path],
    split: str = 'train',
) -> Dict[str, Any]:
    """Audit a manifest for DRCT training readiness, verifying disk files and decodability."""
    records = read_drct_manifest(manifest_path)
    val_res = validate_drct_records(records)
    if not val_res['passed']:
        return {
            'is_ready_for_full_drct': False,
            'is_ready_for_rgb_control': False,
            'error': '; '.join(val_res['errors']),
        }

    split_records = [r for r in records if r.get('split') == split]
    
    real_origs = [r for r in split_records if r.get('sample_kind') == 'real_original']
    fake_origs = [r for r in split_records if r.get('sample_kind') == 'fake_original']
    real_recs = [r for r in split_records if r.get('sample_kind') == 'real_reconstruction']
    fake_recs = [r for r in split_records if r.get('sample_kind') == 'fake_reconstruction']

    missing_image_files = []
    corrupt_image_files = []
    valid_real_recs = []
    valid_fake_recs = []

    for r in real_recs:
        p = Path(r['path'])
        if not p.is_file() or p.stat().st_size == 0:
            missing_image_files.append(str(p))
            continue
        try:
            with Image.open(p) as img:
                img.verify()
            valid_real_recs.append(r)
        except Exception:
            corrupt_image_files.append(str(p))

    for r in fake_recs:
        p = Path(r['path'])
        if not p.is_file() or p.stat().st_size == 0:
            missing_image_files.append(str(p))
            continue
        try:
            with Image.open(p) as img:
                img.verify()
            valid_fake_recs.append(r)
        except Exception:
            corrupt_image_files.append(str(p))

    valid_rec_parents = {r['parent_id'] for r in valid_real_recs}
    paired_real_origs = sum(1 for r in real_origs if r['sample_id'] in valid_rec_parents)
    missing_recs = len(real_origs) - paired_real_origs
    fake_rec_parents = {r['parent_id'] for r in valid_fake_recs}
    missing_fake_recs = sum(r['sample_id'] not in fake_rec_parents for r in fake_origs)
    unverified_links = sum(r.get('linkage_basis') not in ('explicit_mapping', 'generator_callback')
                           for r in real_recs + fake_recs)

    is_ready_full = (
        len(real_origs) > 0 and
        len(fake_origs) > 0 and
        len(valid_real_recs) > 0 and
        missing_recs == 0 and
        missing_fake_recs == 0 and
        unverified_links == 0 and
        len(missing_image_files) == 0 and
        len(corrupt_image_files) == 0
    )
    is_ready_rgb = (len(real_origs) > 0 and len(fake_origs) > 0)

    return {
        'split': split,
        'total_split_samples': len(split_records),
        'real_originals': len(real_origs),
        'fake_originals': len(fake_origs),
        'real_reconstructions_declared': len(real_recs),
        'real_reconstructions_valid_on_disk': len(valid_real_recs),
        'fake_reconstructions_declared': len(fake_recs),
        'fake_reconstructions_valid_on_disk': len(valid_fake_recs),
        'missing_image_files_count': len(missing_image_files),
        'corrupt_image_files_count': len(corrupt_image_files),
        'paired_real_originals': paired_real_origs,
        'missing_real_reconstructions': missing_recs,
        'missing_fake_reconstructions': missing_fake_recs,
        'unverified_reconstruction_links': unverified_links,
        'recipe': 'originals_and_reconstructions_of_both_classes',
        'is_ready_for_full_drct': is_ready_full,
        'is_ready_for_rgb_control': is_ready_rgb,
    }


def main():
    parser = argparse.ArgumentParser(description="DRCT Data Preparation and Audit CLI")
    subparsers = parser.add_subparsers(dest='command', required=True)

    # Inventory command
    inv_parser = subparsers.add_parser('inventory', help="Inventory a source dataset manifest into DRCT base format")
    inv_parser.add_argument('--manifest', '--source_manifest', dest='source_manifest', required=True, help="Input CSV manifest")
    inv_parser.add_argument('--output', '--output_manifest', dest='output_manifest', required=True, help="Output DRCT base manifest")
    inv_parser.add_argument('--dataroot', default=None, help="Image root directory for hash calculation")
    inv_parser.add_argument('--no_gate', action='store_true', help="Skip gate sidecar generation")

    # Inventory directory command
    dir_parser = subparsers.add_parser('inventory_dir', help="Inventory a dataset directory with train/val real/fake folders")
    dir_parser.add_argument('--dataset_root', required=True, help="Path to dataset root folder")
    dir_parser.add_argument('--output', '--output_manifest', dest='output_manifest', required=True, help="Output manifest path")
    dir_parser.add_argument('--splits', nargs='+', default=None, help="List of split names (default: ['train', 'dev'])")
    dir_parser.add_argument('--source_name', default=None, help="Explicit dataset source name")
    dir_parser.add_argument('--no_gate', action='store_true', help="Skip gate sidecar generation")
    dir_parser.add_argument('--duplicate_policy', choices=['error', 'drop_train'], default='error',
                            help='Explicitly exclude same-label training copies of evaluation images; never delete files')

    # Import precomputed reconstructions command
    import_parser = subparsers.add_parser('import_rec', help="Import precomputed reconstruction pairs")
    import_parser.add_argument('--manifest', required=True, help="Input DRCT base manifest")
    import_parser.add_argument('--reconstruction_dir', '--rec_dir', dest='reconstruction_dir', required=True, help="Directory containing reconstructions")
    import_parser.add_argument('--output', '--output_manifest', dest='output_manifest', required=True, help="Output manifest")
    import_parser.add_argument('--split', default='train', help="Split (default: train)")
    import_parser.add_argument('--reconstruct_fakes', action='store_true', help="Import fake reconstructions if available")
    import_parser.add_argument('--mapping_csv', help='Explicit parent/content/provenance mapping; paths relative to rec_dir or absolute')
    import_parser.add_argument('--no_gate', action='store_true', help="Skip gate sidecar generation")

    # Prepare command
    rec_parser = subparsers.add_parser('prepare', help="Generate or prepare reconstructions resumably")
    rec_parser.add_argument('--manifest', default=None, help="Path to DRCT input manifest")
    rec_parser.add_argument('--dataset_root', default=None, help="Optional dataset directory if inventorying directly")
    rec_parser.add_argument('--output', '--output_manifest', dest='output_manifest', required=True, help="Output updated manifest")
    rec_parser.add_argument('--reconstruction_dir', '--reconstruction_root', '--rec_dir', dest='rec_dir', default=None, help="Directory for reconstructed images")
    rec_parser.add_argument('--split', default='train', help="Split to prepare (e.g. train)")
    rec_parser.add_argument('--rec_model_revision', default='sd21_unclip_v1', help="Model revision identifier")
    rec_parser.add_argument('--reconstruct_fakes', action='store_true', help="Also generate/import fake reconstructions")
    rec_parser.add_argument('--dry_run', action='store_true', help="Check pending without generating")
    rec_parser.add_argument('--no_gate', action='store_true', help="Skip gate sidecar generation")

    # Audit command
    audit_parser = subparsers.add_parser('audit', help="Audit manifest readiness for DRCT training")
    audit_parser.add_argument('--manifest', required=True, help="Path to DRCT manifest to check")
    audit_parser.add_argument('--split', default='train', help="Split to audit")
    audit_parser.add_argument('--hashes', action='store_true', help="Verify disk file hashes during audit")
    audit_parser.add_argument('--require_ready', action='store_true', help="Exit with non-zero code if full DRCT is not ready")

    args = parser.parse_args()

    if args.command == 'inventory':
        res = inventory_source_manifest(
            args.source_manifest, args.output_manifest, root=args.dataroot, generate_gate=not args.no_gate
        )
        print(f"[Inventory] Success: {res['total_inventoried']} samples written to {res['output_manifest']}")

    elif args.command == 'inventory_dir':
        res = inventory_dataset_directory(
            args.dataset_root, args.output_manifest, splits=args.splits, source_name=args.source_name,
            generate_gate=not args.no_gate, duplicate_policy=args.duplicate_policy
        )
        print(f"[InventoryDir] Success: {res['total_inventoried']} samples written to {res['output_manifest']}")
        print(f"  Splits: {res['splits']}; Real: {res['real_count']}, Fake: {res['fake_count']}")

    elif args.command == 'import_rec':
        res = import_precomputed_reconstructions(
            inventory_manifest_path=args.manifest,
            reconstruction_dir=args.reconstruction_dir,
            output_manifest_path=args.output_manifest,
            target_split=args.split,
            reconstruct_fakes=args.reconstruct_fakes,
            generate_gate=not args.no_gate,
            mapping_csv=args.mapping_csv,
        )
        print(f"[ImportRec] Matched={res['matched_count']}, Missing={res['missing_count']}, Ambiguous={res['ambiguous_count']}, Total={res['total_records']}")

    elif args.command == 'prepare':
        manifest_to_use = args.manifest
        if manifest_to_use is None and args.dataset_root:
            tmp_inv = Path(args.output_manifest).with_suffix('.base_inv.csv')
            inventory_dataset_directory(args.dataset_root, tmp_inv)
            manifest_to_use = str(tmp_inv)

        if not manifest_to_use or not Path(manifest_to_use).is_file():
            raise FileNotFoundError(f"Base manifest not found or specified: {manifest_to_use}")

        rec_target_dir = args.rec_dir or str(Path(args.output_manifest).parent / 'reconstructions')
        res = prepare_reconstructions(
            inventory_manifest_path=manifest_to_use,
            output_manifest_path=args.output_manifest,
            reconstruction_output_dir=rec_target_dir,
            target_split=args.split,
            rec_model_revision=args.rec_model_revision,
            reconstruct_fakes=args.reconstruct_fakes,
            dry_run=args.dry_run,
            generate_gate=not args.no_gate,
        )
        print(f"[Prepare] Status: reused={res['reused_count']}, generated={res['generated_count']}, missing={res['missing_count']}")

    elif args.command == 'audit':
        res = check_drct_readiness(args.manifest, split=args.split)
        print(f"[Audit] Split: {res.get('split')}")
        print(f"  Real originals: {res.get('real_originals')}")
        print(f"  Fake originals: {res.get('fake_originals')}")
        print(f"  Real reconstructions (declared): {res.get('real_reconstructions_declared')}")
        print(f"  Real reconstructions (valid on disk): {res.get('real_reconstructions_valid_on_disk')}")
        print(f"  Fake reconstructions (declared): {res.get('fake_reconstructions_declared')}")
        print(f"  Fake reconstructions (valid on disk): {res.get('fake_reconstructions_valid_on_disk')}")
        print(f"  Missing reconstructions: {res.get('missing_real_reconstructions')}")
        print(f"  Full DRCT Ready: {res.get('is_ready_for_full_drct')}")
        print(f"  RGB Control Ready: {res.get('is_ready_for_rgb_control')}")
        
        # Also run canonical gate audit and update .verified.json
        try:
            gate_path = generate_gate_sidecar(args.manifest, require_hashes=args.hashes)
            print(f"  Canonical gate verified and written: {gate_path}")
        except Exception as e:
            print(f"  Canonical gate audit FAILED: {e}", file=sys.stderr)
            sys.exit(1)

        if args.require_ready and not res.get('is_ready_for_full_drct'):
            sys.exit(1)


if __name__ == '__main__':
    main()
