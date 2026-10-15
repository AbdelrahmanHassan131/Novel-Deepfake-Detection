"""DRCT Manifest Specification, Validation, and Utilities.

Manages manifest records for DRCT with explicit parent linkage, sample kind,
reconstruction metadata, and split integrity.
"""
import csv
import json
import os
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union


DRCT_SAMPLE_KINDS = {
    'real_original',
    'fake_original',
    'real_reconstruction',
    'fake_reconstruction',
}

DRCT_REQUIRED_COLUMNS = [
    'sample_id',
    'parent_id',
    'path',
    'label',
    'split',
    'sample_kind',
    'group_id',
    'dataset_source',
    'sha256',
    'rec_model_revision',
    'rec_settings',
    'rec_seed',
]


def file_sha256(path: Union[str, Path]) -> str:
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def compute_reconstruction_cache_key(
    parent_id: str,
    parent_sha256: str,
    rec_model_revision: str,
    rec_settings: Dict[str, Any],
    rec_seed: int = 0,
    seed: Optional[int] = None,
) -> str:
    """Generate a stable, deterministic cache key covering input identity and reconstruction settings."""
    effective_seed = seed if seed is not None else rec_seed
    payload = {
        'parent_id': str(parent_id),
        'parent_sha256': str(parent_sha256),
        'rec_model_revision': str(rec_model_revision),
        'rec_settings': rec_settings,
        'rec_seed': int(effective_seed),
    }
    raw = json.dumps(payload, sort_keys=True)
    return hashlib.sha256(raw.encode('utf-8')).hexdigest()



def validate_drct_records(
    records: List[Dict[str, Any]],
    protected_splits: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """Validate a set of DRCT manifest records.
    
    Ensures:
      1. Every sample has unique sample_id.
      2. Valid sample_kind in DRCT_SAMPLE_KINDS.
      3. Binary labels:
         - real_original: 0
         - fake_original: 1
         - real_reconstruction: 1 (reconstructed real is synthetic/hard fake)
         - fake_reconstruction: 1
      4. Parent linkage:
         - For reconstructions, parent_id must exist in records as an original.
         - Derivative strictly inherits parent's split and group_id.
         - Derivatives cannot cross splits.
         - Parents in protected/evaluation splits must NOT have reconstructions generated for training.
    """
    if protected_splits is None:
        protected_splits = {'test', 'internal_test', 'external_test', 'final_test', 'val_unseen'}

    errors = []
    sample_map: Dict[str, Dict[str, Any]] = {}
    id_seen = set()

    for r in records:
        sid = r.get('sample_id')
        if not sid:
            errors.append("Record missing 'sample_id'.")
            continue
        if sid in id_seen:
            errors.append(f"Duplicate sample_id: '{sid}'.")
        id_seen.add(sid)
        sample_map[sid] = r

        kind = r.get('sample_kind')
        if kind not in DRCT_SAMPLE_KINDS:
            errors.append(f"Sample '{sid}' has invalid sample_kind: '{kind}'. Allowed: {sorted(list(DRCT_SAMPLE_KINDS))}")

        # Check label consistency with sample kind
        try:
            lbl = int(r.get('label'))
        except (ValueError, TypeError):
            errors.append(f"Sample '{sid}' has non-integer label: {r.get('label')}")
            continue

        if kind == 'real_original' and lbl != 0:
            errors.append(f"Sample '{sid}' is real_original but has label {lbl} != 0.")
        elif kind in ('fake_original', 'real_reconstruction', 'fake_reconstruction') and lbl != 1:
            errors.append(f"Sample '{sid}' is {kind} but has label {lbl} != 1.")

    # Validate parent linkages, contradictory hashes, cross-split groups, and cross-split hashes
    hash_to_labels: Dict[str, Set[int]] = {}
    hash_to_splits: Dict[str, Set[str]] = {}
    group_to_splits: Dict[str, Set[str]] = {}

    for sid, r in sample_map.items():
        kind = r.get('sample_kind')
        pid = r.get('parent_id')
        split = r.get('split')
        group = r.get('group_id')
        h = r.get('sha256')
        src = str(r.get('dataset_source', '')).lower()

        # Track hash to labels to catch same-hash contradictory labels
        if h and str(r.get('label')) in ('0', '1'):
            hash_to_labels.setdefault(h, set()).add(int(r['label']))
            hash_to_splits.setdefault(h, set()).add(str(split))

        # Track group to splits to catch groups spanning multiple splits
        if group:
            group_to_splits.setdefault(group, set()).add(str(split))

        # External development collection must not be eligible as training parents
        is_external_dev = (
            'val to be deleted' in src
            or 'unseen' in src
            or 'external_dev' in src
            or split in ('external_dev', 'external_test', 'val_unseen')
        )

        if kind in ('real_reconstruction', 'fake_reconstruction'):
            if not pid or pid not in sample_map:
                errors.append(f"Reconstruction '{sid}' references missing parent_id: '{pid}'.")
                continue

            parent = sample_map[pid]
            parent_split = parent.get('split')
            parent_group = parent.get('group_id')
            parent_kind = parent.get('sample_kind')
            parent_src = str(parent.get('dataset_source', '')).lower()

            # Parent must be an original
            if parent_kind not in ('real_original', 'fake_original'):
                errors.append(f"Parent '{pid}' of '{sid}' is not an original sample (kind: '{parent_kind}').")
            elif kind != parent_kind.replace('_original', '_reconstruction'):
                errors.append(f"Reconstruction '{sid}' kind does not match parent '{pid}'.")

            # Must inherit split
            if split != parent_split:
                errors.append(f"Reconstruction '{sid}' split ('{split}') does not match parent split ('{parent_split}').")

            # Must inherit group
            if group != parent_group:
                errors.append(f"Reconstruction '{sid}' group ('{group}') does not match parent group ('{parent_group}').")

            # Parents in protected splits cannot be used to generate training reconstructions
            if parent_split in protected_splits and split not in protected_splits:
                errors.append(f"Parent '{pid}' in protected split '{parent_split}' cannot generate training derivatives.")

            # External development collection cannot be parent for training
            if split == 'train' and (
                'val to be deleted' in parent_src
                or 'unseen' in parent_src
                or 'external_dev' in parent_src
                or parent_split in ('external_dev', 'external_test', 'val_unseen')
            ):
                errors.append(f"Parent '{pid}' from external development collection cannot be used for training reconstruction.")

    # Check for same-hash contradictory labels
    for h, labels in hash_to_labels.items():
        if len(labels) > 1:
            errors.append(f"Contradictory labels for same content hash {h}: labels {sorted(list(labels))}")

    # Check for identical content hashes appearing across multiple splits (cross-split leakage)
    for h, splits in hash_to_splits.items():
        if len(splits) > 1:
            errors.append(f"Identical content hash {h} appears in multiple splits: {sorted(list(splits))}")

    # Check for groups spanning multiple splits
    for gid, splits in group_to_splits.items():
        if len(splits) > 1:
            errors.append(f"Group '{gid}' spans multiple splits: {sorted(list(splits))}")

    return {
        'passed': len(errors) == 0,
        'errors': errors,
        'total_records': len(records),
    }


def read_drct_manifest(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Read a CSV DRCT manifest into a list of row dicts."""
    path = Path(path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"DRCT manifest not found: {path}")

    with open(path, mode='r', newline='', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        missing = set(DRCT_REQUIRED_COLUMNS) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"DRCT manifest missing required columns: {sorted(list(missing))}")
        records = [dict(row) for row in reader]
    return records


def write_drct_manifest(path: Union[str, Path], records: List[Dict[str, Any]]) -> None:
    """Atomically write records to a CSV DRCT manifest."""
    path = Path(path).resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f".tmp_{os.getpid()}_{hashlib.md5(str(path).encode()).hexdigest()[:6]}")

    # Preserve identity/video/generator metadata, including future provenance
    # columns. Auditing an in-memory field which is then dropped is not evidence.
    columns = DRCT_REQUIRED_COLUMNS + sorted(set().union(*(r.keys() for r in records)) - set(DRCT_REQUIRED_COLUMNS))
    with open(tmp_path, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for r in records:
            # Ensure all required fields exist
            row = {col: r.get(col, '') for col in columns}
            writer.writerow(row)

    # Atomic replace
    os.replace(tmp_path, path)


class DRCTManifest:
    """Wrapper class for managing DRCT manifest records."""

    def __init__(self, records: List[Dict[str, Any]]):
        self.records = records

    @classmethod
    def load(cls, path: Union[str, Path]) -> "DRCTManifest":
        records = read_drct_manifest(path)
        return cls(records)

    def save(self, path: Union[str, Path]) -> None:
        write_drct_manifest(path, self.records)

    def get_split(self, split: str) -> List[Dict[str, Any]]:
        return [r for r in self.records if r.get('split') == split]

    def validate(self, protected_splits: Optional[Set[str]] = None) -> Dict[str, Any]:
        return validate_drct_records(self.records, protected_splits=protected_splits)

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self):
        return iter(self.records)

