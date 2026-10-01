"""Explicit binary labels and auditable source groups for forensic datasets."""
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

UNKNOWN = {
    '', 'unknown', 'none', 'nan', 'n/a', 'null',
    'n/a_photograph', 'n/a_synthesis', 'n/a_still'
}
GROUP_FIELDS = ('group_id', 'source_video_id', 'identity_id', 'original_id')
PROTECTED_SPLITS = {'test', 'internal_test', 'external_test', 'final_test'}


def known(value):
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in UNKNOWN:
        return False
    if s.startswith('n/a_') or s.startswith('na_'):
        return False
    return True


def extract_entity_tokens(key, value):
    """Extract individual entity tokens from single or delimiter-separated metadata fields."""
    if not known(value):
        return []
    # If field contains comma, semicolon, or pipe, extract individual tokens
    raw_parts = re.split(r'[,;|]', str(value))
    tokens = []
    for p in raw_parts:
        p_clean = p.strip()
        if known(p_clean):
            tokens.append(p_clean)
    return tokens


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def read_manifest(path, root=None, split=None, check_files=True, remap_prefixes=None, source_roots=None):
    path = Path(path).resolve()
    default_root = Path(root).resolve() if root else path.parent
    if remap_prefixes is None:
        remap_prefixes = {}
    elif isinstance(remap_prefixes, (list, tuple)):
        remap_prefixes = dict(remap_prefixes)
    if source_roots is None:
        source_roots = {}

    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = {'sample_id', 'path', 'label', 'split', 'dataset_source', 'group_id'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Manifest missing columns: {sorted(missing)}')
        rows, ids = [], set()
        for line, row in enumerate(reader, 2):
            if row['label'] not in ('0', '1'):
                raise ValueError(f'{path}:{line}: label must be 0 (real) or 1 (fake)')
            if not known(row['sample_id']) or row['sample_id'] in ids:
                raise ValueError(f'{path}:{line}: missing or duplicate sample_id')
            ids.add(row['sample_id'])
            if not known(row['split']):
                raise ValueError(f'{path}:{line}: split is required')
            if split is not None and row['split'] != split:
                continue

            raw_path_str = row['path'].replace('\\', '/')

            # 1. Apply explicit remap prefix if matches
            for old_p, new_p in remap_prefixes.items():
                old_norm = old_p.replace('\\', '/')
                new_norm = new_p.replace('\\', '/')
                if raw_path_str.startswith(old_norm):
                    raw_path_str = new_norm + raw_path_str[len(old_norm):]
                    break

            # 2. Determine effective root based on dataset_source
            src = row.get('dataset_source')
            effective_root = Path(source_roots[src]).resolve() if (src and src in source_roots) else default_root

            # 3. Resolve path
            is_win_drive = len(raw_path_str) > 2 and raw_path_str[1] == ':' and raw_path_str[2] == '/'
            is_posix_abs = raw_path_str.startswith('/')
            is_absolute = is_win_drive or is_posix_abs

            if not is_absolute:
                # Relative path: ALWAYS resolve under effective_root deterministically.
                # Never check against current working directory first to prevent accidental collisions.
                image_path = (effective_root / raw_path_str).resolve()
            else:
                cand_path = Path(raw_path_str)
                if cand_path.is_file():
                    image_path = cand_path.resolve()
                else:
                    if check_files:
                        raise FileNotFoundError(
                            f"Could not resolve image: absolute path '{row['path']}' does not exist on this system. "
                            "Provide explicit relocation mapping via 'remap_prefixes' (--remap_path_prefix) "
                            f"or 'source_roots' for dataset_source '{src}' instead of relying on ambiguous suffix guessing."
                        )
                    image_path = cand_path

            if check_files and not image_path.is_file():
                raise FileNotFoundError(f"Could not resolve image at '{image_path}' (source path: '{row['path']}', root: '{effective_root}')")
            row.update(path=str(image_path), label=int(row['label']))
            rows.append(row)
    if not rows:
        raise ValueError(f'No samples in {path} for split={split!r}')
    return rows


def audit_rows(rows, require_groups=True, hash_files=False):
    """IDs must be globally namespaced; linked originals/derivatives share groups."""
    errors, warnings = [], []
    seen = {key: {} for key in ('sample_id', 'path', 'sha256') + GROUP_FIELDS}
    counts, source_labels = Counter(), defaultdict(Counter)
    for row in rows:
        split = row['split']
        counts[(split, int(row['label']))] += 1
        source_labels[(split, row.get('dataset_source', 'unknown'))][int(row['label'])] += 1
        if require_groups and not known(row.get('group_id')):
            errors.append(f"Missing group_id: {row['sample_id']}")
        if not known(row.get('dataset_source')):
            errors.append(f"Missing dataset_source: {row['sample_id']}")
        if hash_files:
            actual = sha256(row['path'])
            if known(row.get('sha256')) and row['sha256'] != actual:
                errors.append(f"Content hash mismatch: {row['sample_id']}")
            row['sha256'] = actual
        for key, values in seen.items():
            val = row.get(key)
            tokens = [str(Path(val).resolve()).casefold()] if (key == 'path' and known(val)) else extract_entity_tokens(key, val)
            for token in tokens:
                previous = values.get(token)
                if previous and previous != split:
                    errors.append(f'{key} overlaps {previous} and {split}: {token}')
                values[token] = split
    for (split, source), labels in sorted(source_labels.items()):
        if len(labels) == 1:
            warnings.append(f'{split}/{source} contains only label {next(iter(labels))}; inspect source shortcuts')
    for split in {r['split'] for r in rows}:
        if {int(r['label']) for r in rows if r['split'] == split} != {0, 1}:
            warnings.append(f'{split} has one class; ROC AUC is undefined')
    return {'passed': not errors, 'errors': errors, 'warnings': warnings,
            'counts': [{'split': s, 'label': c, 'count': n} for (s, c), n in sorted(counts.items())],
            'metadata_coverage': {key: sum(known(r.get(key)) for r in rows) for key in GROUP_FIELDS + ('sha256', 'generator')},
            'samples': len(rows), 'unique_groups': len({r['group_id'] for r in rows if known(r.get('group_id'))}),
            'hashes_verified': hash_files, 'near_duplicates_verified': False,
            'source_class_counts': [{'split': s, 'source': src, 'real': c[0], 'fake': c[1]}
                                    for (s, src), c in sorted(source_labels.items())]}


def write_manifest(path, rows):
    fields = list(dict.fromkeys(k for row in rows for k in row))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_manifest_gate(manifest_path, gate_path=None, enforce_class_coverage=True, require_hashes=False):
    """Verify that a manifest has successfully passed preparation audit and matches verified gate digest.

    Fails closed (raising RuntimeError) if:
    - manifest file does not exist
    - gate file does not exist
    - gate file records verified=False
    - current manifest content sha256 does not match gate manifest_sha256
    - require_hashes is True and gate_data does not record hashes_verified=True
    - enforce_class_coverage is True and manifest does not contain both classes (0 and 1)
    """
    m_path = Path(manifest_path).resolve()
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {m_path}")

    g_path = Path(gate_path).resolve() if gate_path else m_path.with_suffix('.verified.json')
    if not g_path.is_file():
        raise RuntimeError(
            f"Preparation gate not found: {g_path}.\n"
            f"You must run 'python prepare_dataset.py audit --manifest {m_path} --hashes' "
            "and ensure it passes before starting training."
        )

    try:
        gate_data = json.loads(g_path.read_text(encoding='utf-8'))
    except Exception as exc:
        raise RuntimeError(f"Corrupted preparation gate file {g_path}: {exc}") from exc

    if not gate_data.get('verified', False):
        raise RuntimeError(
            f"Preparation gate {g_path} indicates audit failed or is unverified. "
            "Training is blocked until data audit passes."
        )

    expected_digest = gate_data.get('manifest_sha256')
    if not expected_digest:
        raise RuntimeError(f"Preparation gate {g_path} is missing 'manifest_sha256' field.")

    actual_digest = hashlib.sha256(m_path.read_bytes()).hexdigest()
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"Manifest digest mismatch between manifest file and audit gate!\n"
            f"  Manifest:        {m_path}\n"
            f"  Current SHA256:  {actual_digest}\n"
            f"  Gate SHA256:     {expected_digest}\n"
            "The manifest was modified after the audit passed, or a stale gate file is present.\n"
            "Re-run 'python prepare_dataset.py audit' to re-verify."
        )

    if require_hashes and not gate_data.get('hashes_verified', False):
        raise RuntimeError(
            f"Preparation gate {g_path} was generated without verifying content hashes. "
            "Strict mode requires running 'python prepare_dataset.py audit --hashes' before training."
        )

    if enforce_class_coverage:
        classes = set()
        if 'classes' in gate_data and isinstance(gate_data['classes'], (list, tuple)):
            classes = set(gate_data['classes'])
        elif 'counts' in gate_data and isinstance(gate_data['counts'], list):
            classes = {c.get('label') for c in gate_data['counts'] if 'label' in c}
        else:
            with m_path.open(newline='', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if 'label' in r and r['label'] in ('0', '1'):
                        classes.add(int(r['label']))
        if classes != {0, 1}:
            raise RuntimeError(
                f"Preparation gate {g_path} failed class coverage enforcement: found classes {sorted(classes)}. "
                "Both real (0) and fake (1) samples must be present."
            )

    return gate_data
