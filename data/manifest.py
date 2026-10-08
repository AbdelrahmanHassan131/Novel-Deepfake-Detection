_MANIFEST_CACHE = {}
_FILE_HASH_CACHE = {}


def clear_manifest_cache():
    """Clear process-level manifest cache and file hash cache."""
    _MANIFEST_CACHE.clear()
    _FILE_HASH_CACHE.clear()

"""Explicit binary labels and auditable source groups for forensic datasets."""
import csv
import numpy as np
import hashlib
import json
import re
import time
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
    p = Path(path).resolve()
    stat = p.stat()
    cache_key = (str(p), stat.st_mtime_ns, stat.st_size)
    if cache_key in _FILE_HASH_CACHE:
        return _FILE_HASH_CACHE[cache_key]

    digest = hashlib.sha256()
    with open(p, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    val = digest.hexdigest()
    _FILE_HASH_CACHE[cache_key] = val
    return val


def read_manifest(path, root=None, split=None, check_files=True, remap_prefixes=None, source_roots=None, progress_every=0):
    path = Path(path).resolve()
    default_root = Path(root).resolve() if root else path.parent
    if remap_prefixes is None:
        remap_prefixes = {}
    elif isinstance(remap_prefixes, (list, tuple)):
        remap_prefixes = dict(remap_prefixes)
    if source_roots is None:
        source_roots = {}

    stat = path.stat()
    cache_key = (
        str(path),
        stat.st_mtime_ns,
        stat.st_size,
        str(default_root),
        str(split) if split is not None else '',
        bool(check_files),
        tuple(sorted(remap_prefixes.items())),
        tuple(sorted(source_roots.items())),
    )
    if cache_key in _MANIFEST_CACHE:
        return [dict(r) for r in _MANIFEST_CACHE[cache_key]]

    with path.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        missing = {'sample_id', 'path', 'label', 'split', 'dataset_source', 'group_id'} - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f'Manifest missing columns: {sorted(missing)}')
        rows, ids = [], set()
        last_progress = time.monotonic()
        if progress_every:
            print(f'Reading manifest and checking paths: {path}', flush=True)
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
                image_path = (effective_root / raw_path_str).resolve() if check_files else (effective_root / raw_path_str)
            else:
                cand_path = Path(raw_path_str)
                if check_files:
                    if cand_path.is_file():
                        image_path = cand_path.resolve()
                    else:
                        raise FileNotFoundError(
                            f"Could not resolve image: absolute path '{row['path']}' does not exist on this system. "
                            "Provide explicit relocation mapping via 'remap_prefixes' (--remap_path_prefix) "
                            f"or 'source_roots' for dataset_source '{src}' instead of relying on ambiguous suffix guessing."
                        )
                else:
                    image_path = cand_path

            if check_files and not image_path.is_file():
                raise FileNotFoundError(f"Could not resolve image at '{image_path}' (source path: '{row['path']}', root: '{effective_root}')")
            row.update(path=str(image_path), label=int(row['label']))
            rows.append(row)
            if progress_every and (len(rows) % progress_every == 0 or time.monotonic() - last_progress >= 30):
                print(f'Manifest progress: {len(rows):,} rows loaded...', flush=True)
                last_progress = time.monotonic()
    if not rows:
        raise ValueError(f'No samples in {path} for split={split!r}')
    _MANIFEST_CACHE[cache_key] = [dict(r) for r in rows]
    return rows


def audit_rows(rows, require_groups=True, hash_files=False, hash_cache=None, progress_every=0):
    """IDs must be globally namespaced; linked originals/derivatives share groups."""
    errors, warnings = [], []
    seen = {key: {} for key in ('sample_id', 'path', 'sha256') + GROUP_FIELDS}
    counts, source_labels = Counter(), defaultdict(Counter)
    content_labels = {}
    last_progress = time.monotonic()
    if progress_every:
        print(f'Auditing {len(rows):,} rows; content hashing={hash_files}...', flush=True)
    for index, row in enumerate(rows, 1):
        split = row['split']
        counts[(split, int(row['label']))] += 1
        source_labels[(split, row.get('dataset_source', 'unknown'))][int(row['label'])] += 1
        if require_groups and not known(row.get('group_id')):
            errors.append(f"Missing group_id: {row['sample_id']}")
        if not known(row.get('dataset_source')):
            errors.append(f"Missing dataset_source: {row['sample_id']}")
        if hash_files:
            actual = hash_cache.get_or_compute(row['path']) if hash_cache else sha256(row['path'])
            if known(row.get('sha256')) and row['sha256'] != actual:
                errors.append(f"Content hash mismatch: {row['sample_id']}")
            row['sha256'] = actual
        for identity_key in ('path', 'sha256'):
            identity = row.get(identity_key)
            if known(identity):
                key = (identity_key, str(identity))
                label = int(row['label'])
                if key in content_labels and content_labels[key] != label:
                    errors.append(f'Conflicting labels for {identity_key}: {identity}')
                content_labels[key] = label
        for key, values in seen.items():
            val = row.get(key)
            tokens = [str(Path(val).resolve()).casefold()] if (key == 'path' and known(val)) else extract_entity_tokens(key, val)
            for token in tokens:
                previous = values.get(token)
                if previous and previous != split:
                    errors.append(f'{key} overlaps {previous} and {split}: {token}')
                values[token] = split
        if progress_every and (index % progress_every == 0 or time.monotonic() - last_progress >= 30 or index == len(rows)):
            print(f'Audit progress: {index:,}/{len(rows):,} rows; {len(errors):,} issues so far.', flush=True)
            last_progress = time.monotonic()
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


def generate_source_coverage_report(rows):
    """Generate metadata-only source coverage, annotation, and leakage vulnerability report.

    Does NOT decode or read images from disk. Analyzes provenance metadata:
    - Real/fake counts by dataset_source
    - Verified generator counts
    - Parent collection vs verified original sources
    - Annotation coverage rates
    - Independent group coverage (distinguishing independent:<hash> from real entities)
    - Fixed-partition conflicts across splits
    - One-class-only source warnings (shortcut risk)
    """
    total_samples = len(rows)
    if not total_samples:
        return {'status': 'empty_manifest', 'samples': 0}

    counts_by_source = defaultdict(Counter)
    counts_by_generator = defaultdict(Counter)
    counts_by_split_source = defaultdict(Counter)
    annotation_counts = Counter()

    unique_groups = set()
    path_hash_groups = set()
    entity_tokens_seen = defaultdict(dict)
    partition_conflicts = []

    known_metadata_fields = (
        'group_id', 'source_video_id', 'identity_id', 'original_id',
        'generator', 'manipulation_type', 'age_group', 'gender',
        'ethnicity', 'skin_tone', 'sha256'
    )

    for row in rows:
        label = int(row['label'])
        src = row.get('dataset_source', 'unknown')
        gen = row.get('generator', 'unknown')
        split = row.get('split', 'unassigned')

        counts_by_source[src][label] += 1
        counts_by_generator[gen][label] += 1
        counts_by_split_source[(split, src)][label] += 1

        for field in known_metadata_fields:
            if known(row.get(field)):
                annotation_counts[field] += 1

        gid = row.get('group_id')
        if known(gid):
            unique_groups.add(gid)
            if str(gid).startswith('independent:') or str(gid).startswith('path:'):
                path_hash_groups.add(gid)

        # Partition overlap detection across splits
        for key in ('sample_id', 'path') + GROUP_FIELDS + ('sha256',):
            val = row.get(key)
            if known(val):
                tokens = extract_entity_tokens(key, val) if key != 'path' else [str(val).casefold()]
                for tok in tokens:
                    prev_split = entity_tokens_seen[key].get(tok)
                    if prev_split and prev_split != split:
                        conflict = f"{key} '{tok}' appears in both '{prev_split}' and '{split}'"
                        if conflict not in partition_conflicts:
                            partition_conflicts.append(conflict)
                    entity_tokens_seen[key][tok] = split

    # Analyze single-class sources
    single_class_sources = []
    for src, l_counts in sorted(counts_by_source.items()):
        if len(l_counts) == 1:
            only_label = 'real' if 0 in l_counts else 'fake'
            single_class_sources.append({
                'source': src,
                'label': only_label,
                'count': l_counts[0 if only_label == 'real' else 1],
                'warning': f"Source '{src}' contains only {only_label} samples; high risk of detector learning collection-specific shortcuts."
            })

    single_class_by_split = []
    for (split, src), l_counts in sorted(counts_by_split_source.items()):
        if len(l_counts) == 1:
            only_label = 'real' if 0 in l_counts else 'fake'
            single_class_by_split.append({
                'split': split,
                'source': src,
                'label': only_label,
                'count': l_counts[0 if only_label == 'real' else 1],
            })

    # Distinguish collection names like diffgan
    collections_flagged = []
    known_collections = {'diffgan', 'faceforensics', 'faceforensics++', 'celeb_df', 'wilddeepfake', 'dfdc', 'forgerynet'}
    for src in counts_by_source:
        if str(src).lower() in known_collections:
            collections_flagged.append({
                'collection_name': src,
                'note': f"'{src}' is an aggregated collection name, not a verified single generator or camera source. Disentangle constituent generators for sound generalization claims."
            })

    # Source-held-out readiness assessment
    has_multiple_sources = len(counts_by_source) > 1
    has_verified_generators = annotation_counts['generator'] > 0 and len(counts_by_generator) > 1
    source_held_out_ready = has_multiple_sources and (len(single_class_sources) < len(counts_by_source))

    missing_metadata_report = []
    if not has_multiple_sources:
        missing_metadata_report.append("Only one dataset_source is declared. Cross-source generalization experiments require at least two distinct verified sources.")
    if annotation_counts['generator'] < total_samples * 0.5:
        missing_metadata_report.append(f"Generator annotation is missing for {total_samples - annotation_counts['generator']:,} of {total_samples:,} images ({((total_samples - annotation_counts['generator'])/total_samples)*100:.1f}%).")
    if len(path_hash_groups) > 0:
        missing_metadata_report.append(f"Found {len(path_hash_groups):,} synthetic placeholder groups ('independent:<hash>'). These are path-derived fallbacks, not verified independent individuals or video sessions.")

    report = {
        'total_samples': total_samples,
        'unique_sources': len(counts_by_source),
        'unique_groups': len(unique_groups),
        'synthetic_placeholder_groups': len(path_hash_groups),
        'provenance_note': 'Do not treat independent:<path hash> as evidence of independent people/videos. Distinguish collection names (e.g. diffgan) from verified original sources.',
        'source_coverage': [
            {'source': src, 'real': c[0], 'fake': c[1], 'total': c[0] + c[1], 'balance_ratio': round(c[1] / max(1, c[0]), 3)}
            for src, c in sorted(counts_by_source.items())
        ],
        'generator_coverage': [
            {'generator': gen, 'real': c[0], 'fake': c[1], 'total': c[0] + c[1]}
            for gen, c in sorted(counts_by_generator.items())
        ],
        'annotation_coverage': {
            field: {
                'annotated_count': annotation_counts[field],
                'coverage_pct': round((annotation_counts[field] / total_samples) * 100, 2)
            }
            for field in known_metadata_fields
        },
        'collections_identified': collections_flagged,
        'single_class_sources': single_class_sources,
        'single_class_by_split': single_class_by_split,
        'partition_conflicts': partition_conflicts,
        'partition_conflicts_count': len(partition_conflicts),
        'source_held_out_ready': source_held_out_ready,
        'missing_metadata_report': missing_metadata_report,
    }
    return report


def find_connected_components(rows, require_groups=False):
    """Union-Find across all identity/original/video/hash links.

    Every sample connected by any link is assigned to the exact same component.
    Parent rows remain immutable.
    """
    n = len(rows)
    parent = list(range(n))

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
                "fill group_id from original metadata before grouping."
            )
        for key in GROUP_FIELDS + ('sha256',):
            val = row.get(key)
            tokens = extract_entity_tokens(key, val)
            ns = 'video' if key in ('source_video_id', 'original_id') else key
            for token_val in tokens:
                token = (ns, token_val)
                if token in seen:
                    parent[find(i)] = find(seen[token])
                else:
                    seen[token] = i

    groups = defaultdict(list)
    for i in range(n):
        groups[find(i)].append(dict(rows[i]))

    # Deterministic component hash based on canonical entity tokens
    hashed_groups = {}
    for g_rows in groups.values():
        entity_tokens = set()
        for r in g_rows:
            for key in GROUP_FIELDS + ('sha256',):
                val = r.get(key)
                ns = 'video' if key in ('source_video_id', 'original_id') else key
                for t in extract_entity_tokens(key, val):
                    entity_tokens.add(f"{ns}:{t}")
        if entity_tokens:
            canonical_anchor = min(entity_tokens)
        else:
            canonical_anchor = f"sample:{min(str(r.get('sample_id', '')) for r in g_rows)}"
        group_hash = "connected:" + hashlib.sha256(canonical_anchor.encode()).hexdigest()
        hashed_groups[group_hash] = g_rows

    return hashed_groups


def select_representative_dev_cohort(dev_rows, target_size=15000, seed=42, target_real=None, target_fake=None, source_quotas=None):
    """Select a deterministic, connected-component and source-aware representative dev subset.

    Preserves parent cohort immutability.
    Enforces connected-component atomicity across source_video_id, identity_id, original_id, and hashes.
    Orders components deterministically before seeded permutation to ensure row-order invariance.
    Honors class and declared source quotas, reporting exact shortfalls for indivisible components.
    """
    if not dev_rows:
        raise ValueError("Cannot select representative dev cohort from empty rows")

    if target_real is None and target_fake is None:
        target_real = target_size // 2
        target_fake = target_size - target_real
    elif target_real is None or target_fake is None:
        raise ValueError("Specify both target_real and target_fake, or target_size alone")

    # Step 1: Discover connected components (parent cohort unmodified)
    components_map = find_connected_components(dev_rows, require_groups=False)

    if target_real < 0 or target_fake < 0 or target_real + target_fake <= 0:
        raise ValueError('Representative cohort requires nonnegative class targets and positive total')
    if source_quotas and any(q < 0 for q in source_quotas.values()):
        raise ValueError('Source quotas must be nonnegative')
    components = [components_map[k] for k in sorted(components_map)]
    rng = np.random.default_rng(seed)
    rng.shuffle(components)
    selected_rows = []
    selected_components = set()
    cur_real = cur_fake = selected_comp_count = 0
    cur_source_counts = Counter()

    def add_component(index):
        nonlocal cur_real, cur_fake, selected_comp_count
        if index in selected_components:
            return
        comp = components[index]
        labels = Counter(int(r['label']) for r in comp)
        sources = Counter(r.get('dataset_source', 'unknown') for r in comp)
        if cur_real + labels[0] > target_real or cur_fake + labels[1] > target_fake:
            return
        if source_quotas and any(cur_source_counts[src] + count > source_quotas[src]
                                for src, count in sources.items() if src in source_quotas):
            return
        selected_components.add(index)
        selected_rows.extend(dict(r) for r in comp)
        cur_real += labels[0]
        cur_fake += labels[1]
        selected_comp_count += 1
        cur_source_counts.update(sources)

    # Prioritize declared domains, then fill remaining capacity. Never split a
    # component or exceed class/source caps; report indivisibility shortfalls.
    if source_quotas:
        by_source = defaultdict(list)
        for i, comp in enumerate(components):
            for src in {r.get('dataset_source', 'unknown') for r in comp}:
                by_source[src].append(i)
        for src in sorted(source_quotas, key=lambda x: (len(by_source[x]), x)):
            for i in by_source[src]:
                if cur_source_counts[src] >= source_quotas[src]:
                    break
                add_component(i)
    for i in range(len(components)):
        add_component(i)

    # Deterministic sort of output rows by sample_id
    selected_rows.sort(key=lambda r: str(r['sample_id']))

    # Audit the resulting subset
    audit_res = audit_rows(selected_rows)
    if not audit_res['passed']:
        raise ValueError('Representative cohort audit failed: ' + '; '.join(audit_res['errors'][:10]))

    source_shortfalls = {}
    if source_quotas:
        for src, q in source_quotas.items():
            source_shortfalls[src] = max(0, q - cur_source_counts.get(src, 0))

    cohort_digest = hashlib.sha256(
        ','.join(str(r['sample_id']) for r in selected_rows).encode()
    ).hexdigest()

    parent_digest = hashlib.sha256(
        ','.join(sorted(str(r['sample_id']) for r in dev_rows)).encode()
    ).hexdigest()

    summary = {
        'total_selected': len(selected_rows),
        'real_count': cur_real,
        'fake_count': cur_fake,
        'target_real': target_real,
        'target_fake': target_fake,
        'shortfall_real': max(0, target_real - cur_real),
        'shortfall_fake': max(0, target_fake - cur_fake),
        'source_quotas': source_quotas,
        'source_counts': dict(cur_source_counts),
        'source_shortfalls': source_shortfalls,
        'connected_components_total': len(components_map),
        'connected_components_selected': selected_comp_count,
        'parent_cohort_size': len(dev_rows),
        'parent_cohort_sha256_digest': parent_digest,
        'cohort_sha256_digest': cohort_digest,
        'audit_result': audit_res,
        'note': 'Representative dev cohort preserves parent immutability and connected-component atomicity.'
    }
    return selected_rows, summary
