"""Rebuild an engineering pilot from an audited parent without opening images.

Conservatively connects vid_<64hex>_face_<frame>_<index> filenames across classes,
parent dependency groups and content hashes BEFORE repartitioning. Filename
families are not verified original-video/person/source identities. This repairs
a measurable partition defect; it does not certify cross-source generalization.
"""
import argparse
import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

FAMILY = re.compile(r'^(vid_[0-9a-f]{64})_face_\d+_\d+\.[a-z0-9]+$', re.I)
UNKNOWN = {'', 'unknown', 'none', 'nan', 'n/a', 'null'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def family_id(path):
    match = FAMILY.fullmatch(str(path).replace('\\', '/').rsplit('/', 1)[-1])
    return match[1].lower() if match else None


def known(value):
    value = str(value or '').strip().lower()
    return value not in UNKNOWN and not value.startswith(('n/a_', 'na_'))


def tokens(row):
    for field in ('group_id', 'source_video_id', 'original_id', 'identity_id', 'sha256'):
        for value in re.split(r'[,;|]', str(row.get(field, ''))):
            if known(value):
                yield ('video' if field in {'source_video_id', 'original_id'} else field, value.strip())
    yield 'path', row['path'].replace('\\', '/').casefold()
    family = family_id(row['path'])
    if family:
        yield 'filename_family', family


def save_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temporary.replace(path)


def build_rows(rows, train_size=100000, dev_size=15000, seed=42, cap=8, dev_fraction=.15):
    if any(size <= 0 or size % 2 for size in (train_size, dev_size)):
        raise ValueError('Train/dev targets must be positive even totals')
    if cap < 1 or not 0 < dev_fraction < 1:
        raise ValueError('cap must be positive; dev_fraction must be between 0 and 1')
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    seen, ids, content_labels = {}, set(), {}
    old_families = defaultdict(set)
    for i, row in enumerate(rows):
        if row.get('split') not in {'train', 'dev'}:
            raise ValueError('Only an existing engineering train/dev pool can be repartitioned; never final/test data')
        if row.get('grouping_basis') != 'image_level_unverified':
            raise ValueError('This repair targets the explicitly unverified per-image Colab pool')
        if row.get('label') not in ('0', '1', 0, 1):
            raise ValueError('Expected canonical real=0, fake=1')
        sid, sha = row.get('sample_id'), row.get('sha256', '')
        if not sid or sid in ids or not re.fullmatch('[0-9a-f]{64}', sha):
            raise ValueError('Duplicate/missing sample ID or missing SHA256')
        ids.add(sid)
        label = int(row['label'])
        if content_labels.setdefault(sha, label) != label:
            raise ValueError('Parent still has contradictory labels; quarantine first')
        family = family_id(row['path'])
        if family:
            old_families[family].add(row['split'])
        for token in tokens(row):
            if token in seen:
                parents[find(i)] = find(seen[token])
            else:
                seen[token] = i
        if (i + 1) % 50000 == 0:
            print(f'Connected {i + 1:,}/{len(rows):,} metadata rows; no images opened.', flush=True)
    del seen, ids, content_labels
    components = defaultdict(list)
    for i in range(len(rows)):
        components[find(i)].append(i)
    pools = {'train': [], 'dev': []}
    for indices in components.values():
        anchor = min(rows[i]['sample_id'] for i in indices)
        group = 'family_connected:' + hashlib.sha256(anchor.encode()).hexdigest()
        fraction = int(hashlib.sha256(f'partition:{seed}:{group}'.encode()).hexdigest()[:12], 16) / 16**12
        split = 'dev' if fraction < dev_fraction else 'train'
        # Drop duplicate content within the family, then cap each class in the
        # connected group. Selection never breaks the component's partition.
        ranked = sorted(indices, key=lambda i: hashlib.sha256(f'sample:{seed}:{rows[i]["sample_id"]}'.encode()).digest())
        selected, hashes, counts = [], set(), Counter()
        for i in ranked:
            row = rows[i]
            label = int(row['label'])
            if row['sha256'] in hashes or counts[label] >= cap:
                continue
            selected.append(i)
            hashes.add(row['sha256'])
            counts[label] += 1
        order = hashlib.sha256(f'quota:{seed}:{group}'.encode()).hexdigest()
        pools[split].append((order, group, selected, counts))
    output, summaries = [], {}
    for split, target in (('train', train_size), ('dev', dev_size)):
        counts = Counter()
        for _, group, indices, group_counts in sorted(pools[split]):
            if any(counts[label] + group_counts[label] > target // 2 for label in (0, 1)):
                continue
            for i in indices:
                row = dict(rows[i], parent_split=rows[i]['split'], group_id=group, split=split,
                           grouping_method='content_parent_links_and_filename_family',
                           filename_family=family_id(rows[i]['path']) or 'unknown')
                # Preserve image_level_unverified: no invented source/video IDs.
                output.append(row)
            counts.update(group_counts)
        if any(counts[label] == 0 for label in (0, 1)):
            raise ValueError(f'{split} cannot retain both classes at this grouping/cap; inspect available data')
        summaries[split] = dict(target=target, real=counts[0], fake=counts[1],
                                actual=counts[0] + counts[1], shortfall=target - counts[0] - counts[1])
    # Independent post-selection validation, including every parent relationship
    # and the filename grouping that the earlier image-only audit did not know.
    assigned = {}
    for row in output:
        for token in tokens(row):
            if assigned.setdefault(token, row['split']) != row['split']:
                raise ValueError('Repaired output still has cross-partition overlap')
    extensions = Counter((r['split'], int(r['label']), r['path'].rsplit('.', 1)[-1].lower()) for r in output)
    summary = dict(splits=summaries, parent_samples=len(rows), output_samples=len(output),
                   parent_filename_families_crossing_splits=sum(len(s) > 1 for s in old_families.values()),
                   repaired_filename_families_crossing_splits=0,
                   component_count=len(components), maximum_rows_per_component_per_class=cap,
                   extension_counts=[dict(split=s, label=c, extension=e, count=n)
                                     for (s, c, e), n in sorted(extensions.items())],
                   limitations=['Filename families are conservative dependencies, not verified source/video/identity metadata.',
                                'Unknown relationships and near-duplicates can remain.',
                                'Format/semantic/source confounding is NOT solved by this split repair.',
                                'New development membership; do not compare its accuracy directly with the old dev cohort.',
                                'Previously inspected external data remains external development, not a final test.'])
    return sorted(output, key=lambda r: r['sample_id']), summary


def repair(parent, output_dir, train_size, dev_size, seed, cap, dev_fraction, immutable_dataset, allow_shortfall):
    if not immutable_dataset:
        raise ValueError('Hash inheritance requires --immutable_dataset: exactly the same image bytes and paths')
    parent = Path(parent)
    gate_path = parent.with_suffix('.verified.json')
    gate = json.loads(gate_path.read_text(encoding='utf-8-sig'))
    parent_hash = digest(parent)
    if gate.get('verified') is not True or gate.get('hashes_verified') is not True or gate.get('manifest_sha256') != parent_hash:
        raise ValueError('Parent needs a valid completed hash audit gate')
    recovery = parent.parent / 'recovery_report.json'
    if not recovery.is_file() or gate.get('recovery_report_sha256') != digest(recovery):
        raise ValueError('Use clean_parent/selected_manifest.csv with its bound recovery report')
    output = Path(output_dir)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty NEW output directory; old evidence is never overwritten')
    gate_hash = digest(gate_path)
    with parent.open(newline='', encoding='utf-8-sig') as stream:
        rows = list(csv.DictReader(stream))
    selected, report = build_rows(rows, train_size, dev_size, seed, cap, dev_fraction)
    if digest(parent) != parent_hash or digest(gate_path) != gate_hash:
        raise ValueError('Parent evidence changed during repair')
    output.mkdir(parents=True, exist_ok=True)
    identity = dict(parent_manifest_sha256=parent_hash, parent_gate_sha256=gate_hash,
                    seed=seed, cap=cap, dev_fraction=dev_fraction, train_size=train_size, dev_size=dev_size)
    report['preparation_identity'] = identity
    save_json(output / 'repair_report.json', report)
    if any(s['shortfall'] for s in report['splits'].values()) and not allow_shortfall:
        raise ValueError('Group-respecting selection has a shortfall; see repair_report.json. Choose a new output directory with --allow_shortfall to accept.')
    target = output / 'selected_manifest.csv'
    fields = list(dict.fromkeys(k for row in selected for k in row))
    temporary = target.with_suffix('.csv.tmp')
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(selected)
    temporary.replace(target)
    save_json(target.with_suffix('.verified.json'), dict(
        verified=True, hashes_verified=True, manifest_sha256=digest(target), classes=[0, 1],
        samples=len(selected), splits=['train', 'dev'], preparation_identity=identity,
        hash_verification_mode='inherited_unchanged_content_from_recovered_parent_new_family_split',
        near_duplicates_verified=False, source_video_identity_verified=False,
        repair_report_sha256=digest(output / 'repair_report.json')))
    print(json.dumps(report, indent=2), flush=True)
    print(f'Repaired engineering manifest: {target}. No images were opened or hashed.', flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--parent', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--train_size', type=int, default=100000)
    p.add_argument('--dev_size', type=int, default=15000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cap', type=int, default=8)
    p.add_argument('--dev_fraction', type=float, default=.15)
    p.add_argument('--immutable_dataset', action='store_true')
    p.add_argument('--allow_shortfall', action='store_true')
    args = p.parse_args()
    repair(args.parent, args.output_dir, args.train_size, args.dev_size, args.seed,
           args.cap, args.dev_fraction, args.immutable_dataset, args.allow_shortfall)


if __name__ == '__main__':
    main()
