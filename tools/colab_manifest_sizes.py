"""CPU-only partition/subset helpers; never open image files or load models.

Content-hash evidence for subsets is inherited only from a digest-bound, recovered
parent. Input images must remain unchanged. Image-level groups are not evidence
of video/identity independence.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data.manifest import audit_rows, verify_manifest_gate, write_manifest


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_rows(path):
    with Path(path).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))


def save_json(path, value):
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def partition(inventory, output, dev_size, seed):
    """Reserve dev from a flat train-only inventory BEFORE the full hash audit."""
    if dev_size <= 0 or dev_size % 2:
        raise ValueError('dev_size must be a positive even count')
    output = Path(output)
    if output.exists():
        raise ValueError('Output exists; reuse it or choose a new preparation directory')
    rows = read_rows(inventory)
    if not rows or any(r['split'] not in {'pool', 'unassigned'} for r in rows):
        raise ValueError('Inventory must come from the train folder itself, with no existing split assignments')
    if any(r.get('grouping_basis') != 'image_level_unverified' for r in rows):
        raise ValueError('This helper is specifically for explicit unverified image-level inventory')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate sample IDs')
    chosen = set()
    for label in ('0', '1'):
        candidates = [r for r in rows if r['label'] == label]
        candidates.sort(key=lambda r: hashlib.sha256(f"dev:{seed}:{r['sample_id']}".encode()).digest())
        if len(candidates) <= dev_size // 2:
            raise ValueError(f'Not enough class {label} images for dev plus training')
        chosen.update(r['sample_id'] for r in candidates[:dev_size // 2])
    for row in rows:
        row['split'] = 'dev' if row['sample_id'] in chosen else 'train'
    temporary = output.with_suffix(output.suffix + '.tmp')
    write_manifest(temporary, rows)
    temporary.replace(output)
    save_json(output.with_suffix('.partition.json'), {
        'inventory_sha256': digest(inventory), 'seed': seed,
        'requested_dev_size': dev_size, 'grouping_basis': 'image_level_unverified',
        'note': 'Provisional splits: must hash the full pool and recover before use.'})
    print(f'Provisional: {len(rows) - len(chosen):,} train + {len(chosen):,} dev. Full audit required.', flush=True)


def subsets(parent, output_dir, sizes, seed, immutable_dataset):
    if not immutable_dataset:
        raise ValueError('Pass --immutable_dataset only for unchanged image bytes and paths')
    parent = Path(parent)
    verify_manifest_gate(parent, require_hashes=True)
    parent_hash = digest(parent)
    gate_path = parent.with_suffix('.verified.json')
    parent_gate_hash = digest(gate_path)
    parent_gate = json.loads(gate_path.read_text())
    recovery_path = parent.parent / 'recovery_report.json'
    if not recovery_path.is_file() or parent_gate.get('recovery_report_sha256') != digest(recovery_path):
        raise ValueError('Expected a recovered parent with matching recovery report')
    rows = read_rows(parent)
    if any(r['split'] not in {'train', 'dev'} for r in rows):
        raise ValueError('This workflow supports one train and one shared dev partition')
    # Check the parent metadata again; image content is not reopened.
    report = audit_rows(rows, hash_files=False, progress_every=50000)
    if not report['passed']:
        raise ValueError(f'Parent metadata audit failed: {report["errors"][:5]}')
    if any(len(r.get('sha256', '')) != 64 for r in rows):
        raise ValueError('Missing parent content hashes')
    dev = [r for r in rows if r['split'] == 'dev']
    groups = defaultdict(list)
    for row in rows:
        if row['split'] == 'train':
            groups[row['group_id']].append(row)
    by_class = {0: [], 1: []}
    for group, members in groups.items():
        labels = {int(r['label']) for r in members}
        if len(labels) != 1:
            raise ValueError('Mixed-label group: this image-level helper cannot select it safely')
        by_class[next(iter(labels))].append(group)
    for label in by_class:
        by_class[label].sort(key=lambda g: hashlib.sha256(f'train:{seed}:{g}'.encode()).digest())
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summaries = {}
    for size in sizes:
        if size != 'all' and (not size.isdecimal() or int(size) <= 0 or int(size) % 2):
            raise ValueError('Sizes must be positive even counts or all')
        selected = []
        for label in (0, 1):
            count = 0
            for group in by_class[label]:
                members = groups[group]
                # Prefix selection preserves nesting; never split a duplicate group.
                if size != 'all' and count + len(members) > int(size) // 2:
                    break
                selected.extend(members)
                count += len(members)
        counts = Counter(int(r['label']) for r in selected)
        if not all(counts[k] for k in (0, 1)) or {int(r['label']) for r in dev} != {0, 1}:
            raise ValueError('Both classes are required in training and dev')
        target = output / size
        target.mkdir(exist_ok=True)
        manifest = target / 'selected_manifest.csv'
        identity = dict(parent_manifest_sha256=parent_hash, parent_gate_sha256=parent_gate_hash,
                        seed=seed, requested_train_size=size, version=1)
        if manifest.exists() and manifest.with_suffix('.verified.json').exists():
            verify_manifest_gate(manifest, require_hashes=True)
            existing = json.loads(manifest.with_suffix('.verified.json').read_text())
            if existing.get('subset_identity') != identity:
                raise ValueError('Existing subset uses different settings; use a new output directory')
        else:
            # Every output row is an unchanged reference to a validated parent row.
            # Removing rows preserves the parent's no-cross-split-overlap guarantee.
            if digest(parent) != parent_hash or digest(gate_path) != parent_gate_hash:
                raise ValueError('Parent evidence changed during selection')
            temporary = manifest.with_suffix('.csv.tmp')
            write_manifest(temporary, selected + dev)
            temporary.replace(manifest)
            save_json(manifest.with_suffix('.verified.json'), dict(
                verified=True, hashes_verified=True, manifest_sha256=digest(manifest),
                classes=[0, 1], splits=['train', 'dev'], samples=len(selected) + len(dev),
                subset_identity=identity,
                hash_verification_mode='unchanged_subset_of_completed_recovered_parent_audit',
                near_duplicates_verified=False))
        summary = dict(identity, actual_training_samples=len(selected), training_real=counts[0],
                       training_fake=counts[1], dev_samples=len(dev),
                       shortfall=0 if size == 'all' else int(size) - len(selected),
                       grouping_basis='image_level_unverified',
                       note='Same dev in every size; whole connected groups retained; counts may be below target.')
        save_json(target / 'selection_report.json', summary)
        summaries[size] = summary
        print(f'{size}: {len(selected):,} train ({counts[0]:,} real / {counts[1]:,} fake), {len(dev):,} shared dev', flush=True)
    save_json(output / 'sizes_summary.json', summaries)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='action', required=True)
    p = commands.add_parser('partition')
    p.add_argument('--inventory', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dev_size', type=int, default=15000)
    p.add_argument('--seed', type=int, default=42)
    p = commands.add_parser('subsets')
    p.add_argument('--parent', required=True)
    p.add_argument('--output_dir', required=True)
    p.add_argument('--sizes', nargs='+', default=['100000', '500000', '1000000', 'all'])
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--immutable_dataset', action='store_true')
    args = parser.parse_args()
    if args.action == 'partition':
        partition(args.inventory, args.output, args.dev_size, args.seed)
    else:
        subsets(args.parent, args.output_dir, args.sizes, args.seed, args.immutable_dataset)


if __name__ == '__main__':
    main()
