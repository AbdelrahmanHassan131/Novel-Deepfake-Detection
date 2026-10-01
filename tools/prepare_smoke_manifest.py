"""Bound a user-supplied smoke CSV before any image I/O (standard library only).

This does not sample or invent provenance. The existing audit remains mandatory.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path, PureWindowsPath

MAX_TOTAL = 300
SPLIT_LIMITS = {'train': 100, 'dev': 100, 'internal_test': 100}
REQUIRED = {'sample_id', 'path', 'label', 'split', 'dataset_source', 'group_id'}
UNKNOWN = {'', 'unknown', 'none', 'null', 'nan', 'n/a'}


def prepare(source, root, output):
    source, root, output = Path(source), Path(root).resolve(), Path(output)
    if source.resolve() == output.resolve() or output.exists():
        raise ValueError('Use a fresh output path; never overwrite the input manifest.')
    with source.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        if not REQUIRED.issubset(fields) or len(set(fields)) != len(fields):
            raise ValueError(f'Smoke CSV needs unique columns including {sorted(REQUIRED)}')
        rows = []
        for row in reader:
            rows.append(row)
            if len(rows) > MAX_TOTAL:
                raise ValueError('Smoke input exceeds 300 rows. Supply a separate bounded manifest, not the full pool.')

    counts, ids = Counter(), set()
    for row in rows:
        if None in row or any(row.get(k) is None for k in fields):
            raise ValueError('Malformed CSV row')
        for key in REQUIRED:
            value = row[key].strip()
            if value.lower() in UNKNOWN or value.lower().startswith(('n/a_', 'na_')):
                raise ValueError(f'Missing verified {key}; do not fabricate provenance for smoke data.')
        if row['sample_id'] in ids:
            raise ValueError('Duplicate sample_id')
        ids.add(row['sample_id'])
        if row['split'] not in SPLIT_LIMITS or row['label'] not in {'0', '1'}:
            raise ValueError('Use assigned train/dev/internal_test splits and labels real=0, fake=1.')
        counts[(row['split'], row['label'])] += 1
    for split, limit in SPLIT_LIMITS.items():
        if sum(counts[(split, label)] for label in ('0', '1')) > limit:
            raise ValueError(f'{split} exceeds {limit} smoke rows')
    for split in ('train', 'dev'):
        if any(counts[(split, label)] == 0 for label in ('0', '1')):
            raise ValueError(f'{split} must contain both classes from independent groups')
    if any(counts[('train', label)] > 50 for label in ('0', '1')):
        raise ValueError('Smoke training permits at most 50 real and 50 fake rows')

    # Resolve only after ALL count/schema checks. Never search suffixes or scan images.
    for row in rows:
        raw = row['path'].replace('\\', '/')
        if PureWindowsPath(raw).is_absolute() and not Path(raw).is_absolute():
            raise ValueError('Relocate Windows image paths explicitly before using this CSV on Colab.')
        path = Path(raw)
        row['path'] = str((path if path.is_absolute() else root / path).resolve())
    report = {'samples': len(rows), 'counts': {s: {l: counts[(s, l)] for l in ('0', '1')} for s in SPLIT_LIMITS},
              'audit_status': 'NOT RUN; hash and relationship audit required before training'}
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    output.with_suffix('.bounds.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.manifest, args.root, args.output), indent=2))


if __name__ == '__main__':
    main()
