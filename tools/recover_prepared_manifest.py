"""Recover a completed hash audit without reopening an explicitly unchanged dataset.

Preserves evaluation rows; removes entire training components linked to evaluation.
Only reads/writes CSV and JSON files. Never deletes images or invents extra samples.
"""
import argparse
import csv
import hashlib
import json
import ntpath
import os
import re
from collections import Counter, defaultdict
from pathlib import Path

VERSION = 1
GROUP_FIELDS = ('group_id', 'source_video_id', 'identity_id', 'original_id')
UNKNOWN = {'', 'unknown', 'none', 'nan', 'n/a', 'null', 'n/a_photograph', 'n/a_synthesis', 'n/a_still'}
EVALUATION = {'dev', 'val', 'external_dev', 'test', 'internal_test', 'external_test', 'final_test'}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def known(value):
    text = str(value or '').strip().lower()
    return text not in UNKNOWN and not text.startswith(('n/a_', 'na_'))


def tokens(row):
    for key in GROUP_FIELDS:
        for value in re.split(r'[,;|]', row.get(key, '')):
            if known(value):
                yield key, value.strip()
    yield 'sha256', row['sha256']
    # Input is an already-audited absolute-path manifest; no image path stat calls.
    yield 'path', row['path'].replace('\\', '/').casefold()


def write_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2), encoding='utf-8')
    temp.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def recover(manifest, audit_report, output_dir, immutable_dataset=False, allow_shortfall=False):
    if not immutable_dataset:
        raise ValueError('Hash reuse requires --immutable_dataset: images and paths must be unchanged since the completed audit.')
    manifest, audit_report, output = Path(manifest), Path(audit_report), Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    target, gate_path = output / 'selected_manifest.csv', output / 'selected_manifest.verified.json'
    report_path = output / 'recovery_report.json'
    destinations = {p.resolve() for p in (target, gate_path, report_path, output / 'audit_report.json', output / 'excluded_training.csv')}
    if manifest.resolve() in destinations or audit_report.resolve() in destinations:
        raise ValueError('Use a separate output directory; preserve the original audited manifest.')
    print('Checking saved manifest and completed audit evidence (no images opened)...', flush=True)
    manifest_hash, audit_hash = digest(manifest), digest(audit_report)
    audit = json.loads(audit_report.read_text(encoding='utf-8-sig'))
    if audit.get('hashes_verified') is not True or audit.get('manifest_sha256') != manifest_hash:
        raise ValueError('Audit must record hashes_verified=true and match the exact input CSV digest. Do not regenerate the input inventory.')
    # A failed overlap audit still contains valid completed content-hash evidence.
    for error in audit.get('errors', []):
        if not re.match(r'^(sha256|path|sample_id|group_id|source_video_id|identity_id|original_id) overlaps ', error):
            raise ValueError(f'Unsupported original audit failure requires review: {error}')
    identity = dict(version=VERSION, source_manifest_sha256=manifest_hash, source_audit_sha256=audit_hash,
                    immutable_dataset=True, allow_shortfall=allow_shortfall)
    if gate_path.exists():
        gate = json.loads(gate_path.read_text())
        if (gate.get('recovery_identity') == identity and gate.get('verified') is True
                and target.is_file() and gate.get('manifest_sha256') == digest(target)
                and report_path.is_file() and gate.get('recovery_report_sha256') == digest(report_path)):
            print(f'Already prepared and verified. Reuse: {target}', flush=True)
            return json.loads(report_path.read_text())
        raise ValueError('Existing output has different inputs/policy or changed files. Use a new output directory.')

    with manifest.open(newline='', encoding='utf-8-sig') as stream:
        reader = csv.DictReader(stream)
        fields = reader.fieldnames or []
        required = {'sample_id', 'path', 'label', 'split', 'dataset_source', 'group_id', 'sha256'}
        if not required.issubset(fields) or len(fields) != len(set(fields)):
            raise ValueError('Input manifest has missing or duplicate columns')
        rows, ids = [], set()
        for row in reader:
            if None in row or any(v is None for v in row.values()):
                raise ValueError('Malformed input CSV')
            if row['sample_id'] in ids or any(not known(row[k]) for k in required):
                raise ValueError('Missing metadata or duplicate sample_id')
            if row['label'] not in {'0', '1'} or row['split'] not in EVALUATION | {'train'}:
                raise ValueError('Expected binary labels and explicit train/evaluation assignments')
            if not re.fullmatch('[0-9a-f]{64}', row['sha256']):
                raise ValueError('Every row must contain a completed SHA256 digest')
            if not (os.path.isabs(row['path']) or ntpath.isabs(row['path'])):
                raise ValueError('Expected absolute image paths from the completed audit; do not rewrite paths before recovery')
            ids.add(row['sample_id'])
            rows.append(row)
            if len(rows) % 50000 == 0:
                print(f'Read {len(rows):,} metadata rows...', flush=True)
    if audit.get('samples') != len(rows):
        raise ValueError('Audit sample count does not match the input CSV')
    if digest(manifest) != manifest_hash or digest(audit_report) != audit_hash:
        raise ValueError('Input evidence changed during recovery; no gate issued')

    parent = list(range(len(rows)))
    def find(index):
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index
    seen = {}
    for i, row in enumerate(rows):
        for token in tokens(row):
            if token in seen:
                parent[find(i)] = find(seen[token])
            else:
                seen[token] = i
        if (i + 1) % 50000 == 0:
            print(f'Checked relationships for {i + 1:,}/{len(rows):,} rows...', flush=True)
    del seen
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    removed, removal_reason = set(), {}
    for indices in groups.values():
        splits = {rows[i]['split'] for i in indices}
        eval_splits = splits - {'train'}
        if len(eval_splits) > 1:
            examples = [rows[i]['path'] for i in indices[:5]]
            raise ValueError(f'Linked evaluation partitions {sorted(eval_splits)} require review; not changing them: {examples}')
        conflicting_hashes, eval_labels = defaultdict(set), defaultdict(set)
        for i in indices:
            conflicting_hashes[rows[i]['sha256']].add(rows[i]['label'])
            if rows[i]['split'] != 'train':
                eval_labels[rows[i]['sha256']].add(rows[i]['label'])
        label_conflict = any(len(labels) > 1 for labels in conflicting_hashes.values())
        if any(len(labels) > 1 for labels in eval_labels.values()):
            raise ValueError('Identical evaluation images have conflicting labels. Correct the evaluation metadata explicitly.')
        if eval_splits or label_conflict:
            for i in indices:
                if rows[i]['split'] == 'train':
                    removed.add(i)
                    removal_reason[i] = 'linked_to_evaluation' if eval_splits else 'conflicting_content_labels'
        # Record dependency grouping for later group-aware metrics, including same-split duplicates.
        anchor = min(rows[i]['sample_id'] for i in indices)
        group_id = 'recovered:' + hashlib.sha256(anchor.encode()).hexdigest()
        for i in indices:
            rows[i]['group_id'] = group_id
    kept = [r for i, r in enumerate(rows) if i not in removed]
    before = Counter((r['split'], r['label']) for r in rows)
    after = Counter((r['split'], r['label']) for r in kept)
    counts = lambda counter: [{'split': s, 'label': int(l), 'count': n} for (s, l), n in sorted(counter.items())]
    report = dict(recovery_identity=identity, input_samples=len(rows), output_samples=len(kept),
                  removed_training_samples=len(removed), before=counts(before), after=counts(after),
                  evaluation_rows_preserved=True, image_files_changed=False,
                  hash_verification_mode='inherited_completed_audit_on_explicitly_unchanged_dataset',
                  original_training_target=before[('train', '0')] + before[('train', '1')],
                  actual_training_samples=after[('train', '0')] + after[('train', '1')])
    write_json(report_path, report)
    write_csv(output / 'excluded_training.csv',
              [dict(rows[i], exclusion_reason=removal_reason[i]) for i in sorted(removed)],
              list(dict.fromkeys(fields + ['exclusion_reason'])))
    if any(after[('train', label)] == 0 or after[('dev', label)] == 0 for label in ('0', '1')):
        raise ValueError('Recovery must retain both classes in train and dev. See recovery_report.json.')
    if removed and not allow_shortfall:
        raise ValueError(f'Removing {len(removed):,} training rows leaves {report["actual_training_samples"]:,}. '
                         'Review recovery_report.json and rerun with --allow_shortfall to explicitly accept this count. No gate issued.')
    # Independent post-filter audit over ALL declared relationships and paths.
    checked = {}
    for row in kept:
        for token in tokens(row):
            previous = checked.setdefault(token, row['split'])
            if previous != row['split']:
                raise ValueError('Recovery still has cross-split overlap; no gate issued')
    write_csv(target, kept, fields)
    target_hash = digest(target)
    recovered_audit = dict(passed=True, errors=[], hashes_verified=True, samples=len(kept),
                           counts=counts(after), manifest_sha256=target_hash,
                           hash_verification_mode=report['hash_verification_mode'], recovery_identity=identity)
    write_json(output / 'audit_report.json', recovered_audit)
    write_json(gate_path, dict(verified=True, hashes_verified=True, manifest_sha256=target_hash,
                              manifest_path=str(target.resolve()), samples=len(kept), classes=[0, 1],
                              splits=sorted({r['split'] for r in kept}), unique_groups=len({r['group_id'] for r in kept}),
                              audit_report=str((output / 'audit_report.json').resolve()),
                              hash_verification_mode=report['hash_verification_mode'], recovery_identity=identity,
                              recovery_report_sha256=digest(report_path)))
    print(json.dumps(report, indent=2), flush=True)
    print(f'Prepared and verified: {target}\nReuse this manifest for every model; do not rerun inventory.', flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--audit_report', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--immutable_dataset', action='store_true')
    parser.add_argument('--allow_shortfall', action='store_true')
    args = parser.parse_args()
    recover(args.manifest, args.audit_report, args.output_dir, args.immutable_dataset, args.allow_shortfall)


if __name__ == '__main__':
    main()
