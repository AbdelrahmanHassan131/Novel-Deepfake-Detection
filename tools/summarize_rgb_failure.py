"""Analyze existing CSV/JSON predictions only. No images, torch or checkpoints.

Hindsight threshold accuracy is a diagnostic upper bound on these labeled rows,
NOT a selected threshold, deployable accuracy estimate, or new test result.
"""
import argparse
import csv
import io
import itertools
import json
import math
import statistics
import zipfile
from collections import Counter, defaultdict
from pathlib import Path
from repair_rgb_family_split import family_id


def ranking(rows, key):
    pairs = sorted((float(r[key]), int(r['label'])) for r in rows)
    if not pairs or any(not math.isfinite(s) or y not in (0, 1) for s, y in pairs):
        raise ValueError('Expected finite scores and binary labels')
    n0 = sum(y == 0 for _, y in pairs)
    n1 = len(pairs) - n0
    if not n0 or not n1:
        raise ValueError('Both classes required')
    negatives = wins = 0
    tp, fp = n1, n0
    best = .5
    for _, group in itertools.groupby(pairs, key=lambda p: p[0]):
        labels = [y for _, y in group]
        a = labels.count(0)
        b = len(labels) - a
        wins += b * (negatives + a / 2)
        negatives += a
        tp -= b
        fp -= a
        best = max(best, .5 * (tp / n1 + (n0 - fp) / n0))
    return dict(roc_auc=wins / (n0 * n1), hindsight_max_balanced_accuracy=best,
                unique_scores=len(set(s for s, _ in pairs)))


def run(predictions, archive, output):
    with Path(predictions).open(newline='', encoding='utf-8-sig') as stream:
        external = list(csv.DictReader(stream))
    with zipfile.ZipFile(archive) as z:
        names = z.namelist()
        dev_name = [n for n in names if n.startswith('rgb_pilots/') and n.endswith('/checkpoints/best_dev_predictions.csv')]
        pool_name = [n for n in names if n.startswith('prepared_data/') and n.endswith('/sizes/100000/selected_manifest.csv')]
        if len(dev_name) != 1 or len(pool_name) != 1:
            raise ValueError('Expected one RGB best-dev CSV and one prepared 100K manifest in the supplied archive')
        with z.open(dev_name[0]) as stream:
            dev = list(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')))
        with z.open(pool_name[0]) as stream:
            pool = list(csv.DictReader(io.TextIOWrapper(stream, encoding='utf-8-sig')))
    hashes = {r['checkpoint_sha256'] for r in external + dev}
    if len(hashes) != 1 or not next(iter(hashes)):
        raise ValueError('Development and external predictions must refer to the same checkpoint')
    families = defaultdict(set)
    formats = Counter()
    for row in pool:
        family = family_id(row['path'])
        if family:
            families[row['split']].add(family)
        formats[(row['split'], int(row['label']), row['path'].rsplit('.', 1)[-1].lower())] += 1
    overlap = families['train'] & families['dev']
    affected = Counter((r['split'], int(r['label'])) for r in pool if family_id(r['path']) in overlap)
    dev_groups = defaultdict(list)
    for row in dev:
        dev_groups[(int(row['label']), row['path'].rsplit('.', 1)[-1].lower())].append(row)
    dev_summary = []
    for (label, extension), rows in sorted(dev_groups.items()):
        dev_summary.append(dict(label=label, extension=extension, samples=len(rows),
            recall_at_05=sum((float(r['probability']) >= .5) == bool(label) for r in rows) / len(rows),
            median_probability=statistics.median(float(r['probability']) for r in rows)))
    report = dict(checkpoint_sha256=next(iter(hashes)), samples=len(external),
        score_diagnostics={k: ranking(external, k) for k in ('probability', 'logit')},
        hindsight_note='Not an evaluation score or threshold recommendation. No threshold exported; labels used only to bound what score thresholding could achieve.',
        filename_family_overlap=len(overlap),
        affected_rows=[dict(split=s, label=c, count=n) for (s, c), n in sorted(affected.items())],
        extension_counts=[dict(split=s, label=c, extension=e, count=n) for (s, c, e), n in sorted(formats.items())],
        development_by_extension=dev_summary,
        limitations=['Extensions are not verified file encodings.',
                    'Filename families indicate a conservative partition dependency, not independently verified source identities.',
                    'No image/profile/model diagnostics were run; causes of external failure remain partly unverified.'])
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--predictions', required=True)
    p.add_argument('--archive', required=True)
    p.add_argument('--output', required=True)
    a = p.parse_args()
    run(a.predictions, a.archive, a.output)
