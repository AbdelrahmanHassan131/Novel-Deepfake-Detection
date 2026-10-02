# Recover your completed audit once, then reuse the prepared dataset

**Update: contradictory real/fake labels in validation.** The tool now writes `label_conflicts.csv` and `conflict_summary.json` before stopping. The command below explicitly enables `--quarantine_label_conflicts`: entire connected groups containing byte-identical images with contradictory labels are excluded from a new derived selection. Original files and labels are unchanged. This can remove validation rows as well as training rows; it does not preserve the original evaluation cohort. Keep the exclusion report and use the same resulting cohort for every model comparison.

Your uploaded output shows successful selection of **100,000 training images** and **211,434 development images**. The completed audit found **910 distinct file hashes shared across train and dev** (1,519 overlap messages). These are byte-identical files, not the earlier filename-grouping bug. Repeating inventory cannot fix this.

The failed audit already saved the image hashes into `selected_manifest.csv`. Its `audit_report.json` records `hashes_verified: true` and the digest of that exact CSV. Preserve both files. Recovery uses that completed evidence and checks the CSV digest; it does not rescan the million-image folder or reopen image files.

## 1. Upload the recovery script to Kaggle

Copy the new `tools/recover_prepared_manifest.py` into the same location in your Kaggle repository. It uses only Python's standard library. You do not need model checkpoints, GPU work, package installation, or another inventory for this step.

For future normal audits, also update `prepare_dataset.py`, `data/manifest.py`, and `data/hash_cache.py`. They add audit progress messages and a persistent SQLite hash cache. The recovery script below works independently of those changes.

## 2. Run this replacement cell once

The input folder below is the exact folder from your uploaded log. If you moved it, update `OLD_RUN`. Do not run your old inventory/pilot/audit cell again first: that would replace evidence and repeat work.

`--immutable_dataset` confirms that the images and their paths are unchanged since that completed audit (for example, the same attached read-only Kaggle dataset version). If files changed, cached evidence cannot establish their current contents: use a fresh content audit instead.

`--allow_shortfall` explicitly accepts **fewer than 100K training images after duplicate removal**. Recovery normally keeps every development image. With the explicit quarantine option below, it also excludes whole ambiguous groups from development; all other train/dev overlap is removed from training. It never silently refills with unaudited images or changes labels. Read the printed real/fake counts; they may no longer be exactly balanced. No image files are deleted or moved.

```bash
%%bash
set -euo pipefail
cd /kaggle/working/Novel-Deepfake-Detection

OLD_RUN="/kaggle/working/deepfake_experiments/train_100000_seed42_27cf24839a"
PREPARED_DIR="/kaggle/working/prepared_data/recovered_100k_seed42_v2"

python -u tools/recover_prepared_manifest.py \
  --manifest "$OLD_RUN/selected_manifest.csv" \
  --audit_report "$OLD_RUN/audit_report.json" \
  --output_dir "$PREPARED_DIR" \
  --immutable_dataset \
  --allow_shortfall \
  --quarantine_label_conflicts
```

It checks saved evidence, prints metadata progress, resolves connected groups using existing IDs and hashes, rechecks overlap after filtering, and creates a new audit/gate. Repeating the same recovery command with the same input evidence and policy reuses the completed result. It refuses changed inputs/outputs instead of overwriting a verified result. Conflicts between two evaluation partitions still require review. Without `--quarantine_label_conflicts`, contradictory evaluation labels also block recovery, with diagnostic files written first. With it, the tool excludes the whole ambiguous group rather than assigning a guessed correct label. Recovery still fails if either training or development loses a required class. It never moves rows between splits.

This is a metadata operation over 311,434 rows, so it still needs time and memory. It does not perform the expensive image-reading phase. No specific runtime is promised.

## 3. Point your training commands to the recovered selection

After recovery succeeds, run this Python cell in your existing notebook session:

```python
import os
from pathlib import Path
from data.manifest import verify_manifest_gate

prepared = Path('/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv')
gate = verify_manifest_gate(prepared, enforce_class_coverage=True, require_hashes=True)
os.environ['MANIFEST'] = str(prepared)
print(f'Ready to reuse: {prepared} ({gate["samples"]:,} total train + development rows)')
```

Continue with your RGB-128 training command, then wavelet-128 and fusion. The existing training commands read `$MANIFEST`. Keep the original setup's other variables. Do not rerun the setup cell afterward without configuring reuse: it would replace `MANIFEST` again.

For future fresh model runs, the updated `TRAINING_COMMANDS.md` setup has:

```python
PREPARED_MANIFEST = Path('/kaggle/working/prepared_data/recovered_100k_seed42_v2/selected_manifest.csv')
```

Set that value in the setup itself. It keeps the prepared data separate from new `RUN_ROOT` checkpoint directories and skips preparation. The main `colab_pilot_pipeline.ipynb` has the same `PREPARED_MANIFEST` setting. The saved CSV controls the cohort in reuse mode; changing `TRAIN_SIZE` does not expand it.

For resuming an existing trained checkpoint, keep that checkpoint's original manifest. This recovered CSV is a new training cohort and cannot be substituted into an old run's resume command.

## 4. Save these outputs

Save/download the entire `/kaggle/working/prepared_data/recovered_100k_seed42_v2` folder:

- `selected_manifest.csv`: cleaned selection pointing to existing image files.
- `selected_manifest.verified.json`: gate bound to the cleaned CSV digest.
- `audit_report.json`: successful overlap audit with inherited hash-evidence provenance.
- `recovery_report.json`: exact counts, exclusions, and original evidence digests.
- `excluded_training.csv`: training rows excluded from this selection and why.
- `excluded_samples.csv`: all excluded rows, including evaluation rows, with reasons.
- `label_conflicts.csv`: contradictory-content rows and their connected group members, including original labels and paths.
- `conflict_summary.json`: conflict counts and any unresolved evaluation-partition overlaps.

Keep the old `selected_manifest.csv` and old `audit_report.json` as provenance evidence, and keep `pool_manifest.fixed.csv` for future larger selections. Saving these files does not require another image copy. Kaggle working storage must be saved/downloaded before the session is discarded.

## 5. Future sizes and interrupted audits

- Same repaired selection, any model/seed/GPU configuration: reuse it; no inventory, selection, or hash audit needs repeating while images and paths remain unchanged. Training still reads its metadata and actual batches.
- New 200K/400K/full selection: reuse the full inventory. Previously unexamined images still need hash checking. Do not claim a larger set is verified by this smaller audit.
- For a future new audit, add a persistent cache path:

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"
python -u prepare_dataset.py audit \
  --manifest "$MANIFEST" \
  --root "$DATA_ROOT" \
  --output "$RUN_ROOT/audit_report.json" \
  --hash_cache /kaggle/working/prepared_data/hashes.sqlite \
  --hashes
```

The SQLite cache commits incrementally and reuses hashes when the same path's size and nanosecond modification time match. Keep this database for subsequent audits; changing file paths or timestamps can require rehashing. It is not a replacement for revalidation of deliberately changed data. Your previous audit did not write this cache; the recovery command above instead reuses its already-completed CSV/report evidence.

Quarantining contradictory labels is a documented data-quality exclusion, not a performance-based filter. The report states whether evaluation membership changed and records before/after split/class counts. Do not otherwise shrink or rebalance validation to hide leakage, and do not compare metrics from different cohorts as though they were the same experiment. Development duplicates within the same split remain present, with linked group IDs for group-aware reporting. Source and identity metadata still need to be correct; recovery does not invent missing provenance or prove that all near-duplicates have been found.

Implementation validation: source and command syntax checked only; regression tests written but NOT RUN locally. No actual Kaggle files were recovered from this machine, and no training/inference/image scans were run here.
