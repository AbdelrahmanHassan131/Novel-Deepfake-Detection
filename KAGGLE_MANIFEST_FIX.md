# Fix the train/dev group conflict from independent-image inventory

**Completed audit reports SHA256 train/dev overlaps?** Follow [KAGGLE_PREPARATION_RECOVERY.md](KAGGLE_PREPARATION_RECOVERY.md). Reuse the completed hash evidence and preserve validation; do not rerun inventory to repair actual duplicate content. The main notebook now accepts `PREPARED_MANIFEST` to reuse the recovered selection.

The old `--independent_images` inventory used `source + filename stem` as the group ID. For example, `train/real/0001.png` and `val/fake/0001.jpg` both became `diffgan_0001`. The selector correctly rejected this shared group across splits, but the inventory had created a false relationship.

`prepare_dataset.py` now derives independent-image group IDs from the source and complete relative path, including directories and extension. Existing sample IDs stay unchanged. True content duplicates and known identity/original/video links are still checked. Conflict errors now include representative paths and metadata.

## What to do on Kaggle

1. Update **`/kaggle/working/Novel-Deepfake-Detection/prepare_dataset.py`** with the corrected file from this repository. Updating only the notebook commands cannot fix the old implementation.
2. Keep your existing setup variables. No checkpoint or training is needed for this repair. Use an experiment directory in which training has not started.
3. Replace the failed preparation cell with the cell below. It creates a NEW inventory under `RUN_ROOT`, preserves your original `pool_manifest.csv`, then selects and audits the images. Your training commands already use `MANIFEST`, so they need no change.

Use `--independent_images` only for verified independent still images. For video frames, multiple photos of a shared identity, or linked real/manipulated pairs, use their verified relationship metadata instead. Do not regenerate an already enriched manifest with this independent-image shortcut. The collection name `diffgan` also does not establish the original source of every image in a mixed collection; preserve original-source metadata for cross-source experiments.

```bash
%%bash
set -euo pipefail
cd "$REPO_ROOT"

# New CSV: do not overwrite the original inventory or any enriched metadata.
FIXED_POOL="$RUN_ROOT/pool_manifest.fixed.csv"
python prepare_dataset.py inventory \
  --root "$DATA_ROOT" \
  --source diffgan \
  --output "$FIXED_POOL" \
  --independent_images

extra=()
if [[ "$ALLOW_SHORTFALL" == "1" ]]; then extra+=(--allow_shortfall); fi
python prepare_dataset.py pilot \
  --manifest "$FIXED_POOL" \
  --root "$DATA_ROOT" \
  --output "$MANIFEST" \
  --train_size "$TRAIN_SIZE" \
  --frame_cap "$FRAME_CAP" \
  --dev_ratio 0.1 --test_ratio 0.1 \
  --require_groups --seed "$SEED" "${extra[@]}"

python prepare_dataset.py audit \
  --manifest "$MANIFEST" \
  --root "$DATA_ROOT" \
  --output "$RUN_ROOT/audit_report.json" \
  --hashes
```

After this succeeds, keep the corrected full inventory for later sizes. In your setup cell for future experiments, set `POOL_MANIFEST` to the actual `pool_manifest.fixed.csv` path printed under this run's directory. Do not rebuild it from the obsolete CSV. Keep images in their existing directory.

If the corrected run still reports a conflict or the hash audit finds overlaps, inspect the paths/metadata in the error. That may be a real duplicate or relationship across the existing train/val folders and requires a deliberate data correction. Do not disable the audit or mark everything `unassigned` simply to suppress it. Likewise, this fix does not bypass an insufficient-real/fake-images error: the requested count still has to be available.

Local verification: syntax and command-interface checks only. Regression cases were added for same filenames across splits/classes/extensions, stable IDs across roots, source separation, and continued rejection of true duplicates/shared identities. Runtime tests, image scanning, training, and inference have NOT been run locally.
