# DRCT-128 direct repair report

Reviewed and repaired: 2026-10-15.

The previous completion report missed executable workflow failures. These fixes are implemented in the repository; no additional implementation handoff is needed for them. Follow `COLAB_DRCT_128_TRAINING.md` for the updated Colab sequence.

## Repairs made

| Area | Failure found | Implemented correction |
|---|---|---|
| Evaluation | DRCT CSV labels remained strings in dataset records, causing metric calculation to fail after inference. | Normalize record labels to integers, consistently with loader targets. |
| Manifest serialization | Identity/video/generator metadata was discarded when writing a DRCT CSV. | Preserve all extra columns and inherit provenance in imported/generated reconstruction rows. |
| Audit | A failed hash audit could print a warning and exit successfully; supplied in-memory rows could differ from the written file. | Audit the serialized file, reject failures with nonzero CLI exit, and revoke stale verification gates. |
| Reconstruction pairing | One filename could match multiple parents. A real-only reconstruction set could be declared ready for full DRCT. | Reject ambiguous reuse; add explicit mapping with parent/content hashes and generation metadata; require reconstruction coverage for both classes. Filename-only matches remain unverified. |
| Reconstruction preparation | The command could appear successful without an actual generation backend. | Fail with an actionable message when no backend is supplied; direct users to the precomputed importer. |
| Contrastive loss | Default pairwise-distance computation produced nonzero distances for identical vectors at realistic batch sizes. | Compute distances from direct vector differences; verify exact-zero loss and finite gradients at batches 2, 32 and 64. |
| Native checkpoint import | Legitimate qualified CLIP text positional/logit-scale entries were not excluded. | Explicitly exclude those text entries; still reject missing required detector weights. |
| Input contract | Wrong CLIP resolution and nonfinite loss settings were not rejected early. | Reject non-224 CLIP input sizes and nonfinite objective settings. |
| Checkpoint writing | Drive copying could observe a partially written final checkpoint. | Save to a temporary file, then atomically publish the completed checkpoint. |
| Colab persistence | Wrong experiment/gate paths, incomplete restoration, backup shutdown races and incomplete resume arguments. | Add tested workflow helpers for exact run descriptors, full experiment mirroring, final synchronization on caught failure, and resume with the complete original command. |
| Preparation backup | Reconstruction files could be missing/outside the backed-up root or corrupted in the bundle. | Check containment and presence; verify their audited hashes while copying, without rehashing the original dataset on restore. |
| Colab commands | Fresh training lacked explicit pretrained weight access; benchmark settings differed; obsolete gate flags and incompatible dependency pins remained. | Rewrite the guide with explicit initialization, correct paths, compatible dependency installation, preparation/restore, benchmark, training, resume, calibration and MHA steps. |

## Model interface retained

- `DRCT_128` returns one raw binary logit, with real=0 and fake=1.
- Its learned penultimate representation remains 128-dimensional for MHA.
- Fusion uses the DRCT RGB preprocessing and keeps the wavelet branch separate.
- Train a new fusion head; an existing Wang-based fusion head is not interchangeable.
- Calibration uses raw logits and the saved checkpoint's internal development predictions.

## Validation

CPU-only tests use temporary tiny images and dummy/injected encoders. No production training, downloaded pretrained weights, or external-dataset inference was run.

The focused regression selection contains 163 tests across DRCT, MHA, checkpoint/resume, preprocessing and RGB behavior. The broad run passed 162; one older test omitted the engineering-pilot flag now required because directory inventories explicitly mark image-level grouping as unverified. Its fixture was corrected to pass both existing engineering flags without weakening production validation; the targeted rerun is recorded below.

New workflow coverage actually executes preparation, `train.py`, `evaluate.py`, logit calibration, deletion of the isolated temporary local experiment, restoration from its backup and resume. It checks the checkpoint-bound calibration artifact and that a completed one-epoch run does not resume with the default 10,000-epoch budget.

Additional regressions cover failed audits, retained grouping metadata, ambiguous reconstruction ownership, missing fake reconstructions, exact-zero contrastive distances, interrupted checkpoint publication, backup on failed training, corrupted reconstruction backups and all Python cells in the Colab guide.

Targeted rerun: **2 passed in 18.30 seconds** (the corrected pilot fixture and the updated guide's Python-cell validation). All 163 selected regression cases have now passed across the broad and follow-up runs. The broad run emitted 14 existing PyTorch/pkg_resources deprecation warnings. `git diff --check` passed.

## What still requires real experiment inputs

1. **Full DRCT needs genuine diffusion reconstructions of both real and fake training originals.** The repository has a verified-mapping importer and a callback interface, not a verified diffusion generator. Supply actual reconstruction images and their mapping; do not substitute copies or ordinary augmentations.
2. With only original real/fake folders, explicitly select `rgb_control` in the guide. This trains pretrained CLIP + a 128D projection + binary BCE; it is not full DRCT.
3. Actual CLIP pretrained loading, GPU throughput/AMP and Colab Drive behavior still need the guide's bounded GPU benchmark. CPU integration tests cannot establish their performance.
4. This is a 128D binary adaptation, not a certified reproduction of the paper. Its PIL preprocessing has not been shown pixel-identical to the upstream pipeline. No unseen-domain accuracy is guaranteed by these repairs.
5. Directory inventory preserves the supplied train/dev split and detects exact-content conflicts. Without verified identity/video/source metadata it cannot establish independence; the guide explicitly labels those runs engineering pilots.
6. The repeatedly inspected 60,000-image collection is external development data. Keep its status explicit when reporting results.

## Next action

Push these local changes, then clone the updated repository in Colab and follow `COLAB_DRCT_128_TRAINING.md`. Keep the full preparation bundle and run directory on Drive. Resume using the recorded commit and exact run descriptor. Existing manifests that lost provenance must be rebuilt from the original metadata; this repair cannot recover discarded fields automatically.
