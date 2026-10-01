# Prompt 2 diagnostic status

Status: pending the exact historical checkpoint and its original run options.

Read-only inspection found no `.pth`, `.pt`, or `.ckpt` model artifact in this repository (outside the environment), and only the placeholder `checkpoints/experiment_name/opt.txt`. Therefore the reported 60% result cannot be reproduced or attributed to label orientation, preprocessing, missing experts, calibration, or domain shift.

The supplied external folders were inspected only at directory level and with four representative PNGs:

- `F:\LDM\LDM` is treated as fake-only; root archives/downloads are excluded.
- `F:\val to be deleted 2\fake` and `F:\val to be deleted 2\real` provide folder labels, but no documented provenance or matched-control metadata.
- The sampled PNGs are 256 × 256. No claim is made that this is the historical model preprocessing.

`diagnostic_manifest_v1.csv` contains six explicit-label development records with `unknown` group/provenance fields. It is not a benchmark or a leakage audit. The previously inspected external data must remain development-only.

Required to continue: the exact checkpoint responsible for the 60% result and its original configuration/label mapping (or the associated full-protocol checkpoint). With those, the saved diagnostic manifest can be expanded reproducibly and evaluated without fitting a threshold on it.
