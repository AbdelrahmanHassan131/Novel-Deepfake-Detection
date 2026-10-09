"""Bounded paired-preprocessing format sensitivity diagnostic utility.

Compares model behavior on original decoded RGB vs:
  (a) lossless PNG round-trip (RGB-equality control)
  (b) JPEG quality 95 round-trip (mild compression intervention, subsampling=0)
  (c) JPEG quality 75 round-trip (moderate compression intervention, subsampling=0)

Strict contract:
- Validates manifests before loading model (rejects empty or >128-row manifests, invalid labels, duplicate IDs, missing files).
- Preserves original disk images untouched (in-memory interventions only; SHA256 verified before and after).
- Strict PNG RGB equality control: fails if pixel diff != 0.0 or tensor diff != 0.0.
- Production preprocessing: reuses evaluation transform matching RGBDataset and respects checkpoint options.
- Canonical scores: honors model score_sign and rejects unexpected output shapes.
- Detailed reporting: broken down by cohort (training diagnostic vs external dev), class, and stratum.
"""

import argparse
import copy
import csv
import hashlib
import io
import json
import os
import platform
import re
import sys
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, Any, List, Tuple

import numpy as np
import PIL
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# Ensure project root is in path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.transforms.augmentations import data_augment
from data.transforms.identity import identity_image
from data.transforms.resize import custom_resize
from evaluation.checkpoint_loader import CheckpointLoader

MAX_SAMPLES_CEILING = 128
SUPPORTED_ARCHITECTURES = {"Wang2020Raw", "Wang2020_128", "DummyMockModel"}

RE_FAMILY = re.compile(r"^(vid_[0-9a-fA-F]{64})_face_\d+_\d+\.[a-zA-Z0-9]+$")
RE_NUM = re.compile(r"^\d+\.[a-zA-Z0-9]+$")


def compute_file_sha256(path: Path) -> str:
    """Compute SHA256 hex digest of a file."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_and_validate_manifest(manifest_path: str) -> List[Dict[str, Any]]:
    """Strict manifest loader and validator.
    
    Rejects manifests that are empty, exceed 128 rows, contain invalid labels,
    contain duplicate sample IDs, or have unresolvable image paths.
    """
    m_path = Path(manifest_path).resolve()
    if not m_path.is_file():
        raise FileNotFoundError(f"Manifest file not found: {m_path}")

    with open(m_path, "r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        fieldnames = set(reader.fieldnames or [])
        required = {"sample_id", "path", "label"}
        missing = required - fieldnames
        if missing:
            raise ValueError(f"Manifest missing required columns: {sorted(missing)}")
        rows = list(reader)

    if len(rows) == 0:
        raise ValueError("Manifest contains 0 rows (manifest is empty).")

    if len(rows) > MAX_SAMPLES_CEILING:
        raise ValueError(
            f"Manifest exceeds bounded maximum of {MAX_SAMPLES_CEILING} samples ({len(rows)} provided). "
            f"Please provide a bounded manifest of at most {MAX_SAMPLES_CEILING} samples."
        )

    seen_ids = set()
    validated = []
    for idx, r in enumerate(rows, 1):
        sid = r.get("sample_id", "").strip()
        if not sid:
            raise ValueError(f"Manifest line {idx + 1}: missing sample_id")
        if sid in seen_ids:
            raise ValueError(f"Manifest line {idx + 1}: duplicate sample_id '{sid}'")
        seen_ids.add(sid)

        lbl_str = r.get("label", "").strip()
        if lbl_str not in ("0", "1"):
            raise ValueError(f"Manifest line {idx + 1} ('{sid}'): label must be strictly '0' or '1', got '{lbl_str}'")
        label = int(lbl_str)

        split = r.get("split", "").strip()
        if "split" in fieldnames and not split:
            raise ValueError(f"Manifest line {idx + 1} ('{sid}'): split is required and cannot be empty")

        raw_p = r.get("path", "").strip()
        if not raw_p:
            raise ValueError(f"Manifest line {idx + 1} ('{sid}'): path cannot be empty")

        p = Path(raw_p)
        if not p.is_absolute():
            p = (m_path.parent / p).resolve()
        else:
            p = p.resolve()

        if not p.is_file():
            raise FileNotFoundError(f"Image file not found for sample_id '{sid}': {p}")

        actual_sha = compute_file_sha256(p)
        expected_sha = r.get("sha256", "").strip()
        if expected_sha and expected_sha.lower() not in ("", "unknown", "none") and expected_sha != actual_sha:
            raise ValueError(
                f"Manifest line {idx + 1} ('{sid}'): SHA256 mismatch for {p}: expected {expected_sha}, got {actual_sha}"
            )

        cohort = r.get("cohort", "").strip()
        if not cohort or cohort not in ("training_diagnostic", "external_development"):
            raise ValueError(
                f"Manifest line {idx + 1} ('{sid}'): explicit 'cohort' metadata is required and must be "
                f"'training_diagnostic' or 'external_development', got '{cohort}'"
            )

        subcoll = r.get("subcollection", r.get("stratum", "")).strip()
        if not subcoll:
            bname = p.name
            if RE_NUM.match(bname):
                subcoll = "numeric_real" if label == 0 else "numeric_fake"
            elif RE_FAMILY.match(bname):
                subcoll = "video_real" if label == 0 else "video_fake"
            else:
                subcoll = f"{cohort}_{'fake' if label == 1 else 'real'}"

        row_data = dict(r)
        row_data.update({
            "sample_id": sid,
            "path": str(p),
            "label": label,
            "cohort": cohort,
            "subcollection": subcoll,
            "image_sha256": actual_sha,
        })
        validated.append(row_data)

    return validated


def build_production_eval_transform(opt: Any) -> transforms.Compose:
    """Build evaluation transform pipeline exactly matching production RGBDataset."""
    eval_opt = copy.deepcopy(opt)
    eval_opt.isTrain = False
    eval_opt.no_flip = True
    eval_opt.data_aug = False
    eval_opt.blur_prob = 0.0
    eval_opt.jpg_prob = 0.0
    eval_opt.noise_prob = 0.0
    eval_opt.downscale_prob = 0.0

    crop_size = getattr(eval_opt, "cropSize", getattr(eval_opt, "crop_size", 224))

    if getattr(eval_opt, "no_crop", False):
        crop_func = transforms.Lambda(identity_image)
    else:
        crop_func = transforms.CenterCrop(crop_size)

    flip_func = transforms.Lambda(identity_image)

    if getattr(eval_opt, "no_resize", False):
        rz_func = transforms.Lambda(identity_image)
    else:
        rz_func = transforms.Lambda(partial(custom_resize, opt=eval_opt))

    return transforms.Compose([
        rz_func,
        transforms.Lambda(partial(data_augment, opt=eval_opt)),
        crop_func,
        flip_func,
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])


def apply_interventions(orig_pil: Image.Image) -> Dict[str, Tuple[Image.Image, float]]:
    """Apply paired interventions to a PIL RGB image in memory.
    
    Returns:
        dict mapping condition_name -> (processed_pil_image, max_pixel_abs_diff_vs_original)
    """
    orig_rgb = orig_pil.convert("RGB")
    orig_arr = np.array(orig_rgb, dtype=np.float32)

    interventions = {}
    # Condition 0: original
    interventions["original"] = (orig_rgb, 0.0)

    # Condition 1: PNG lossless round-trip (RGB equality control)
    png_buf = io.BytesIO()
    orig_rgb.save(png_buf, format="PNG")
    png_buf.seek(0)
    png_rgb = Image.open(png_buf).convert("RGB")
    png_arr = np.array(png_rgb, dtype=np.float32)
    png_diff = float(np.max(np.abs(orig_arr - png_arr)))
    interventions["png_control"] = (png_rgb, png_diff)

    # Condition 2: JPEG Quality 95 round-trip (subsampling=0)
    jpg95_buf = io.BytesIO()
    orig_rgb.save(jpg95_buf, format="JPEG", quality=95, subsampling=0)
    jpg95_buf.seek(0)
    jpg95_rgb = Image.open(jpg95_buf).convert("RGB")
    jpg95_arr = np.array(jpg95_rgb, dtype=np.float32)
    jpg95_diff = float(np.max(np.abs(orig_arr - jpg95_arr)))
    interventions["jpeg95"] = (jpg95_rgb, jpg95_diff)

    # Condition 3: JPEG Quality 75 round-trip (subsampling=0)
    jpg75_buf = io.BytesIO()
    orig_rgb.save(jpg75_buf, format="JPEG", quality=75, subsampling=0)
    jpg75_buf.seek(0)
    jpg75_rgb = Image.open(jpg75_buf).convert("RGB")
    jpg75_arr = np.array(jpg75_rgb, dtype=np.float32)
    jpg75_diff = float(np.max(np.abs(orig_arr - jpg75_arr)))
    interventions["jpeg75"] = (jpg75_rgb, jpg75_diff)

    return interventions


def compute_slice_metrics(subset: List[Dict[str, Any]], condition: str) -> Dict[str, Any]:
    """Compute confusion, recall, balanced accuracy, and score drift for a subset."""
    if not subset:
        return {"samples": 0}

    cond_rows = [r for r in subset if r["condition"] == condition]
    orig_rows = [r for r in subset if r["condition"] == "original"]
    assert len(cond_rows) == len(orig_rows), f"Condition row count mismatch: {len(cond_rows)} vs {len(orig_rows)}"

    n_samples = len(cond_rows)
    n_real = sum(1 for r in cond_rows if r["label"] == 0)
    n_fake = sum(1 for r in cond_rows if r["label"] == 1)

    tp = sum(1 for r in cond_rows if r["label"] == 1 and r["prediction_at_05"] == 1)
    tn = sum(1 for r in cond_rows if r["label"] == 0 and r["prediction_at_05"] == 0)
    fp = sum(1 for r in cond_rows if r["label"] == 0 and r["prediction_at_05"] == 1)
    fn = sum(1 for r in cond_rows if r["label"] == 1 and r["prediction_at_05"] == 0)

    real_recall = float(tn / n_real) if n_real > 0 else None
    fake_recall = float(tp / n_fake) if n_fake > 0 else None
    bacc = float((real_recall + fake_recall) / 2) if (real_recall is not None and fake_recall is not None) else None

    signed_logit_shifts = [c["signed_logit_shift"] for c in cond_rows]
    abs_logit_shifts = [c["abs_logit_shift"] for c in cond_rows]
    signed_prob_shifts = [c["signed_prob_shift"] for c in cond_rows]
    abs_prob_shifts = [c["abs_prob_shift"] for c in cond_rows]
    flips = sum(c["decision_flipped"] for c in cond_rows)

    return {
        "samples": n_samples,
        "class_denominators": {"real": n_real, "fake": n_fake},
        "confusion_at_05": {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "total_errors": fp + fn},
        "rates": {
            "real_recall": real_recall,
            "fake_recall": fake_recall,
            "balanced_accuracy": bacc,
        },
        "decision_flips_vs_original": {
            "count": flips,
            "rate": float(flips / n_samples) if n_samples > 0 else 0.0,
        },
        "score_shifts_vs_original": {
            "signed_logit_shift_mean": float(np.mean(signed_logit_shifts)),
            "signed_logit_shift_std": float(np.std(signed_logit_shifts)),
            "abs_logit_shift_mean": float(np.mean(abs_logit_shifts)),
            "abs_logit_shift_max": float(np.max(abs_logit_shifts)),
            "signed_prob_shift_mean": float(np.mean(signed_prob_shifts)),
            "signed_prob_shift_std": float(np.std(signed_prob_shifts)),
            "abs_prob_shift_mean": float(np.mean(abs_prob_shifts)),
            "abs_prob_shift_max": float(np.max(abs_prob_shifts)),
        },
        "tensor_distortion_vs_original": {
            "mean_l1_diff": float(np.mean([c["tensor_l1_diff"] for c in cond_rows])),
            "max_linf_diff": float(np.max([c["tensor_linf_diff"] for c in cond_rows])),
            "max_pixel_diff": float(np.max([c["pixel_max_diff"] for c in cond_rows])),
        }
    }


def run_format_diagnostic(
    checkpoint_path: str,
    manifest_path: str,
    output_dir: str,
    device: str = "cpu",
    batch_size: int = 16,
    num_workers: int = 0,
    model_override: Any = None,
) -> Dict[str, Any]:
    """Run format sensitivity evaluation on a bounded manifest."""
    if num_workers > 0:
        raise ValueError("Only --workers 0 is supported in bounded diagnostic on Windows.")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive (> 0), got {batch_size}")

    # 1. Strict input validation BEFORE loading model
    rows = load_and_validate_manifest(manifest_path)
    print(f"Validated bounded manifest: {len(rows)} samples.")

    # 2. Output directory creation and writability check (reject nonempty directory)
    out_path = Path(output_dir).resolve()
    if out_path.exists() and any(out_path.iterdir()):
        raise ValueError(
            f"Output directory '{output_dir}' already exists and is not empty. "
            f"Please specify a fresh, empty directory to preserve historical evidence."
        )
    out_path.mkdir(parents=True, exist_ok=True)
    test_probe = out_path / ".probe_write"
    try:
        with open(test_probe, "w", encoding="utf-8") as f:
            f.write("probe")
        test_probe.unlink()
    except Exception as e:
        raise PermissionError(f"Output directory '{output_dir}' is not writable: {e}") from e

    # 3. Model loading and option extraction
    dev = torch.device(device)
    if model_override is not None:
        base_model = model_override
        arch = getattr(base_model, "arch", "DummyMockModel")
        ckpt_sha = "dummy_mock_model"
        ckpt_meta = {"arch": arch, "checkpoint_path": "in_memory_dummy"}
        raw_model = getattr(base_model, "model", base_model)
        score_sign = float(getattr(base_model, "score_sign", 1.0))
        model_opt = getattr(base_model, "opt", SimpleNamespace(
            cropSize=224, loadSize=256, rz_interp=["bilinear"], no_crop=False, no_resize=False
        ))
    else:
        print(f"Loading checkpoint: {checkpoint_path} on {device}...")
        loader = CheckpointLoader(checkpoint_path, device=device)
        base_model = loader.load()
        arch = loader.metadata.get("arch", "unknown")
        if arch not in SUPPORTED_ARCHITECTURES:
            raise ValueError(f"Unsupported architecture '{arch}'. Diagnostic restricted to: {sorted(SUPPORTED_ARCHITECTURES)}")
        ckpt_sha = loader.metadata.get("checkpoint_sha256", "unknown")
        ckpt_meta = loader.metadata
        raw_model = getattr(base_model, "model", base_model)
        score_sign = float(getattr(base_model, "score_sign", 1.0))
        model_opt = getattr(base_model, "opt", SimpleNamespace(
            cropSize=224, loadSize=256, rz_interp=["bilinear"], no_crop=False, no_resize=False
        ))

    raw_model.eval()
    raw_model.to(dev)

    # 4. Production preprocessing transform
    transform = build_production_eval_transform(model_opt)
    effective_options = {
        "cropSize": getattr(model_opt, "cropSize", 224),
        "loadSize": getattr(model_opt, "loadSize", 256),
        "rz_interp": getattr(model_opt, "rz_interp", ["bilinear"]),
        "no_crop": getattr(model_opt, "no_crop", False),
        "no_resize": getattr(model_opt, "no_resize", False),
        "score_sign": score_sign,
    }

    condition_names = ["original", "png_control", "jpeg95", "jpeg75"]
    sample_results = []
    
    # We will gather all forward pass requests in batches
    # Each sample generates 4 condition images
    print(f"Applying in-memory interventions and evaluating {len(rows)} samples (batch_size={batch_size})...")
    
    sample_records = []
    for row in rows:
        img_p = Path(row["path"])
        file_sha_before = compute_file_sha256(img_p)

        with Image.open(img_p) as pil_img:
            interventions = apply_interventions(pil_img)

        file_sha_after = compute_file_sha256(img_p)
        if file_sha_before != file_sha_after:
            raise RuntimeError(f"FATAL: File on disk was modified during reading: {img_p}")

        # Check PNG RGB pixel equality control immediately
        png_pixel_diff = interventions["png_control"][1]
        if png_pixel_diff != 0.0:
            raise RuntimeError(
                f"PNG RGB-equality control failed for {img_p}: pixel diff is {png_pixel_diff} (expected 0.0)"
            )

        # Compute tensors for all conditions
        tensors = {}
        for cname in condition_names:
            c_img, _ = interventions[cname]
            tensors[cname] = transform(c_img)

        # Check PNG tensor equality control
        png_tensor_diff = float((tensors["png_control"] - tensors["original"]).abs().max().item())
        if png_tensor_diff != 0.0:
            raise RuntimeError(
                f"PNG tensor-equality control failed for {img_p}: tensor diff is {png_tensor_diff} (expected 0.0)"
            )

        sample_records.append({
            "row": row,
            "interventions": interventions,
            "tensors": tensors
        })

    # Execute batched forward passes
    # Flatten items: list of (sample_idx, cond_name, tensor)
    flat_items = []
    for s_idx, s_rec in enumerate(sample_records):
        for cname in condition_names:
            flat_items.append((s_idx, cname, s_rec["tensors"][cname]))

    forward_outputs = {}
    actual_batch_size = max(1, batch_size)
    for b_start in range(0, len(flat_items), actual_batch_size):
        b_items = flat_items[b_start : b_start + actual_batch_size]
        batch_tensors = torch.stack([item[2] for item in b_items], dim=0).to(dev)

        with torch.no_grad():
            out = raw_model(batch_tensors)
            if isinstance(out, (tuple, list)):
                raise ValueError(
                    f"Model output is tuple/list of length {len(out)}; expected single tensor output from RGB Wang model."
                )
            if out.dim() > 2 or (out.dim() == 2 and out.shape[1] > 1):
                raise ValueError(f"Unexpected model output shape {tuple(out.shape)}; expected (B, 1) or (B,).")
            
            raw_logits = out.view(-1).cpu().numpy()
            if len(raw_logits) != len(b_items):
                raise ValueError(f"Model returned {len(raw_logits)} outputs for batch of {len(b_items)} images.")
            if np.any(np.isnan(raw_logits)) or np.any(np.isinf(raw_logits)):
                raise ValueError("Model forward pass produced non-finite logit values (NaN or Inf).")

        for (s_idx, cname, _), raw_log in zip(b_items, raw_logits):
            can_log = float(raw_log * score_sign)
            prob = float(torch.sigmoid(torch.tensor(can_log)).item())
            dec = int(prob >= 0.5)
            forward_outputs[(s_idx, cname)] = {
                "raw_logit": float(raw_log),
                "canonical_logit": can_log,
                "probability": prob,
                "decision": dec
            }

    # Assemble per-sample paired records
    all_sample_rows = []
    for s_idx, s_rec in enumerate(sample_records):
        row = s_rec["row"]
        tensors = s_rec["tensors"]
        orig_tensor = tensors["original"]
        orig_out = forward_outputs[(s_idx, "original")]
        orig_dec = orig_out["decision"]
        orig_logit = orig_out["canonical_logit"]
        orig_prob = orig_out["probability"]

        for cname in condition_names:
            c_out = forward_outputs[(s_idx, cname)]
            c_tensor = tensors[cname]
            l1_diff = float((c_tensor - orig_tensor).abs().mean().item())
            linf_diff = float((c_tensor - orig_tensor).abs().max().item())
            px_diff = s_rec["interventions"][cname][1]

            signed_logit_shift = c_out["canonical_logit"] - orig_logit
            abs_logit_shift = abs(signed_logit_shift)
            signed_prob_shift = c_out["probability"] - orig_prob
            abs_prob_shift = abs(signed_prob_shift)
            flipped = int(c_out["decision"] != orig_dec)

            rec = {
                "sample_id": row["sample_id"],
                "path": row["path"],
                "label": row["label"],
                "cohort": row["cohort"],
                "subcollection": row["subcollection"],
                "image_sha256": row["image_sha256"],
                "condition": cname,
                "pixel_max_diff": px_diff,
                "tensor_l1_diff": l1_diff,
                "tensor_linf_diff": linf_diff,
                "raw_logit": c_out["raw_logit"],
                "canonical_logit": c_out["canonical_logit"],
                "probability": c_out["probability"],
                "prediction_at_05": c_out["decision"],
                "decision_flipped": flipped,
                "signed_logit_shift": signed_logit_shift,
                "abs_logit_shift": abs_logit_shift,
                "signed_prob_shift": signed_prob_shift,
                "abs_prob_shift": abs_prob_shift,
            }
            sample_results.append(rec)

    # 5. Write sample CSV
    csv_path = out_path / "format_sensitivity_samples.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "sample_id", "path", "label", "cohort", "subcollection", "image_sha256",
            "condition", "pixel_max_diff", "tensor_l1_diff", "tensor_linf_diff",
            "raw_logit", "canonical_logit", "probability", "prediction_at_05",
            "decision_flipped", "signed_logit_shift", "abs_logit_shift",
            "signed_prob_shift", "abs_prob_shift"
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(sample_results)
    print(f"Saved sample predictions: {csv_path}")

    # 6. Build structured summary report
    png_pixel_max = max(r["pixel_max_diff"] for r in sample_results if r["condition"] == "png_control")
    png_tensor_max = max(r["tensor_linf_diff"] for r in sample_results if r["condition"] == "png_control")
    png_control_passed = (png_pixel_max == 0.0 and png_tensor_max == 0.0)

    cohorts = sorted(list(set(r["cohort"] for r in sample_results)))
    subcollections = sorted(list(set(r["subcollection"] for r in sample_results)))

    summary_by_condition = {}
    for cname in condition_names:
        cond_dict = {
            "overall": compute_slice_metrics(sample_results, cname),
            "by_cohort": {c: compute_slice_metrics([r for r in sample_results if r["cohort"] == c], cname) for c in cohorts},
            "by_class": {
                "real": compute_slice_metrics([r for r in sample_results if r["label"] == 0], cname),
                "fake": compute_slice_metrics([r for r in sample_results if r["label"] == 1], cname),
            },
            "by_subcollection": {s: compute_slice_metrics([r for r in sample_results if r["subcollection"] == s], cname) for s in subcollections}
        }
        summary_by_condition[cname] = cond_dict

    report = {
        "checkpoint_info": {
            "checkpoint_path": str(checkpoint_path),
            "checkpoint_sha256": ckpt_sha,
            "architecture": arch,
            "score_sign": score_sign,
            "metadata": ckpt_meta,
        },
        "manifest_path": str(manifest_path),
        "total_samples": len(rows),
        "effective_evaluation_options": effective_options,
        "runtime_environment": {
            "python_version": platform.python_version(),
            "pytorch_version": torch.__version__,
            "pillow_version": PIL.__version__,
            "platform": platform.platform(),
            "device": str(dev),
            "batch_size": actual_batch_size,
            "workers": num_workers,
            "batch_behavior": f"batched forward evaluation in chunks of {actual_batch_size}",
        },
        "interventions_spec": {
            "pillow_version": PIL.__version__,
            "jpeg_subsampling": 0,
            "qualities": [95, 75],
            "metadata_policy": "in-memory BytesIO stripped of Exif/JFIF app markers",
            "intervention_point": "in-memory decoded PIL RGB prior to production transforms",
        },
        "png_rgb_equality_control": {
            "status": "PASS: exact byte-for-byte decoded pixel and tensor equality across all samples" if png_control_passed else "FAIL: PNG round-trip altered decoded pixels or tensors",
            "max_pixel_abs_diff": png_pixel_max,
            "max_tensor_linf_diff": png_tensor_max,
        },
        "sample_breakdown": {
            "total_samples": len(rows),
            "cohort_counts": {c: sum(1 for r in rows if r["cohort"] == c) for c in cohorts},
            "class_counts": {"real": sum(1 for r in rows if r["label"] == 0), "fake": sum(1 for r in rows if r["label"] == 1)},
            "subcollection_counts": {s: sum(1 for r in rows if r["subcollection"] == s) for s in subcollections},
        },
        "conditions_summary": summary_by_condition,
        "cautionary_guidance": (
            "Evidence Interpretation Guardrail: Any compression intervention showing higher external accuracy "
            "(e.g., JPEG95 or JPEG75) must NOT be interpreted as an optimized deployment preprocessing setting, "
            "nor as proof that compression is the sole causal factor in generalization failure. "
            "Rather, it provides diagnostic evidence regarding model sensitivity to container compression."
        )
    }

    report_path = out_path / "format_sensitivity_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"Saved diagnostic report: {report_path}")

    return report


def parse_args():
    parser = argparse.ArgumentParser(description="Bounded Paired-Preprocessing Format Sensitivity Diagnostic")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pth)")
    parser.add_argument("--manifest", type=str, required=True, help="Path to manifest CSV (max 128 samples)")
    parser.add_argument("--output_dir", type=str, default="output/review/archive_investigation/format_sensitivity", help="Output directory")
    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu", help="Device (cpu or cuda:0)")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size (default 16)")
    parser.add_argument("--workers", type=int, default=0, help="Workers (default 0 for Windows)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_format_diagnostic(
        checkpoint_path=args.checkpoint,
        manifest_path=args.manifest,
        output_dir=args.output_dir,
        device=args.device,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )
