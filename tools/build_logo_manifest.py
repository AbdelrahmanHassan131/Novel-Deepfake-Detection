#!/usr/bin/env python
"""Leave-One-Generator-Out (LOGO) Cross-Domain Manifest Builder & Shortcut Auditor.

Builds a strictly leak-free manifest where:
  - ``train`` contains ONLY seen fake generators + disjoint real images
  - ``dev``   contains ONLY held-out unseen dev generator(s) + disjoint real images
              (so best.pth and threshold calibration optimize for CROSS-DOMAIN generalization!)
  - ``external_test`` (optional) contains completely untouched final test generator(s)

Also audits file extensions (JPG vs PNG) and image resolutions across classes/generators
to warn immediately if packaging shortcuts exist before you spend GPU hours training.

Supports both:
  1. Nested generator folders: ``root/fake/ADM/...``, ``root/fake/DDIM/...``, ``root/real/CelebA/...``
  2. Flat folders with generator filename prefixes: ``ADM_0001.png``, ``DDIM__...``, ``CoDiff__...``
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.drct_manifest import file_sha256, write_drct_manifest
from data.drct_preparation import generate_gate_sidecar

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def infer_generator_name(img_path: Path, class_root: Path) -> str:
    """Infer generator/source tag from nested subfolder or filename prefix."""
    rel = img_path.relative_to(class_root)
    if len(rel.parts) >= 2:
        return rel.parts[0]
    stem = img_path.stem
    if "__" in stem:
        return stem.split("__", 1)[0]
    if "_" in stem:
        prefix = stem.split("_", 1)[0]
        if any(c.isalpha() for c in prefix):
            return prefix
    return class_root.name


def scan_class_folder(class_dir: Path, label: int) -> List[Dict[str, Any]]:
    """Scan all images under a class folder and detect generator tags & extensions."""
    items = []
    for p in sorted(class_dir.rglob("*.*")):
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        gen = infer_generator_name(p, class_dir)
        items.append({
            "path": p.resolve(),
            "label": int(label),
            "generator": gen,
            "ext": p.suffix.lower(),
        })
    return items


def audit_shortcut_risks(real_items: List[Dict[str, Any]], fake_items: List[Dict[str, Any]], sample_res: int = 128) -> Dict[str, Any]:
    """Check extension and resolution distributions across real and fake pools."""
    real_exts = Counter(x["ext"] for x in real_items)
    fake_exts = Counter(x["ext"] for x in fake_items)
    fake_gens = Counter(x["generator"] for x in fake_items)
    real_srcs = Counter(x["generator"] for x in real_items)

    def _sample_dims(items):
        if not items:
            return {}
        rng = random.Random(42)
        subset = rng.sample(items, min(len(items), sample_res))
        dims = Counter()
        for it in subset:
            try:
                with Image.open(it["path"]) as im:
                    dims[f"{im.width}x{im.height}"] += 1
            except Exception:
                dims["unreadable"] += 1
        return dict(dims.most_common(5))

    warnings = []
    real_png_ratio = real_exts.get(".png", 0) / max(1, len(real_items))
    fake_png_ratio = fake_exts.get(".png", 0) / max(1, len(fake_items))
    if abs(real_png_ratio - fake_png_ratio) > 0.35:
        warnings.append(
            f"FORMAT SHORTCUT RISK: PNG ratio differs strongly between real ({real_png_ratio:.1%}) "
            f"and fake ({fake_png_ratio:.1%}). Use symmetric JPEG/blur forensic augmentations!"
        )

    return {
        "real_count": len(real_items),
        "fake_count": len(fake_items),
        "real_sources": dict(real_srcs),
        "fake_generators": dict(fake_gens),
        "real_extensions": dict(real_exts),
        "fake_extensions": dict(fake_exts),
        "real_top_resolutions": _sample_dims(real_items),
        "fake_top_resolutions": _sample_dims(fake_items),
        "warnings": warnings,
    }


def build_logo_manifest(
    real_dir: Path,
    fake_dir: Path,
    output_manifest: Path,
    dev_generators: Sequence[str],
    test_generators: Sequence[str] = (),
    seed: int = 42,
    max_train_per_class: int = 0,
    max_dev_per_class: int = 0,
    compute_hashes: bool = True,
) -> Dict[str, Any]:
    """Build a Leave-One-Generator-Out manifest with disjoint real splits and audit report."""
    real_items = scan_class_folder(Path(real_dir), label=0)
    fake_items = scan_class_folder(Path(fake_dir), label=1)
    if not real_items or not fake_items:
        raise ValueError(f"Both real ({len(real_items)}) and fake ({len(fake_items)}) folders must contain images.")

    audit = audit_shortcut_risks(real_items, fake_items)
    dev_set = {g.strip().lower() for g in dev_generators if g.strip()}
    test_set = {g.strip().lower() for g in test_generators if g.strip()}
    if not dev_set:
        raise ValueError("Provide at least one held-out generator in --dev_generators for cross-domain model selection.")
    if dev_set & test_set:
        raise ValueError(f"Overlap between dev_generators and test_generators: {dev_set & test_set}")

    fake_train, fake_dev, fake_test = [], [], []
    for item in fake_items:
        g_low = item["generator"].lower()
        if g_low in test_set:
            fake_test.append(item)
        elif g_low in dev_set:
            fake_dev.append(item)
        else:
            fake_train.append(item)

    if not fake_train:
        raise ValueError(f"0 fake training samples left after holding out dev={dev_set}, test={test_set}. Found generators: {list(audit['fake_generators'].keys())}")
    if not fake_dev:
        raise ValueError(f"0 fake dev samples matched --dev_generators={dev_set}. Found generators: {list(audit['fake_generators'].keys())}")

    rng = random.Random(int(seed))
    rng.shuffle(fake_train)
    rng.shuffle(fake_dev)
    rng.shuffle(fake_test)
    rng.shuffle(real_items)

    if max_train_per_class > 0:
        fake_train = fake_train[:max_train_per_class]
    if max_dev_per_class > 0:
        fake_dev = fake_dev[:max_dev_per_class]

    # Partition real images proportionally into strictly disjoint train / dev / external_test pools
    n_f_train, n_f_dev, n_f_test = len(fake_train), len(fake_dev), len(fake_test)
    total_f = n_f_train + n_f_dev + n_f_test
    n_real = len(real_items)

    n_r_dev = min(n_f_dev, max(1, int(round(n_real * (n_f_dev / total_f))))) if n_f_dev > 0 else 0
    n_r_test = min(n_f_test, max(1, int(round(n_real * (n_f_test / total_f))))) if n_f_test > 0 else 0
    remaining = max(0, n_real - n_r_dev - n_r_test)
    n_r_train = min(n_f_train, remaining) if max_train_per_class > 0 else remaining

    real_dev = real_items[:n_r_dev]
    real_test = real_items[n_r_dev:n_r_dev + n_r_test]
    real_train = real_items[n_r_dev + n_r_test:n_r_dev + n_r_test + n_r_train]

    # Match 1:1 balance in dev and test; cap train to avoid severe imbalance
    min_dev = min(len(real_dev), len(fake_dev))
    real_dev, fake_dev = real_dev[:min_dev], fake_dev[:min_dev]
    if fake_test and real_test:
        min_test = min(len(real_test), len(fake_test))
        real_test, fake_test = real_test[:min_test], fake_test[:min_test]

    seen_hashes = set()
    records = []

    def _append_split(items: List[Dict[str, Any]], split_name: str):
        for idx, it in enumerate(items):
            fpath = it["path"]
            f_hash = file_sha256(fpath) if compute_hashes else hashlib.sha256(str(fpath).encode()).hexdigest()
            if f_hash in seen_hashes:
                continue
            seen_hashes.add(f_hash)
            lbl = it["label"]
            sid = f"{split_name}_{it['generator']}_{idx:06d}_{f_hash[:8]}"
            records.append({
                "sample_id": sid,
                "parent_id": sid,
                "path": str(fpath),
                "label": lbl,
                "split": split_name,
                "sample_kind": "real_original" if lbl == 0 else "fake_original",
                "group_id": f"{split_name}_{it['generator']}_{idx:06d}",
                "dataset_source": f"{'real' if lbl == 0 else 'fake'}_{it['generator']}",
                "generator": it["generator"],
                "sha256": f_hash,
                "rec_model_revision": "none",
                "rec_settings": "{}",
                "rec_seed": "0",
            })

    # Process evaluation splits first so any duplicate content hash is excluded from train, never eval
    _append_split(real_test + fake_test, "external_test")
    _append_split(real_dev + fake_dev, "dev")
    _append_split(real_train + fake_train, "train")

    output_manifest = Path(output_manifest).resolve()
    output_manifest.parent.mkdir(parents=True, exist_ok=True)
    write_drct_manifest(output_manifest, records)
    gate_path = generate_gate_sidecar(output_manifest, require_hashes=compute_hashes)

    split_counts = Counter((r["split"], r["label"]) for r in records)
    report = {
        "status": "success",
        "manifest": str(output_manifest),
        "gate": str(gate_path),
        "dev_generators": sorted(dev_set),
        "test_generators": sorted(test_set),
        "split_counts": {f"{s}/label_{l}": c for (s, l), c in sorted(split_counts.items())},
        "shortcut_audit": audit,
    }
    report_path = output_manifest.with_suffix(".logo_report.json")
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def parse_args():
    p = argparse.ArgumentParser(description="Build a Leave-One-Generator-Out (LOGO) cross-domain manifest.")
    p.add_argument("--real_dir", required=True, type=Path, help="Folder containing real images")
    p.add_argument("--fake_dir", required=True, type=Path, help="Folder containing fake images (nested by generator or prefixed)")
    p.add_argument("--output_manifest", required=True, type=Path, help="Output CSV manifest path")
    p.add_argument("--dev_generators", required=True, help="Comma-separated held-out generator name(s) for cross-domain dev selection (e.g. 'DDIM')")
    p.add_argument("--test_generators", default="", help="Optional comma-separated untouched generator name(s) for external_test (e.g. 'LDM')")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_train_per_class", type=int, default=0)
    p.add_argument("--max_dev_per_class", type=int, default=0)
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    rep = build_logo_manifest(
        real_dir=args.real_dir,
        fake_dir=args.fake_dir,
        output_manifest=args.output_manifest,
        dev_generators=[g.strip() for g in args.dev_generators.split(",") if g.strip()],
        test_generators=[g.strip() for g in args.test_generators.split(",") if g.strip()],
        seed=args.seed,
        max_train_per_class=args.max_train_per_class,
        max_dev_per_class=args.max_dev_per_class,
    )
    print(json.dumps(rep, indent=2))
