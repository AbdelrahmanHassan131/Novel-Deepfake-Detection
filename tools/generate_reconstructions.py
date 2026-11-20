#!/usr/bin/env python
"""Resumable Paired Diffusion / VAE Reconstruction Generator for DRCT_128 and EFFORT_128.

Generates paired reconstructed images from an audited base inventory CSV and writes
the exact ``reconstruction_mapping.csv`` required by ``data.drct_preparation.import_precomputed_reconstructions``.

Why this solves cross-domain (e.g., CelebA -> LDM) generalization:
  Passing real images (e.g., CelebA) through a pretrained Stable Diffusion VAE or
  few-step diffusion img2img model produces fake images with the EXACT same face,
  crop, lighting, identity, and resolution—forcing the detector to learn pure
  generative/latent-decoder artifacts instead of dataset shortcuts.

Supported backends:
  - ``sd15_vae``   : Fast SD 1.5 VAE encode->decode (stabilityai/sd-vae-ft-mse, ~100 img/s on L4)
  - ``sdxl_vae``   : Fast SDXL VAE encode->decode (madebyollin/sdxl-vae-fp16-fix, ~60 img/s on L4)
  - ``multi_vae``  : Deterministic 50/50 mix of sd15_vae + sdxl_vae (recommended fast multi-LDM setup)
  - ``sd_turbo``   : 2-step SD-Turbo diffusion img2img reconstruction (stabilityai/sd-turbo, true DRCT)
  - ``dummy``      : Fast offline CPU simulator for unit testing (never use for real training)

Example (Colab L4):
    python -u tools/generate_reconstructions.py \\
      --inventory /content/drct_prepared/base_inventory.csv \\
      --output_dir /content/dataset/reconstructions \\
      --mapping_csv /content/drive/MyDrive/deepfake_drct_128/reconstruction_mapping.csv \\
      --backend multi_vae --batch_size 32 --max_per_class 25000 --reconstruct_fakes \\
      --import_manifest /content/drct_prepared/selected_manifest.csv
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image, ImageFilter

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data.drct_manifest import file_sha256, read_drct_manifest


MAPPING_COLUMNS = [
    "parent_id",
    "path",
    "parent_sha256",
    "sha256",
    "rec_model_revision",
    "rec_settings",
    "rec_seed",
]


def _atomic_save_image(pil_img: Image.Image, dest: Path, fmt: str = "jpg", jpg_quality: int = 95) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest.parent / f".tmp_{dest.name}.{os.getpid()}"
    try:
        if fmt == "png":
            pil_img.save(tmp_path, format="PNG")
        else:
            pil_img.save(tmp_path, format="JPEG", quality=int(jpg_quality))
        os.replace(tmp_path, dest)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _resolve_target_size(orig_size: Tuple[int, int], max_size: int) -> Tuple[int, int]:
    """Ensure dimensions are divisible by 8 (VAE requirement) and capped at max_size."""
    w, h = orig_size
    if max_size > 0 and max(w, h) > max_size:
        scale = max_size / max(w, h)
        w = max(8, int(round(w * scale)))
        h = max(8, int(round(h * scale)))
    w = max(8, (w // 8) * 8)
    h = max(8, (h // 8) * 8)
    return w, h


class _DummyReconstructor:
    """Offline CPU-only deterministic reconstructor for unit testing."""

    revision = "dummy_offline_test_v1"

    def reconstruct_batch(self, images: List[Image.Image], seed: int = 42) -> List[Image.Image]:
        out = []
        for img in images:
            rgb = img.convert("RGB")
            # Mild deterministic blur + subtle quantization to guarantee distinct SHA256 bytes
            blurred = rgb.filter(ImageFilter.GaussianBlur(radius=0.6))
            arr = np.array(blurred, dtype=np.int16)
            arr = np.clip((arr // 2) * 2 + (seed % 2), 0, 255).astype(np.uint8)
            out.append(Image.fromarray(arr, mode="RGB"))
        return out


class _VAEReconstructor:
    """Fast batched Hugging Face Diffusers AutoencoderKL encode -> decode reconstructor."""

    def __init__(self, model_id: str, device: str = "cuda", dtype: str = "fp16", work_size: int = 256):
        import torch
        from diffusers import AutoencoderKL

        self.model_id = model_id
        self.revision = model_id
        self.device = torch.device(device)
        self.torch_dtype = torch.float16 if (dtype == "fp16" and self.device.type == "cuda") else (
            torch.bfloat16 if (dtype == "bf16" and self.device.type == "cuda") else torch.float32
        )
        self.work_size = int(work_size)
        self.vae = AutoencoderKL.from_pretrained(model_id, torch_dtype=self.torch_dtype).to(self.device)
        self.vae.eval()
        self.vae.requires_grad_(False)

    def reconstruct_batch(self, images: List[Image.Image], seed: int = 42) -> List[Image.Image]:
        import torch

        orig_sizes = [img.size for img in images]
        target = (self.work_size, self.work_size)
        tensors = []
        for img in images:
            resized = img.convert("RGB").resize(target, Image.Resampling.BICUBIC)
            arr = np.asarray(resized, dtype=np.float32) / 127.5 - 1.0  # [-1, 1]
            tensors.append(torch.from_numpy(arr).permute(2, 0, 1))

        batch = torch.stack(tensors, dim=0).to(device=self.device, dtype=self.torch_dtype)
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad():
            posterior = self.vae.encode(batch).latent_dist
            latents = posterior.sample(generator=gen)
            decoded = self.vae.decode(latents).sample

        decoded = ((decoded.float().clamp(-1.0, 1.0) + 1.0) * 127.5).round().to(torch.uint8)
        decoded_np = decoded.permute(0, 2, 3, 1).cpu().numpy()

        results = []
        for arr, orig_sz in zip(decoded_np, orig_sizes):
            rec_img = Image.fromarray(arr, mode="RGB")
            if rec_img.size != orig_sz:
                rec_img = rec_img.resize(orig_sz, Image.Resampling.BICUBIC)
            results.append(rec_img)
        return results


class _SDTurboReconstructor:
    """Fast 2-step SD-Turbo img2img diffusion + VAE reconstructor."""

    def __init__(
        self,
        model_id: str = "stabilityai/sd-turbo",
        device: str = "cuda",
        dtype: str = "fp16",
        work_size: int = 512,
        num_steps: int = 2,
        strength: float = 0.5,
    ):
        import torch
        from diffusers import AutoPipelineForImage2Image

        self.model_id = model_id
        self.revision = f"{model_id}:steps{num_steps}:str{strength}"
        self.device = torch.device(device)
        self.torch_dtype = torch.float16 if (dtype == "fp16" and self.device.type == "cuda") else (
            torch.bfloat16 if (dtype == "bf16" and self.device.type == "cuda") else torch.float32
        )
        self.work_size = int(work_size)
        self.num_steps = int(num_steps)
        self.strength = float(strength)
        self.pipe = AutoPipelineForImage2Image.from_pretrained(
            model_id,
            torch_dtype=self.torch_dtype,
            variant="fp16" if self.torch_dtype == torch.float16 else None,
        ).to(self.device)
        self.pipe.set_progress_bar_config(disable=True)

    def reconstruct_batch(self, images: List[Image.Image], seed: int = 42) -> List[Image.Image]:
        import torch

        orig_sizes = [img.size for img in images]
        inputs = [img.convert("RGB").resize((self.work_size, self.work_size), Image.Resampling.BICUBIC) for img in images]
        gen = torch.Generator(device=self.device).manual_seed(int(seed))
        with torch.no_grad():
            out_imgs = self.pipe(
                prompt=[""] * len(inputs),
                image=inputs,
                num_inference_steps=self.num_steps,
                strength=self.strength,
                guidance_scale=0.0,
                generator=gen,
            ).images

        results = []
        for rec_img, orig_sz in zip(out_imgs, orig_sizes):
            if rec_img.size != orig_sz:
                rec_img = rec_img.resize(orig_sz, Image.Resampling.BICUBIC)
            results.append(rec_img)
        return results


def _build_reconstructors(backend: str, device: str, dtype: str, work_size: int, steps: int, strength: float):
    if backend == "dummy":
        return {"dummy": _DummyReconstructor()}
    if backend == "sd15_vae":
        return {"sd15_vae": _VAEReconstructor("stabilityai/sd-vae-ft-mse", device=device, dtype=dtype, work_size=work_size)}
    if backend == "sdxl_vae":
        return {"sdxl_vae": _VAEReconstructor("madebyollin/sdxl-vae-fp16-fix", device=device, dtype=dtype, work_size=work_size)}
    if backend == "multi_vae":
        return {
            "sd15_vae": _VAEReconstructor("stabilityai/sd-vae-ft-mse", device=device, dtype=dtype, work_size=work_size),
            "sdxl_vae": _VAEReconstructor("madebyollin/sdxl-vae-fp16-fix", device=device, dtype=dtype, work_size=work_size),
        }
    if backend == "sd_turbo":
        return {
            "sd_turbo": _SDTurboReconstructor(
                model_id="stabilityai/sd-turbo",
                device=device,
                dtype=dtype,
                work_size=max(256, work_size),
                num_steps=steps,
                strength=strength,
            )
        }
    raise ValueError(f"Unsupported backend: {backend!r}")


def _assign_engine(sample_id: str, engine_names: List[str]) -> str:
    if len(engine_names) == 1:
        return engine_names[0]
    digest = int(hashlib.sha256(sample_id.encode("utf-8")).hexdigest()[:8], 16)
    return engine_names[digest % len(engine_names)]


def generate_reconstructions(
    inventory_path: Path,
    output_dir: Path,
    mapping_csv: Path,
    backend: str = "multi_vae",
    target_split: str = "train",
    reconstruct_fakes: bool = True,
    max_per_class: int = 0,
    batch_size: int = 32,
    work_size: int = 256,
    steps: int = 2,
    strength: float = 0.5,
    seed: int = 42,
    save_format: str = "match_parent",
    jpg_quality: int = 95,
    device: str = "cuda",
    dtype: str = "fp16",
    import_manifest: Optional[Path] = None,
) -> Dict[str, Any]:
    """Generate paired reconstructions and emit a verified ``reconstruction_mapping.csv``."""
    records = read_drct_manifest(inventory_path)
    selected_real = [r for r in records if r["split"] == target_split and r["sample_kind"] == "real_original"]
    selected_fake = [r for r in records if r["split"] == target_split and r["sample_kind"] == "fake_original"] if reconstruct_fakes else []

    if max_per_class > 0:
        selected_real = selected_real[:max_per_class]
        selected_fake = selected_fake[:max_per_class]

    targets = selected_real + selected_fake
    if not targets:
        raise ValueError(f"No original samples found for split={target_split!r} in {inventory_path}.")

    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    mapping_csv = Path(mapping_csv).resolve()
    mapping_csv.parent.mkdir(parents=True, exist_ok=True)

    # Load any existing valid mapping entries for resumability
    existing_map: Dict[str, Dict[str, str]] = {}
    if mapping_csv.is_file():
        with open(mapping_csv, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                pid = row.get("parent_id")
                rel_p = row.get("path")
                if pid and rel_p:
                    full_p = Path(rel_p) if Path(rel_p).is_absolute() else (output_dir / rel_p)
                    if full_p.is_file() and full_p.stat().st_size > 0:
                        existing_map[pid] = row

    reconstructors = _build_reconstructors(backend, device=device, dtype=dtype, work_size=work_size, steps=steps, strength=strength)
    engine_names = sorted(reconstructors.keys())

    # Group pending targets by engine so each batch runs cleanly on one model
    pending_by_engine: Dict[str, List[Dict[str, Any]]] = {name: [] for name in engine_names}
    completed_rows: Dict[str, Dict[str, str]] = {}

    for rec in targets:
        pid = rec["sample_id"]
        if pid in existing_map and existing_map[pid].get("parent_sha256") == rec.get("sha256"):
            completed_rows[pid] = existing_map[pid]
        else:
            eng = _assign_engine(pid, engine_names)
            pending_by_engine[eng].append(rec)

    skipped = len(completed_rows)
    total_pending = sum(len(v) for v in pending_by_engine.values())
    print(f"[Reconstruct] Targets={len(targets):,} (real={len(selected_real):,}, fake={len(selected_fake):,}) | Resumed={skipped:,} | Pending={total_pending:,}", flush=True)

    generated = 0
    for eng_name, items in pending_by_engine.items():
        if not items:
            continue
        engine = reconstructors[eng_name]
        print(f"[Reconstruct] Running engine '{eng_name}' ({engine.revision}) on {len(items):,} images...", flush=True)
        for start in range(0, len(items), batch_size):
            chunk = items[start:start + batch_size]
            pil_batch = []
            valid_chunk = []
            for rec in chunk:
                src_path = Path(rec["path"])
                with Image.open(src_path) as im:
                    pil_batch.append(im.convert("RGB").copy())
                valid_chunk.append(rec)

            batch_seed = int(seed) + start
            rec_images = engine.reconstruct_batch(pil_batch, seed=batch_seed)

            for rec, rec_img in zip(valid_chunk, rec_images):
                pid = rec["sample_id"]
                src_path = Path(rec["path"])
                parent_sha = rec.get("sha256") or file_sha256(src_path)
                if save_format == "match_parent":
                    fmt = "png" if src_path.suffix.lower() == ".png" else "jpg"
                else:
                    fmt = save_format.lower()
                ext = ".png" if fmt == "png" else ".jpg"
                rel_out = Path(rec["sample_kind"]) / eng_name / f"{pid}_rec{ext}"
                abs_out = output_dir / rel_out

                _atomic_save_image(rec_img, abs_out, fmt=fmt, jpg_quality=jpg_quality)
                rec_sha = file_sha256(abs_out)
                settings_obj = {
                    "backend": eng_name,
                    "work_size": work_size,
                    "save_format": fmt,
                    "jpg_quality": jpg_quality if fmt == "jpg" else None,
                    "steps": steps if eng_name == "sd_turbo" else 1,
                    "strength": strength if eng_name == "sd_turbo" else None,
                }
                completed_rows[pid] = {
                    "parent_id": pid,
                    "path": rel_out.as_posix(),
                    "parent_sha256": parent_sha,
                    "sha256": rec_sha,
                    "rec_model_revision": engine.revision,
                    "rec_settings": json.dumps(settings_obj, sort_keys=True),
                    "rec_seed": str(batch_seed),
                }
                generated += 1

            if generated % max(batch_size * 10, 200) < batch_size or (start + batch_size) >= len(items):
                _write_mapping_atomic(mapping_csv, [completed_rows[r["sample_id"]] for r in targets if r["sample_id"] in completed_rows])
                print(f"  [{eng_name}] {min(start + batch_size, len(items)):,} / {len(items):,} complete", flush=True)

    ordered_rows = [completed_rows[r["sample_id"]] for r in targets if r["sample_id"] in completed_rows]
    _write_mapping_atomic(mapping_csv, ordered_rows)
    print(f"[Reconstruct] Saved verified mapping CSV ({len(ordered_rows):,} pairs) -> {mapping_csv}", flush=True)

    summary: Dict[str, Any] = {
        "status": "success",
        "backend": backend,
        "total_targets": len(targets),
        "resumed": skipped,
        "generated": generated,
        "mapping_csv": str(mapping_csv),
        "output_dir": str(output_dir),
    }

    if import_manifest is not None:
        from data.drct_preparation import import_precomputed_reconstructions, check_drct_readiness
        import_res = import_precomputed_reconstructions(
            inventory_manifest_path=inventory_path,
            reconstruction_dir=output_dir,
            output_manifest_path=import_manifest,
            target_split=target_split,
            reconstruct_fakes=reconstruct_fakes,
            generate_gate=True,
            mapping_csv=mapping_csv,
        )
        readiness = check_drct_readiness(import_manifest, split=target_split)
        summary["imported_manifest"] = str(import_manifest)
        summary["import_summary"] = import_res
        summary["readiness"] = readiness
        print(f"[Reconstruct] Imported DRCT manifest -> {import_manifest} | ready={readiness.get('is_ready_for_full_drct')}", flush=True)

    return summary


def _write_mapping_atomic(mapping_csv: Path, rows: List[Dict[str, str]]) -> None:
    tmp = mapping_csv.with_suffix(".csv.tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MAPPING_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, mapping_csv)


def parse_args():
    p = argparse.ArgumentParser(description="Generate paired diffusion/VAE reconstructions for DRCT & EFFORT training.")
    p.add_argument("--inventory", required=True, type=Path, help="Path to base_inventory.csv")
    p.add_argument("--output_dir", required=True, type=Path, help="Directory to write reconstructed images")
    p.add_argument("--mapping_csv", required=True, type=Path, help="Output reconstruction_mapping.csv path")
    p.add_argument("--backend", choices=["sd15_vae", "sdxl_vae", "multi_vae", "sd_turbo", "dummy"], default="multi_vae")
    p.add_argument("--target_split", default="train")
    p.add_argument("--reconstruct_fakes", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max_per_class", type=int, default=0, help="Optional cap per class (0 = all train images)")
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--work_size", type=int, default=256, help="Working resolution for VAE/diffusion (256 or 512)")
    p.add_argument("--steps", type=int, default=2, help="Diffusion steps for sd_turbo")
    p.add_argument("--strength", type=float, default=0.5, help="Denoising strength for sd_turbo")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_format", choices=["match_parent", "jpg", "png"], default="match_parent")
    p.add_argument("--jpg_quality", type=int, default=95)
    p.add_argument("--device", default="cuda")
    p.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="fp16")
    p.add_argument("--import_manifest", type=Path, default=None, help="Optional output path to immediately import & gate DRCT manifest")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    res = generate_reconstructions(
        inventory_path=args.inventory,
        output_dir=args.output_dir,
        mapping_csv=args.mapping_csv,
        backend=args.backend,
        target_split=args.target_split,
        reconstruct_fakes=args.reconstruct_fakes,
        max_per_class=args.max_per_class,
        batch_size=args.batch_size,
        work_size=args.work_size,
        steps=args.steps,
        strength=args.strength,
        seed=args.seed,
        save_format=args.save_format,
        jpg_quality=args.jpg_quality,
        device=args.device,
        dtype=args.dtype,
        import_manifest=args.import_manifest,
    )
    print(json.dumps(res, indent=2))
