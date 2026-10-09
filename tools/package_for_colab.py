"""Package current code and verified manifests for Google Colab execution.

Creates a clean deployment archive or directory sync excluding local virtual environments,
git history, large diagnostic outputs, and old checkpoints.
Generates an audited package inventory with SHA-256 digests.
"""

import argparse
import hashlib
import json
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import List, Dict, Any

REPO_ROOT = Path(__file__).resolve().parent.parent

EXCLUDE_DIRS = {
    ".git", ".venv", "venv", "env", "__pycache__", ".pytest_cache",
    ".idea", ".vscode", "tmp", "scratch"
}

EXCLUDE_EXTS = {
    ".pyc", ".pyo", ".pyd", ".log", ".tmp"
}

# Only include relevant directories and files
INCLUDE_PATHS = [
    "train.py",
    "train_pipeline.py",
    "prepare_dataset.py",
    "analyze_predictions.py",
    "models",
    "data",
    "training",
    "experiment",
    "evaluation",
    "config",
    "protocols",
    "tools/benchmark_training_throughput.py",
    "tools/evaluate_mixture_predictions.py",
    "tools/build_controlled_mixture_manifests.py",
    "output/controlled_mixture/manifests_v2",
    "COLAB_RGB_MIXTURE_PILOT.md",
]


def digest_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_files(root: Path) -> List[Path]:
    files = []
    for item in INCLUDE_PATHS:
        src = root / item
        if not src.exists():
            continue
        if src.is_file():
            files.append(src)
        elif src.is_dir():
            for p in src.rglob("*"):
                if p.is_file():
                    if any(part in EXCLUDE_DIRS for part in p.parts):
                        continue
                    if p.suffix.lower() in EXCLUDE_EXTS:
                        continue
                    files.append(p)
    return sorted(files)


def build_package(output_path: Path, repo_root: Path = REPO_ROOT) -> Dict[str, Any]:
    files = collect_files(repo_root)
    out_p = Path(output_path).resolve()
    out_p.parent.mkdir(parents=True, exist_ok=True)

    inventory = []
    if out_p.suffix.lower() == ".zip":
        with zipfile.ZipFile(out_p, "w", zipfile.ZIP_DEFLATED) as z:
            for f in files:
                rel = f.relative_to(repo_root).as_posix()
                sha = digest_file(f)
                size = f.stat().st_size
                z.write(f, rel)
                inventory.append({"path": rel, "sha256": sha, "size_bytes": size})
    else:
        # Directory copy
        out_p.mkdir(parents=True, exist_ok=True)
        for f in files:
            rel = f.relative_to(repo_root)
            dest = out_p / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)
            sha = digest_file(f)
            inventory.append({"path": rel.as_posix(), "sha256": sha, "size_bytes": f.stat().st_size})

    manifest_data = {
        "package_path": str(out_p),
        "total_files": len(inventory),
        "inventory": inventory,
    }

    inv_path = out_p.parent / f"{out_p.stem}_inventory.json"
    with open(inv_path, "w", encoding="utf-8") as f:
        json.dump(manifest_data, f, indent=2)

    print(f"Packaged {len(inventory)} files to {out_p}")
    print(f"Inventory saved to {inv_path}")
    return manifest_data


def main():
    p = argparse.ArgumentParser(description="Package repository for Colab pilot execution")
    p.add_argument("--output", default="output/controlled_mixture/pilot_colab_package.zip",
                   help="Path to output zip or destination folder")
    args = p.parse_args()
    build_package(Path(args.output))


if __name__ == "__main__":
    main()
