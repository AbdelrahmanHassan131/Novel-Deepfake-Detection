"""Fresh-training multi-stage pipeline without historical checkpoint dependencies.

Orchestrates the three-stage protocol:
  Stage 1: Standalone RGB Expert (Wang2020_128) trained on the training split.
  Stage 2: Standalone Wavelet Expert (WolterWavelet2021_128) trained on the training split.
  Stage 3: Controlled Fusion Head (MHA_128 / Fusion_128) trained with frozen experts.

Lazy and non-intrusive: no models are created or checkpoints read on module import.
Supports plan mode (dry-run command generation & manifest creation) and runtime execution.
"""
from dataclasses import dataclass, field, asdict
import hashlib
import json
import os
from pathlib import Path
import sys
from typing import Dict, List, Optional, Tuple, Any


def file_sha256(path: Path) -> str:
    """Compute sha256 checksum of a file without external dependencies."""
    digest = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


FUSION_TYPE_TO_ARCH = {
    'concat': 'Fusion_128',
    'gated': 'Fusion_128',
    'token_attention': 'MHA_128',
}

VALID_FUSION_COMBINATIONS = {
    ('Fusion_128', 'concat'): 'ConcatenationFusionClassifier',
    ('Fusion_128', 'gated'): 'GatedFusion',
    ('MHA_128', 'token_attention'): 'TokenAttentionFusion',
    ('MHA_128', 'gated'): 'GatedFusion',
}


def inspect_expert_checkpoint(
    checkpoint_path: str,
    expected_type: str,
    expected_embed_dim: int = 128,
    opt: Optional[Any] = None,
) -> Dict[str, Any]:
    """Inspect and validate an expert checkpoint file statically.

    Ensures:
      1. Checkpoint file exists and is readable.
      2. Contains model weights with matching embedding dimension.
      3. Protocol metadata matches expected architecture, label mapping, and preprocessing.
      4. Does NOT instantiate full models or download anything.
    """
    path = Path(checkpoint_path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Expert checkpoint not found: {path}")

    import torch
    from models.shared.expert_loading import validate_expert_checkpoint

    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    validate_expert_checkpoint(
        checkpoint,
        expected_type=expected_type,
        expected_embed_dim=expected_embed_dim,
        opt=opt,
    )

    protocol = checkpoint.get('protocol', {})
    checksum = file_sha256(path)

    return {
        'path': str(path),
        'sha256': checksum,
        'expected_type': expected_type,
        'embed_dim': expected_embed_dim,
        'has_protocol_metadata': bool(protocol),
        'options': protocol.get('options', {}),
        'label_mapping': protocol.get('label_mapping', {'real': 0, 'fake': 1}),
        'verified': True,
    }



@dataclass
class StageSpec:
    name: str
    stage_num: int
    arch: str
    output_dir: str
    status: str = 'pending'
    command: List[str] = field(default_factory=list)
    checkpoint_path: Optional[str] = None
    checkpoint_sha256: Optional[str] = None
    prerequisites: List[str] = field(default_factory=list)


class FreshTrainingPipeline:
    """Manages multi-stage fresh training from split manifests to fusion evaluation."""

    def __init__(
        self,
        experiment_dir: str,
        dataroot: str,
        manifest: str,
        val_manifest: Optional[str] = None,
        val_root: Optional[str] = None,
        manifest_split: str = 'train',
        val_manifest_split: str = 'dev',
        embed_dim: int = 128,
        fusion_type: str = 'token_attention',
        rgb_arch: str = 'Wang2020_128',
        wavelet_arch: str = 'WolterWavelet2021_128',
        fusion_arch: Optional[str] = None,
        batch_size: int = 32,
        lr: float = 0.0001,
        epochs: int = 10,
        gpu_ids: str = '0',
        seed: int = 42,
        pretrained: bool = True,
        backbone_weights: Optional[str] = None,
        wavelet_log_mode: str = 'signed_log1p',
        use_amp: bool = False,
        grad_accum_steps: int = 1,
        run_id: Optional[str] = None,
        launcher: Optional[Any] = None,
        python_exe: str = sys.executable,
    ):
        self.experiment_dir = Path(experiment_dir).resolve()
        self.dataroot = Path(dataroot).resolve()
        self.manifest = Path(manifest).resolve()
        self.val_manifest = Path(val_manifest).resolve() if val_manifest else self.manifest
        self.val_root = Path(val_root).resolve() if val_root else self.dataroot
        self.manifest_split = manifest_split
        self.val_manifest_split = val_manifest_split
        self.embed_dim = embed_dim
        self.fusion_type = fusion_type
        self.rgb_arch = rgb_arch
        self.wavelet_arch = wavelet_arch
        self.fusion_arch = fusion_arch or FUSION_TYPE_TO_ARCH.get(fusion_type, 'MHA_128')

        # Validate fusion strategy combination
        if (self.fusion_arch, self.fusion_type) not in VALID_FUSION_COMBINATIONS:
            raise ValueError(
                f"Unsupported fusion configuration: arch='{self.fusion_arch}' with fusion_type='{self.fusion_type}'. "
                f"Valid combinations: {sorted(list(VALID_FUSION_COMBINATIONS.keys()))}"
            )
        self.head_class_name = VALID_FUSION_COMBINATIONS[(self.fusion_arch, self.fusion_type)]

        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.gpu_ids = gpu_ids
        self.seed = seed
        self.pretrained = pretrained
        self.backbone_weights = backbone_weights
        self.wavelet_log_mode = wavelet_log_mode
        self.use_amp = use_amp
        self.grad_accum_steps = grad_accum_steps
        self.run_id = run_id or f"seed{self.seed}"
        self.launcher = launcher
        self.python_exe = python_exe

        # Setup stage specs matching ExperimentManager directory layout:
        # <checkpoints_dir>/<name>_<run_id>/checkpoints/best.pth
        self.stage1_base_dir = self.experiment_dir / "stage1_rgb"
        self.stage1_exp_dir = self.stage1_base_dir / f"stage1_rgb_{self.run_id}"
        self.stage1_ckpt = self.stage1_exp_dir / "checkpoints" / "best.pth"

        self.stage2_base_dir = self.experiment_dir / "stage2_wavelet"
        self.stage2_exp_dir = self.stage2_base_dir / f"stage2_wavelet_{self.run_id}"
        self.stage2_ckpt = self.stage2_exp_dir / "checkpoints" / "best.pth"

        self.stage3_base_dir = self.experiment_dir / f"stage3_fusion_{self.fusion_type}"
        self.stage3_exp_dir = self.stage3_base_dir / f"stage3_fusion_{self.fusion_type}_{self.run_id}"
        self.stage3_ckpt = self.stage3_exp_dir / "checkpoints" / "best.pth"

        self.stages = {
            'rgb': StageSpec(
                name='rgb_expert',
                stage_num=1,
                arch=self.rgb_arch,
                output_dir=str(self.stage1_exp_dir),
                checkpoint_path=str(self.stage1_ckpt),
                prerequisites=[],
            ),
            'wavelet': StageSpec(
                name='wavelet_expert',
                stage_num=2,
                arch=self.wavelet_arch,
                output_dir=str(self.stage2_exp_dir),
                checkpoint_path=str(self.stage2_ckpt),
                prerequisites=[],
            ),
            'fusion': StageSpec(
                name=f'fusion_{self.fusion_type}',
                stage_num=3,
                arch=self.fusion_arch,
                output_dir=str(self.stage3_exp_dir),
                checkpoint_path=str(self.stage3_ckpt),
                prerequisites=['rgb', 'wavelet'],
            ),
        }
        self._build_commands()

    def _build_commands(self) -> None:
        """Construct deterministic CLI training commands for all stages."""
        common = [
            self.python_exe, "train.py",
            "--dataroot", str(self.dataroot),
            "--manifest", str(self.manifest),
            "--manifest_split", self.manifest_split,
            "--val_manifest", str(self.val_manifest),
            "--val_manifest_split", self.val_manifest_split,
            "--val_root", str(self.val_root),
            "--batch_size", str(self.batch_size),
            "--lr", str(self.lr),
            "--epochs", str(self.epochs),
            "--gpu_ids", str(self.gpu_ids),
            "--seed", str(self.seed),
            "--embed_dim", str(self.embed_dim),
            "--wavelet_log_mode", self.wavelet_log_mode,
        ]
        if self.use_amp:
            common.append("--use_amp")
        if self.grad_accum_steps > 1:
            common.extend(["--grad_accum_steps", str(self.grad_accum_steps)])

        # Stage 1: RGB
        cmd1 = list(common) + [
            "--arch", self.rgb_arch,
            "--name", "stage1_rgb",
            "--run_id", self.run_id,
            "--checkpoints_dir", str(self.stage1_base_dir),
        ]
        if self.pretrained:
            cmd1.append("--pretrained")
        else:
            cmd1.append("--no-pretrained")
        if self.backbone_weights:
            cmd1.extend(["--backbone_weights", str(self.backbone_weights)])
        self.stages['rgb'].command = cmd1

        # Stage 2: Wavelet
        cmd2 = list(common) + [
            "--arch", self.wavelet_arch,
            "--name", "stage2_wavelet",
            "--run_id", self.run_id,
            "--checkpoints_dir", str(self.stage2_base_dir),
            "--compute_wavelets",
        ]
        self.stages['wavelet'].command = cmd2

        # Stage 3: Fusion
        rgb_ckpt = self.stages['rgb'].checkpoint_path
        wav_ckpt = self.stages['wavelet'].checkpoint_path
        cmd3 = list(common) + [
            "--arch", self.fusion_arch,
            "--name", f"stage3_fusion_{self.fusion_type}",
            "--run_id", self.run_id,
            "--checkpoints_dir", str(self.stage3_base_dir),
            "--fusion_type", self.fusion_type,
            "--freeze_base_models",
            "--rgb_model_path", rgb_ckpt,
            "--wavelet_model_path", wav_ckpt,
        ]
        self.stages['fusion'].command = cmd3

    def generate_manifest(self, resume_checkpoint: Optional[str] = None) -> Dict[str, Any]:
        """Generate the complete pipeline manifest dictionary."""
        manifest_dict = {
            'pipeline_version': 2,
            'protocol': 'fresh_training_3stage_frozen_fusion',
            'created_at': '2026-09-29',
            'run_id': self.run_id,
            'head_class_name': self.head_class_name,
            'parameters': {
                'dataroot': str(self.dataroot),
                'manifest': str(self.manifest),
                'manifest_split': self.manifest_split,
                'val_manifest': str(self.val_manifest),
                'val_manifest_split': self.val_manifest_split,
                'embed_dim': self.embed_dim,
                'fusion_type': self.fusion_type,
                'rgb_arch': self.rgb_arch,
                'wavelet_arch': self.wavelet_arch,
                'fusion_arch': self.fusion_arch,
                'batch_size': self.batch_size,
                'lr': self.lr,
                'epochs': self.epochs,
                'seed': self.seed,
                'gpu_ids': self.gpu_ids,
                'wavelet_log_mode': self.wavelet_log_mode,
            },
            'stages': {k: asdict(v) for k, v in self.stages.items()},
        }
        if resume_checkpoint:
            manifest_dict['resume'] = {
                'target_checkpoint': resume_checkpoint,
                'resumed_at': '2026-09-29',
            }
        return manifest_dict

    def save_manifest(self, path: Optional[Path] = None) -> Path:
        """Write pipeline_manifest.json to experiment directory, preserving completed stages if existing."""
        self.experiment_dir.mkdir(parents=True, exist_ok=True)
        out_path = path or (self.experiment_dir / "pipeline_manifest.json")
        data = self.generate_manifest()
        # If preserving an existing manifest and not writing to a separate plan file:
        if path is None and out_path.is_file():
            try:
                existing_data = json.loads(out_path.read_text(encoding='utf-8'))
                existing_stages = existing_data.get('stages', {})
                for k, s_dict in existing_stages.items():
                    if k in data['stages'] and s_dict.get('status') == 'completed' and data['stages'][k]['status'] != 'completed':
                        data['stages'][k]['status'] = s_dict['status']
                        data['stages'][k]['checkpoint_sha256'] = s_dict.get('checkpoint_sha256')
            except Exception:
                pass
        out_path.write_text(json.dumps(data, indent=2), encoding='utf-8')
        return out_path

    def validate_fusion_prerequisites(self) -> Dict[str, Any]:
        """Validate that Stages 1 and 2 produced valid, compatible expert checkpoints."""
        rgb_path = self.stages['rgb'].checkpoint_path
        wav_path = self.stages['wavelet'].checkpoint_path

        if not rgb_path or not Path(rgb_path).is_file():
            raise FileNotFoundError(
                f"Cannot launch Fusion stage: RGB expert checkpoint missing at {rgb_path}. "
                f"Complete Stage 1 training first."
            )
        if not wav_path or not Path(wav_path).is_file():
            raise FileNotFoundError(
                f"Cannot launch Fusion stage: Wavelet expert checkpoint missing at {wav_path}. "
                f"Complete Stage 2 training first."
            )

        rgb_info = inspect_expert_checkpoint(rgb_path, expected_type='rgb', expected_embed_dim=self.embed_dim, opt=self)
        wav_info = inspect_expert_checkpoint(wav_path, expected_type='wavelet', expected_embed_dim=self.embed_dim, opt=self)

        return {
            'rgb_expert': rgb_info,
            'wavelet_expert': wav_info,
            'compatible': True,
        }

    def build_resume_command(self, stage: str, checkpoint_path: str) -> List[str]:
        """Build command to resume an interrupted stage from a specific checkpoint."""
        if stage not in self.stages:
            raise ValueError(f"Unknown stage: {stage}")
        base_cmd = [arg for arg in self.stages[stage].command if arg not in ('--continue_train', '--resume_checkpoint')]
        base_cmd.extend(["--continue_train", "--resume_checkpoint", str(Path(checkpoint_path).resolve())])
        return base_cmd

    def run_stage(self, stage: str, resume_checkpoint: Optional[str] = None) -> int:
        """Execute an individual pipeline stage with exit-code propagation and artifact registration."""
        if stage not in self.stages:
            raise ValueError(f"Unknown stage '{stage}'. Must be one of: {list(self.stages.keys())}")

        spec = self.stages[stage]

        # Verify prerequisites for fusion stage
        if stage == 'fusion':
            self.validate_fusion_prerequisites()

        from data.manifest import verify_manifest_gate
        verify_manifest_gate(self.manifest)

        if resume_checkpoint:
            cmd = self.build_resume_command(stage, resume_checkpoint)
        else:
            cmd = list(spec.command)

        print(f"\n[Executing Stage {spec.stage_num}: {spec.name}]")
        print(f"Command: {' '.join(cmd)}")

        spec.status = 'running'
        self.save_manifest()

        ckpt_path = Path(spec.checkpoint_path)
        mtime_before = ckpt_path.stat().st_mtime if ckpt_path.is_file() else None

        import subprocess
        launcher = self.launcher or (lambda c: subprocess.run(c, check=False))
        res = launcher(cmd)
        returncode = res.returncode if hasattr(res, 'returncode') else (res if isinstance(res, int) else 0)

        if returncode != 0:
            spec.status = 'failed'
            self.save_manifest()
            raise RuntimeError(f"Stage '{stage}' failed with exit code {returncode}")

        # Ensure expected checkpoint artifact was produced
        if not ckpt_path.is_file():
            spec.status = 'failed'
            self.save_manifest()
            raise FileNotFoundError(
                f"Stage '{stage}' exited with code 0 but expected artifact was not produced at {ckpt_path}."
            )

        # Reject stale checkpoints from prior attempts if not resuming
        mtime_after = ckpt_path.stat().st_mtime
        if mtime_before is not None and mtime_after == mtime_before and not resume_checkpoint:
            spec.status = 'failed'
            self.save_manifest()
            raise RuntimeError(
                f"Stage '{stage}' artifact at {ckpt_path} was not updated by this run attempt (stale checkpoint detected)."
            )

        spec.status = 'completed'
        spec.checkpoint_sha256 = file_sha256(ckpt_path)
        self.save_manifest()
        return returncode

    def run_stages(self, stages: List[str], resume_checkpoint: Optional[str] = None) -> Dict[str, int]:
        """Execute a sequence of pipeline stages in order."""
        results = {}
        for s in stages:
            rc = self.run_stage(s, resume_checkpoint=resume_checkpoint if s == stages[0] else None)
            results[s] = rc
        return results

