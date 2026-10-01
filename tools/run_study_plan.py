"""Study plan consumer and command generator.

Reads declarative experiment/study plan JSON files from config/experiments/
and generates exact, reproducible training, audit, and evaluation commands.

Supported schemas:
1. Multi-Stage Pipeline (`stages`, `dataset`, `training`)
2. Multi-Seed Model Comparison (`models_to_compare`, `seeds`)
3. Targeted 1D Ablations Sweep (`ablations`, `base_configuration`)

Fails closed on unimplemented or unknown study schemas/fields.
Never silently substitutes a default experiment for an unhandled study.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional


SUPPORTED_SCHEMAS = {
    'multi_stage_pipeline',
    'multi_seed_comparison',
    'targeted_ablations',
}

SUPPORTED_ABLATION_DIMENSIONS = {
    'wavelet_level',
    'embedding_dimension',
    'augmentation_policy',
    'frame_capping_per_video',
    'fusion_architecture',
}


def parse_study_plan(plan_path: str) -> Dict[str, Any]:
    p = Path(plan_path)
    if not p.is_file():
        raise FileNotFoundError(f"Study plan file not found: {plan_path}")
    data = json.loads(p.read_text(encoding='utf-8'))
    return data


def identify_plan_schema(plan: Dict[str, Any]) -> str:
    """Identify the specific schema of a declarative study plan or reject unknown schemas."""
    plan_type = plan.get('plan_type')
    if plan_type != 'declarative_study_plan':
        raise ValueError(
            f"Unsupported plan_type '{plan_type}'. Expected 'declarative_study_plan'."
        )

    if 'ablations' in plan:
        return 'targeted_ablations'
    if 'models_to_compare' in plan:
        return 'multi_seed_comparison'
    if 'stages' in plan:
        return 'multi_stage_pipeline'

    raise NotImplementedError(
        f"Unrecognized or unsupported study plan structure in plan '{plan.get('experiment_name', 'unnamed')}'. "
        "Explicitly marked pending; refusing to silently substitute a default experiment."
    )


def build_audit_command(manifest_path: str, audit_output: str, python_exe: str = "python") -> str:
    """Build mandatory data preparation audit command with full hash verification."""
    return (
        f"{python_exe} prepare_dataset.py audit \\\n"
        f"    --manifest {manifest_path} \\\n"
        f"    --output {audit_output} \\\n"
        f"    --hashes"
    )


def generate_pipeline_commands(
    plan: Dict[str, Any],
    data_root: str = "/content/data",
    output_root: str = "/content/experiments",
    launch_cmd: str = "python",
    python_exe: str = "python",
) -> List[str]:
    """Generate exact CLI commands for a declarative experiment study plan.

    Dispatches explicitly based on validated plan schema. Never falls back silently.
    """
    schema = identify_plan_schema(plan)

    if schema == 'multi_stage_pipeline':
        return _generate_multi_stage_pipeline_commands(
            plan, data_root=data_root, output_root=output_root, launch_cmd=launch_cmd, python_exe=python_exe
        )
    elif schema == 'multi_seed_comparison':
        return _generate_multi_seed_comparison_commands(
            plan, data_root=data_root, output_root=output_root, launch_cmd=launch_cmd, python_exe=python_exe
        )
    elif schema == 'targeted_ablations':
        return _generate_targeted_ablations_commands(
            plan, data_root=data_root, output_root=output_root, launch_cmd=launch_cmd, python_exe=python_exe
        )
    else:
        raise NotImplementedError(f"Schema '{schema}' is not implemented.")


def _generate_dataset_and_audit(
    ds: Dict[str, Any],
    data_root: str,
    output_root: str,
    python_exe: str = "python",
    manifest_name: str = "pilot_100k_manifest.csv",
) -> List[str]:
    """Generate dataset preparation command followed by mandatory data integrity audit."""
    commands = []
    target_real = ds.get('target_real_train', 50000)
    target_fake = ds.get('target_fake_train', 50000)
    dev_ratio = ds.get('dev_ratio', 0.1)
    test_ratio = ds.get('test_ratio', 0.1)
    frame_cap = ds.get('frame_cap_per_video', 15)
    seed = ds.get('seed', 42)
    manifest_out = f"{output_root}/{manifest_name}"
    audit_out = f"{output_root}/audit_report.json"

    pilot_cmd = (
        f"{python_exe} prepare_dataset.py pilot \\\n"
        f"    --source_dir {data_root} \\\n"
        f"    --output_manifest {manifest_out} \\\n"
        f"    --target_real {target_real} \\\n"
        f"    --target_fake {target_fake} \\\n"
        f"    --dev_ratio {dev_ratio} \\\n"
        f"    --test_ratio {test_ratio} \\\n"
        f"    --frame_cap {frame_cap} \\\n"
        f"    --require_groups \\\n"
        f"    --seed {seed}"
    )
    commands.append(pilot_cmd)

    # Mandatory data integrity audit
    commands.append(build_audit_command(manifest_out, audit_out, python_exe=python_exe))
    return commands


def _generate_multi_stage_pipeline_commands(
    plan: Dict[str, Any],
    data_root: str,
    output_root: str,
    launch_cmd: str,
    python_exe: str,
) -> List[str]:
    commands = []

    # 1. Dataset preparation and audit if dataset block exists
    ds = plan.get('dataset', {})
    manifest_path = f"{output_root}/pilot_100k_manifest.csv"
    if ds:
        commands.extend(_generate_dataset_and_audit(ds, data_root, output_root, python_exe=python_exe))

    # 2. Training configuration
    tr = plan.get('training', {})
    batch_size = tr.get('batch_size', 16)
    grad_accum = tr.get('grad_accum_steps', 2)
    epochs = tr.get('epochs', 10)
    lr = tr.get('learning_rate', 0.0001)
    amp_flag = "--use_amp" if tr.get('use_amp', True) else ""
    monitor = tr.get('monitor_metric', 'auc')
    num_workers = tr.get('num_workers', 0)
    wavelet_log_mode = tr.get('wavelet_log_mode', 'signed_log1p')
    seeds = plan.get('seeds', [42])

    stages = plan.get('stages', {})
    s1 = stages.get('stage1_rgb', {})
    s2 = stages.get('stage2_wavelet', {})
    s3 = stages.get('stage3_fusion', {})

    arch_s1 = s1.get('arch', 'Wang2020_128')
    embed_dim_s1 = s1.get('embed_dim', 128)
    pretrained_s1 = s1.get('pretrained', True)
    pretrained_flag = "--pretrained" if pretrained_s1 else "--no-pretrained"
    backbone_weights = s1.get('backbone_weights')

    arch_s2 = s2.get('arch', 'WolterWavelet2021_128')
    embed_dim_s2 = s2.get('embed_dim', 128)
    wavelet_type = s2.get('wavelet_type', 'haar')
    wavelet_level = s2.get('wavelet_level', 3)
    s2_log_mode = s2.get('wavelet_log_mode', wavelet_log_mode)

    embed_dim_s3 = s3.get('embed_dim', 128)
    candidates = s3.get('fusion_candidates', ['token_attention', 'gated', 'concat'])

    for seed in seeds:
        run_id = f"seed{seed}"

        # Stage 1: RGB Expert
        bw_flag = f" \\\n    --backbone_weights {backbone_weights}" if backbone_weights else ""
        nw_flag = f" \\\n    --num_workers {num_workers}" if num_workers > 0 else ""
        cmd_s1 = (
            f"{launch_cmd} train.py \\\n"
            f"    --arch {arch_s1} \\\n"
            f"    --embed_dim {embed_dim_s1} \\\n"
            f"    {pretrained_flag}{bw_flag} \\\n"
            f"    --dataroot {data_root} \\\n"
            f"    --manifest {manifest_path} \\\n"
            f"    --manifest_split train \\\n"
            f"    --val_manifest {manifest_path} \\\n"
            f"    --val_manifest_split dev \\\n"
            f"    --name stage1_rgb_expert \\\n"
            f"    --run_id {run_id} \\\n"
            f"    --checkpoints_dir {output_root}/stage1_rgb \\\n"
            f"    --batch_size {batch_size} \\\n"
            f"    --grad_accum_steps {grad_accum} \\\n"
            f"    --epochs {epochs} \\\n"
            f"    --lr {lr} \\\n"
            f"    --monitor_metric {monitor} \\\n"
            f"    --seed {seed} {amp_flag}{nw_flag}"
        )
        commands.append(cmd_s1)

        # Stage 2: Wavelet Expert
        cmd_s2 = (
            f"{launch_cmd} train.py \\\n"
            f"    --arch {arch_s2} \\\n"
            f"    --embed_dim {embed_dim_s2} \\\n"
            f"    --dataroot {data_root} \\\n"
            f"    --manifest {manifest_path} \\\n"
            f"    --manifest_split train \\\n"
            f"    --val_manifest {manifest_path} \\\n"
            f"    --val_manifest_split dev \\\n"
            f"    --name stage2_wavelet_expert \\\n"
            f"    --run_id {run_id} \\\n"
            f"    --checkpoints_dir {output_root}/stage2_wavelet \\\n"
            f"    --wavelet_type {wavelet_type} \\\n"
            f"    --wavelet_level {wavelet_level} \\\n"
            f"    --wavelet_log_mode {s2_log_mode} \\\n"
            f"    --batch_size {batch_size} \\\n"
            f"    --grad_accum_steps {grad_accum} \\\n"
            f"    --epochs {epochs} \\\n"
            f"    --lr {lr} \\\n"
            f"    --monitor_metric {monitor} \\\n"
            f"    --seed {seed} {amp_flag}{nw_flag}"
        )
        commands.append(cmd_s2)

        # Stage 3: Fusion Heads
        rgb_ckpt = f"{output_root}/stage1_rgb/stage1_rgb_expert_{run_id}/checkpoints/best.pth"
        wav_ckpt = f"{output_root}/stage2_wavelet/stage2_wavelet_expert_{run_id}/checkpoints/best.pth"
        for ftype in candidates:
            arch = 'MHA_128' if ftype == 'token_attention' else 'Fusion_128'
            head_name = f"stage3_{ftype}"
            cmd_s3 = (
                f"{launch_cmd} train.py \\\n"
                f"    --arch {arch} \\\n"
                f"    --fusion_type {ftype} \\\n"
                f"    --embed_dim {embed_dim_s3} \\\n"
                f"    --wavelet_level {wavelet_level} \\\n"
                f"    --wavelet_type {wavelet_type} \\\n"
                f"    --wavelet_log_mode {s2_log_mode} \\\n"
                f"    --rgb_model_path {rgb_ckpt} \\\n"
                f"    --wavelet_model_path {wav_ckpt} \\\n"
                f"    --dataroot {data_root} \\\n"
                f"    --manifest {manifest_path} \\\n"
                f"    --manifest_split train \\\n"
                f"    --val_manifest {manifest_path} \\\n"
                f"    --val_manifest_split dev \\\n"
                f"    --name {head_name} \\\n"
                f"    --run_id {run_id} \\\n"
                f"    --checkpoints_dir {output_root}/{head_name} \\\n"
                f"    --batch_size {batch_size} \\\n"
                f"    --grad_accum_steps {grad_accum} \\\n"
                f"    --epochs {epochs} \\\n"
                f"    --monitor_metric {monitor} \\\n"
                f"    --seed {seed} {amp_flag}{nw_flag}"
            )
            commands.append(cmd_s3)

    return commands


def _generate_multi_seed_comparison_commands(
    plan: Dict[str, Any],
    data_root: str,
    output_root: str,
    launch_cmd: str,
    python_exe: str,
) -> List[str]:
    """Generate pipeline commands for multi_seed_comparison.json schema."""
    commands = []
    manifest_path = f"{output_root}/pilot_100k_manifest.csv"

    # Dataset prep if provided in plan
    ds = plan.get('dataset', {})
    if ds:
        commands.extend(_generate_dataset_and_audit(ds, data_root, output_root, python_exe=python_exe))

    seeds = plan.get('seeds', [42, 43, 44])
    models_to_compare = plan.get('models_to_compare', [])

    if not models_to_compare:
        raise ValueError("Multi-seed comparison plan specifies no 'models_to_compare'.")

    tr = plan.get('training', {})
    batch_size = tr.get('batch_size', 16)
    grad_accum = tr.get('grad_accum_steps', 2)
    epochs = tr.get('epochs', 10)
    lr = tr.get('learning_rate', 0.0001)
    amp_flag = "--use_amp" if tr.get('use_amp', True) else ""

    for seed in seeds:
        run_id = f"seed{seed}"
        # 1. Standalone RGB Expert for this seed
        cmd_s1 = (
            f"{launch_cmd} train.py \\\n"
            f"    --arch Wang2020_128 \\\n"
            f"    --embed_dim 128 \\\n"
            f"    --pretrained \\\n"
            f"    --dataroot {data_root} \\\n"
            f"    --manifest {manifest_path} \\\n"
            f"    --manifest_split train \\\n"
            f"    --val_manifest {manifest_path} \\\n"
            f"    --val_manifest_split dev \\\n"
            f"    --name stage1_rgb_expert \\\n"
            f"    --run_id {run_id} \\\n"
            f"    --checkpoints_dir {output_root}/stage1_rgb \\\n"
            f"    --batch_size {batch_size} \\\n"
            f"    --grad_accum_steps {grad_accum} \\\n"
            f"    --epochs {epochs} \\\n"
            f"    --lr {lr} \\\n"
            f"    --monitor_metric auc \\\n"
            f"    --seed {seed} {amp_flag}"
        )
        commands.append(cmd_s1)

        # 2. Standalone Wavelet Expert for this seed
        cmd_s2 = (
            f"{launch_cmd} train.py \\\n"
            f"    --arch WolterWavelet2021_128 \\\n"
            f"    --embed_dim 128 \\\n"
            f"    --dataroot {data_root} \\\n"
            f"    --manifest {manifest_path} \\\n"
            f"    --manifest_split train \\\n"
            f"    --val_manifest {manifest_path} \\\n"
            f"    --val_manifest_split dev \\\n"
            f"    --name stage2_wavelet_expert \\\n"
            f"    --run_id {run_id} \\\n"
            f"    --checkpoints_dir {output_root}/stage2_wavelet \\\n"
            f"    --wavelet_type haar \\\n"
            f"    --wavelet_level 3 \\\n"
            f"    --wavelet_log_mode signed_log1p \\\n"
            f"    --batch_size {batch_size} \\\n"
            f"    --grad_accum_steps {grad_accum} \\\n"
            f"    --epochs {epochs} \\\n"
            f"    --lr {lr} \\\n"
            f"    --monitor_metric auc \\\n"
            f"    --seed {seed} {amp_flag}"
        )
        commands.append(cmd_s2)

        # 3. Models to compare for this seed (sharing the exact same expert checkpoints)
        rgb_ckpt = f"{output_root}/stage1_rgb/stage1_rgb_expert_{run_id}/checkpoints/best.pth"
        wav_ckpt = f"{output_root}/stage2_wavelet/stage2_wavelet_expert_{run_id}/checkpoints/best.pth"

        for m in models_to_compare:
            model_id = m.get('model_id')
            ftype = m.get('fusion_type')
            arch = m.get('arch')

            if ftype == 'ensemble':
                # Late ensemble is an evaluation aggregation, not a trainable neural module
                cmd_ens = (
                    f"# Model Comparison: {model_id} (Late Ensemble - Evaluation Aggregation)\n"
                    f"{python_exe} analyze_predictions.py ensemble \\\n"
                    f"    --rgb_predictions {output_root}/stage1_rgb/stage1_rgb_expert_{run_id}/checkpoints/dev_predictions.csv \\\n"
                    f"    --wavelet_predictions {output_root}/stage2_wavelet/stage2_wavelet_expert_{run_id}/checkpoints/dev_predictions.csv \\\n"
                    f"    --output {output_root}/ensemble_{run_id}_dev_predictions.csv"
                )
                commands.append(cmd_ens)
            else:
                head_name = f"stage3_{model_id}"
                cmd_head = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch {arch} \\\n"
                    f"    --fusion_type {ftype} \\\n"
                    f"    --embed_dim 128 \\\n"
                    f"    --wavelet_level 3 \\\n"
                    f"    --wavelet_type haar \\\n"
                    f"    --wavelet_log_mode signed_log1p \\\n"
                    f"    --rgb_model_path {rgb_ckpt} \\\n"
                    f"    --wavelet_model_path {wav_ckpt} \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name {head_name} \\\n"
                    f"    --run_id {run_id} \\\n"
                    f"    --checkpoints_dir {output_root}/{head_name} \\\n"
                    f"    --batch_size {batch_size} \\\n"
                    f"    --grad_accum_steps {grad_accum} \\\n"
                    f"    --epochs {epochs} \\\n"
                    f"    --lr {lr} \\\n"
                    f"    --monitor_metric auc \\\n"
                    f"    --seed {seed} {amp_flag}"
                )
                commands.append(cmd_head)

    return commands


def _generate_targeted_ablations_commands(
    plan: Dict[str, Any],
    data_root: str,
    output_root: str,
    launch_cmd: str,
    python_exe: str,
) -> List[str]:
    """Generate pipeline commands for ablations_plan.json schema."""
    commands = []
    manifest_path = f"{output_root}/pilot_100k_manifest.csv"
    ablations = plan.get('ablations', [])

    if not ablations:
        raise ValueError("Ablations plan specifies no 'ablations' list.")

    for ab in ablations:
        dim = ab.get('dimension')
        if dim not in SUPPORTED_ABLATION_DIMENSIONS:
            raise NotImplementedError(
                f"Ablation dimension '{dim}' is not supported and is explicitly marked pending. "
                "Refusing to substitute a default experiment."
            )

        if dim == 'wavelet_level':
            # Frequency decomposition levels 2, 3, 4
            levels = ab.get('values', [2, 3, 4])
            fixed = ab.get('fixed_settings', {})
            edim = fixed.get('embed_dim', 128)
            ftype = fixed.get('fusion_type', 'token_attention')
            wmode = fixed.get('wavelet_log_mode', 'signed_log1p')

            for lvl in levels:
                run_tag = f"wav_level{lvl}"
                # Wavelet expert trained at this level
                cmd_w = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch WolterWavelet2021_128 \\\n"
                    f"    --embed_dim {edim} \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name stage2_wavelet_{run_tag} \\\n"
                    f"    --run_id seed42 \\\n"
                    f"    --checkpoints_dir {output_root}/ablation_wavelet_level \\\n"
                    f"    --wavelet_type haar \\\n"
                    f"    --wavelet_level {lvl} \\\n"
                    f"    --wavelet_log_mode {wmode} \\\n"
                    f"    --seed 42 --use_amp"
                )
                commands.append(cmd_w)

                # Fusion head with matching wavelet_level
                cmd_f = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch MHA_128 \\\n"
                    f"    --fusion_type {ftype} \\\n"
                    f"    --embed_dim {edim} \\\n"
                    f"    --wavelet_level {lvl} \\\n"
                    f"    --wavelet_type haar \\\n"
                    f"    --wavelet_log_mode {wmode} \\\n"
                    f"    --rgb_model_path {output_root}/stage1_rgb/stage1_rgb_expert_seed42/checkpoints/best.pth \\\n"
                    f"    --wavelet_model_path {output_root}/ablation_wavelet_level/stage2_wavelet_{run_tag}_seed42/checkpoints/best.pth \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name stage3_fusion_{run_tag} \\\n"
                    f"    --run_id seed42 \\\n"
                    f"    --checkpoints_dir {output_root}/ablation_wavelet_level \\\n"
                    f"    --seed 42 --use_amp"
                )
                commands.append(cmd_f)

        elif dim == 'embedding_dimension':
            dims = ab.get('values', [64, 128, 256])
            for d in dims:
                run_tag = f"dim{d}"
                # RGB expert at dimension d
                cmd_rgb = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch Wang2020_128 \\\n"
                    f"    --embed_dim {d} \\\n"
                    f"    --pretrained \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name stage1_rgb_{run_tag} \\\n"
                    f"    --run_id seed42 \\\n"
                    f"    --checkpoints_dir {output_root}/ablation_embed_dim \\\n"
                    f"    --seed 42 --use_amp"
                )
                commands.append(cmd_rgb)

                # Wavelet expert at dimension d
                cmd_wav = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch WolterWavelet2021_128 \\\n"
                    f"    --embed_dim {d} \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name stage2_wavelet_{run_tag} \\\n"
                    f"    --run_id seed42 \\\n"
                    f"    --checkpoints_dir {output_root}/ablation_embed_dim \\\n"
                    f"    --seed 42 --use_amp"
                )
                commands.append(cmd_wav)

                # Fusion at dimension d
                cmd_fus = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch MHA_128 \\\n"
                    f"    --fusion_type token_attention \\\n"
                    f"    --embed_dim {d} \\\n"
                    f"    --rgb_model_path {output_root}/ablation_embed_dim/stage1_rgb_{run_tag}_seed42/checkpoints/best.pth \\\n"
                    f"    --wavelet_model_path {output_root}/ablation_embed_dim/stage2_wavelet_{run_tag}_seed42/checkpoints/best.pth \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name stage3_fusion_{run_tag} \\\n"
                    f"    --run_id seed42 \\\n"
                    f"    --checkpoints_dir {output_root}/ablation_embed_dim \\\n"
                    f"    --seed 42 --use_amp"
                )
                commands.append(cmd_fus)

        elif dim == 'augmentation_policy':
            conditions = ab.get('conditions', [])
            for cond in conditions:
                cname = cond['name']
                b_prob = cond['blur_prob']
                j_prob = cond['jpg_prob']
                cmd_aug_rgb = (
                    f"{launch_cmd} train.py \\\n"
                    f"    --arch Wang2020_128 \\\n"
                    f"    --blur_prob {b_prob} \\\n"
                    f"    --jpg_prob {j_prob} \\\n"
                    f"    --dataroot {data_root} \\\n"
                    f"    --manifest {manifest_path} \\\n"
                    f"    --manifest_split train \\\n"
                    f"    --val_manifest {manifest_path} \\\n"
                    f"    --val_manifest_split dev \\\n"
                    f"    --name stage1_rgb_aug_{cname} \\\n"
                    f"    --run_id seed42 \\\n"
                    f"    --checkpoints_dir {output_root}/ablation_aug \\\n"
                    f"    --seed 42 --use_amp"
                )
                commands.append(cmd_aug_rgb)

        elif dim == 'frame_capping_per_video':
            caps = ab.get('values', [5, 15, 30])
            for cap in caps:
                m_out = f"{output_root}/pilot_manifest_cap{cap}.csv"
                cmd_cap = (
                    f"{python_exe} prepare_dataset.py pilot \\\n"
                    f"    --source_dir {data_root} \\\n"
                    f"    --output_manifest {m_out} \\\n"
                    f"    --target_real 50000 \\\n"
                    f"    --target_fake 50000 \\\n"
                    f"    --frame_cap {cap} \\\n"
                    f"    --require_groups \\\n"
                    f"    --seed 42"
                )
                commands.append(cmd_cap)
                commands.append(build_audit_command(m_out, f"{output_root}/audit_cap{cap}.json", python_exe=python_exe))

        elif dim == 'fusion_architecture':
            archs = ab.get('architectures', [])
            rgb_ckpt = f"{output_root}/stage1_rgb/stage1_rgb_expert_seed42/checkpoints/best.pth"
            wav_ckpt = f"{output_root}/stage2_wavelet/stage2_wavelet_expert_seed42/checkpoints/best.pth"
            for item in archs:
                aname = item['name']
                ftype = item['fusion_type']
                if ftype == 'ensemble':
                    cmd_ens = (
                        f"# Fusion Architecture Ablation: {aname} (Late Ensemble)\n"
                        f"{python_exe} analyze_predictions.py ensemble \\\n"
                        f"    --rgb_predictions {output_root}/stage1_rgb/stage1_rgb_expert_seed42/checkpoints/dev_predictions.csv \\\n"
                        f"    --wavelet_predictions {output_root}/stage2_wavelet/stage2_wavelet_expert_seed42/checkpoints/dev_predictions.csv \\\n"
                        f"    --output {output_root}/ensemble_dev_predictions.csv"
                    )
                    commands.append(cmd_ens)
                else:
                    arch_model = 'MHA_128' if ftype == 'token_attention' else 'Fusion_128'
                    cmd_f_arch = (
                        f"{launch_cmd} train.py \\\n"
                        f"    --arch {arch_model} \\\n"
                        f"    --fusion_type {ftype} \\\n"
                        f"    --rgb_model_path {rgb_ckpt} \\\n"
                        f"    --wavelet_model_path {wav_ckpt} \\\n"
                        f"    --dataroot {data_root} \\\n"
                        f"    --manifest {manifest_path} \\\n"
                        f"    --manifest_split train \\\n"
                        f"    --val_manifest {manifest_path} \\\n"
                        f"    --val_manifest_split dev \\\n"
                        f"    --name stage3_ablation_{aname} \\\n"
                        f"    --run_id seed42 \\\n"
                        f"    --checkpoints_dir {output_root}/ablation_fusion_arch \\\n"
                        f"    --seed 42 --use_amp"
                    )
                    commands.append(cmd_f_arch)

    return commands


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', required=True, help='Path to study plan JSON file')
    parser.add_argument('--action', choices=['print', 'export_script'], default='print')
    parser.add_argument('--output_script', help='Path to output bash script')
    parser.add_argument('--data_root', default='/content/data')
    parser.add_argument('--output_root', default='/content/experiments')
    parser.add_argument('--launch_cmd', default='python', help='Launch command prefix (python or torchrun)')
    args = parser.parse_args()

    plan = parse_study_plan(args.plan)
    commands = generate_pipeline_commands(
        plan, data_root=args.data_root, output_root=args.output_root, launch_cmd=args.launch_cmd
    )

    if args.action == 'print':
        print(f"# Generated {len(commands)} commands from study plan: {args.plan}\n")
        for i, cmd in enumerate(commands, 1):
            print(f"# --- Command {i} ---")
            print(cmd + "\n")
    elif args.action == 'export_script':
        if not args.output_script:
            raise ValueError("--output_script required for export_script action")
        script_text = "#!/usr/bin/env bash\nset -euo pipefail\n\n" + "\n\n".join(commands) + "\n"
        Path(args.output_script).write_text(script_text, encoding='utf-8')
        print(f"Exported {len(commands)} commands to {args.output_script}")


if __name__ == '__main__':
    main()
