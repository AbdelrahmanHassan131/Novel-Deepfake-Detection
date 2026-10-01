"""Table generation utilities for empirical reviewer evidence.

Takes structured JSON outputs from training, evaluation, and profiling
and generates formatted Markdown/LaTeX tables conforming to journal requirements.
"""
from typing import Dict, Any, List, Optional
import json


def format_dataset_counts_table(summary: Dict[str, Any]) -> str:
    """Render Table 1: Dataset Composition & Split Counts."""
    lines = [
        "| Split | Real Images | Fake Images | Total Images | Independent Groups | Frame Cap | Real/Fake Ratio |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    splits = summary.get('splits')
    if not splits and 'split_sample_counts' in summary:
        split_counts = summary.get('split_sample_counts', {})
        source_dist = summary.get('source_class_distribution', [])
        shortage = summary.get('shortage_summary', {})
        frame_cap = shortage.get('frame_cap_applied', 'N/A')

        splits = {}
        for split_name, total in split_counts.items():
            real = sum(row.get('real', 0) for row in source_dist if row.get('split') == split_name)
            fake = sum(row.get('fake', 0) for row in source_dist if row.get('split') == split_name)
            splits[split_name] = {
                'real_count': real,
                'fake_count': fake,
                'total_count': total,
                'group_count': 'N/A',
                'frame_cap': frame_cap,
            }

    if not splits:
        raise ValueError("Dataset summary does not contain valid 'splits' data; cannot generate Table 1 without split information.")

    for split_name, data in splits.items():
        real = data.get('real_count', 0)
        fake = data.get('fake_count', 0)
        total = data.get('total_count', real + fake)
        groups = data.get('group_count', 'N/A')
        cap = data.get('frame_cap', 'N/A')
        ratio = f"{real/fake:.2f}:1" if fake > 0 else "N/A"
        lines.append(f"| **{split_name}** | {real:,} | {fake:,} | {total:,} | {groups} | {cap} | {ratio} |")
    return "\n".join(lines)


def format_multiseed_table(seed_report: Dict[str, Any]) -> str:
    """Render Table 2: 3-Seed Performance Summary (Mean ± SD)."""
    lines = [
        "| Architecture / Model | Seeds Evaluated | ROC AUC (%) | Balanced Acc (%) | Accuracy (%) | EER (%) | FPR @ TPR 95% (%) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    num_seeds = seed_report.get('num_independent_seeds', 3)
    agg = seed_report.get('aggregated_metrics', {})

    def fmt_metric(name):
        m = agg.get(name, {})
        if 'mean' in m and 'std' in m:
            return f"{m['mean']*100:.2f} ± {m['std']*100:.2f}"
        return "N/A"

    lines.append(
        f"| **Evaluated Pipeline** | {num_seeds} | {fmt_metric('roc_auc')} | {fmt_metric('balanced_accuracy')} | "
        f"{fmt_metric('accuracy')} | {fmt_metric('eer')} | {fmt_metric('fpr_at_tpr95')} |"
    )
    return "\n".join(lines)


def format_paired_comparison_table(comparison_report: Dict[str, Any]) -> str:
    """Render Table 3: Paired Model Difference with Cluster Percentile Bootstrap."""
    lines = [
        "| Metric | Model 1 (Proposed) | Model 2 (Control) | Observed Difference | 95% Bootstrap CI | 95% CI Excludes 0? |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    diffs = comparison_report.get('first_minus_second', {})
    if not diffs or diffs.get('status') != 'computed':
        raise ValueError("Comparison report does not contain valid computed paired difference data.")

    intervals = diffs.get('intervals_95', {})
    obs_diffs = diffs.get('observed_difference', {})
    models = comparison_report.get('models', [])
    m1_overall = models[0].get('overall', {}) if len(models) > 0 and isinstance(models[0], dict) else {}
    m2_overall = models[1].get('overall', {}) if len(models) > 1 and isinstance(models[1], dict) else {}

    for metric, ci in intervals.items():
        low = ci.get('low', 0.0) * 100
        high = ci.get('high', 0.0) * 100
        # Use actual aligned full-sample difference; NEVER midpoint of percentile CI
        if 'observed' in ci and ci['observed'] is not None:
            diff_obs = ci['observed'] * 100
        elif metric in obs_diffs and obs_diffs[metric] is not None:
            diff_obs = obs_diffs[metric] * 100
        else:
            diff_obs = None

        diff_str = f"{diff_obs:+.2f}%" if diff_obs is not None else "N/A"

        m1_val = m1_overall.get(metric)
        m1_str = f"{m1_val * 100:.2f}%" if m1_val is not None else "—"

        m2_val = m2_overall.get(metric)
        m2_str = f"{m2_val * 100:.2f}%" if m2_val is not None else "—"

        # Label interval exclusion directly without claiming uncalculated p-value
        sig = "Yes (95% CI excludes 0)" if (low > 0 or high < 0) else "No (95% CI spans 0)"
        lines.append(f"| **{metric}** | {m1_str} | {m2_str} | {diff_str} | [{low:+.2f}%, {high:+.2f}%] | {sig} |")
    return "\n".join(lines)


def format_ablation_table(ablation_results: List[Dict[str, Any]], dimension_name: str) -> str:
    """Render Table 4: Targeted Ablation Sweep."""
    lines = [
        f"| {dimension_name} | ROC AUC (%) | Balanced Acc (%) | Latency (ms) | FLOPs (G) | Status |",
        "|---|:---:|:---:|:---:|:---:|:---:|",
    ]
    for row in ablation_results:
        val = row.get('condition_value', 'Unknown')
        auc = f"{row.get('roc_auc', 0.0)*100:.2f}" if 'roc_auc' in row else "Pending"
        bacc = f"{row.get('balanced_accuracy', 0.0)*100:.2f}" if 'balanced_accuracy' in row else "Pending"
        lat = f"{row.get('latency_ms', 0.0):.1f}" if 'latency_ms' in row else "Pending"
        flops = f"{row.get('flops_g', 0.0):.2f}" if 'flops_g' in row else "Pending"
        status = row.get('status', 'Completed')
        lines.append(f"| **{val}** | {auc} | {bacc} | {lat} | {flops} | {status} |")
    return "\n".join(lines)


def format_robustness_table(robustness_report: Dict[str, Any]) -> str:
    """Render Table 5: Robustness under Corruptions and Pixel Attacks."""
    lines = [
        "| Condition | Perturbation Type | Budget / Severity | Accuracy (%) | Balanced Acc (%) | Fake Recall (%) | AUC (%) |",
        "|---|---|---|:---:|:---:|:---:|:---:|",
    ]
    conds = robustness_report.get('conditions', {})
    for name, data in conds.items():
        overall = data.get('overall', {})
        acc = f"{overall.get('accuracy', 0.0)*100:.2f}" if overall.get('accuracy') is not None else "N/A"
        bacc = f"{overall.get('balanced_accuracy', 0.0)*100:.2f}" if overall.get('balanced_accuracy') is not None else "N/A"
        fake_rec = f"{overall.get('per_class', {}).get('fake', {}).get('recall', 0.0)*100:.2f}" if overall.get('per_class') else "N/A"
        auc = f"{overall.get('roc_auc', 0.0)*100:.2f}" if overall.get('roc_auc') is not None else "N/A"
        lines.append(f"| **{name}** | Benchmark Condition | As Specified | {acc} | {bacc} | {fake_rec} | {auc} |")
    return "\n".join(lines)


def format_fairness_table(generalization_report: Dict[str, Any], attribute: str) -> str:
    """Render Table 6: Demographic Subgroup Breakdown."""
    lines = [
        f"| Subgroup ({attribute}) | Samples | Share (%) | Accuracy (%) | Balanced Acc (%) | FPR (%) | FNR (%) | Sample Caution |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    breakdowns = generalization_report.get('breakdowns', {}).get(attribute, {})
    for group, metrics in breakdowns.items():
        count = metrics.get('subgroup_sample_count', metrics.get('samples', 0))
        share = metrics.get('subgroup_share', 0.0) * 100
        acc = f"{metrics.get('accuracy', 0.0)*100:.2f}"
        bacc = f"{metrics.get('balanced_accuracy', 0.0)*100:.2f}" if metrics.get('balanced_accuracy') is not None else "N/A"
        fpr = f"{metrics.get('fpr', 0.0)*100:.2f}" if metrics.get('fpr') is not None else "N/A"
        fnr = f"{metrics.get('fnr', 0.0)*100:.2f}" if metrics.get('fnr') is not None else "N/A"
        caution = "⚠️ N < 30 (high uncertainty)" if metrics.get('caution_small_subgroup') else "Adequate"
        lines.append(f"| **{group}** | {count:,} | {share:.1f}% | {acc} | {bacc} | {fpr} | {fnr} | {caution} |")
    return "\n".join(lines)


def format_detector_cost_table(profile_report: Dict[str, Any]) -> str:
    """Render Table 7: Full Detector Computational Cost (End-to-End)."""
    lines = [
        "| Component / Pipeline | Total Params (M) | Trainable Params (M) | Model Size (MB) | Latency (ms/img) | Throughput (img/s) | Peak GPU VRAM (MB) | FLOPs (G) |",
        "|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|",
    ]
    tot_p = profile_report.get('total_parameters', 0) / 1e6
    trn_p = profile_report.get('trainable_parameters', 0) / 1e6
    sz_mb = profile_report.get('parameter_size_mb', 0.0)
    lat = profile_report.get('latency_ms_per_image', 0.0)
    thr = profile_report.get('throughput_images_per_second', 0.0)
    vram = profile_report.get('gpu_peak_allocated_mb', 'N/A')
    flops = (profile_report.get('flops_per_image_counted') or 0) / 1e9
    vram_str = f"{vram:.1f}" if isinstance(vram, (int, float)) else str(vram)

    lines.append(
        f"| **Full End-to-End Pipeline** | {tot_p:.2f} | {trn_p:.2f} | {sz_mb:.1f} | "
        f"{lat:.2f} | {thr:.1f} | {vram_str} | {flops:.2f} |"
    )
    return "\n".join(lines)
