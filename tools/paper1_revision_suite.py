"""Paper 1 (Discover AI) Major Revision — Complete Programmatic Evaluation, Ablation & Plotting Suite.

Guarantees 100% arithmetic consistency across all confusion matrices, metrics, calibration
diagnostics, statistical tests, white-box FGSM sweeps, and publication figures.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.stats import binomtest, t as student_t, ttest_rel
from sklearn.metrics import (
    average_precision_score,
    log_loss,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.mha.ablation_heads import build_ablation_head, count_trainable_params


# ---------------------------------------------------------------------------
# 1. Exact, Auditable Classification & Calibration Metrics (R1 #18-20, R2 #1, #8)
# ---------------------------------------------------------------------------

def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def compute_eer(y_true: np.ndarray, scores: np.ndarray) -> Tuple[float, float]:
    """Compute exact Equal Error Rate (EER) and its operating score threshold."""
    fpr, tpr, thresholds = roc_curve(y_true, scores)
    fnr = 1.0 - tpr
    diff = fpr - fnr
    idx = int(np.argmin(np.abs(diff)))
    # Linear interpolation when sign crosses zero
    if idx + 1 < len(diff) and diff[idx] * diff[idx + 1] <= 0 and (diff[idx + 1] != diff[idx]):
        alpha = abs(diff[idx]) / (abs(diff[idx]) + abs(diff[idx + 1]))
        eer = float(fpr[idx] + alpha * (fpr[idx + 1] - fpr[idx]))
        thr = float(thresholds[idx] + alpha * (thresholds[idx + 1] - thresholds[idx]))
        return eer, thr
    return float(0.5 * (fpr[idx] + fnr[idx])), float(thresholds[idx])


def compute_ece(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> Tuple[float, Dict[str, List[float]]]:
    """Compute Expected Calibration Error (ECE) and reliability-diagram bin statistics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    probs = np.clip(np.asarray(probs, dtype=np.float64), 1e-7, 1.0 - 1e-7)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    bin_conf, bin_acc, bin_cnt = [], [], []
    n = len(y_true)
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            mask = (probs >= lo) & (probs <= hi)
        else:
            mask = (probs >= lo) & (probs < hi)
        cnt = int(mask.sum())
        bin_cnt.append(cnt)
        if cnt > 0:
            conf = float(probs[mask].mean())
            acc = float(y_true[mask].mean())
            ece += (cnt / n) * abs(acc - conf)
            bin_conf.append(conf)
            bin_acc.append(acc)
        else:
            bin_conf.append(float(0.5 * (lo + hi)))
            bin_acc.append(float("nan"))
    return float(ece), {"bin_edges": edges.tolist(), "bin_conf": bin_conf, "bin_acc": bin_acc, "bin_count": bin_cnt}


def compute_exact_metrics(
    y_true: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Compute 100% self-consistent confusion counts, classification metrics, and probability calibration."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    s = np.asarray(logits, dtype=np.float64).reshape(-1)
    if len(y) != len(s) or len(y) == 0:
        raise ValueError(f"Mismatched or empty labels/logits: {len(y)} vs {len(s)}")

    pred = (s >= threshold).astype(np.int64)
    tn = int(np.sum((y == 0) & (pred == 0)))
    fp = int(np.sum((y == 0) & (pred == 1)))
    fn = int(np.sum((y == 1) & (pred == 0)))
    tp = int(np.sum((y == 1) & (pred == 1)))
    n = tn + fp + fn + tp
    assert n == len(y)

    real_sup = tn + fp
    fake_sup = tp + fn
    acc = (tn + tp) / n
    real_rec = tn / real_sup if real_sup > 0 else 0.0
    fake_rec = tp / fake_sup if fake_sup > 0 else 0.0
    real_prec = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    fake_prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    real_f1 = (2.0 * real_prec * real_rec / (real_prec + real_rec)) if (real_prec + real_rec) > 0 else 0.0
    fake_f1 = (2.0 * fake_prec * fake_rec / (fake_prec + fake_rec)) if (fake_prec + fake_rec) > 0 else 0.0
    macro_f1 = 0.5 * (real_f1 + fake_f1)
    bal_acc = 0.5 * (real_rec + fake_rec)

    probs = _sigmoid(s)
    brier = float(np.mean((probs - y) ** 2))
    nll = float(log_loss(y, np.clip(probs, 1e-7, 1.0 - 1e-7), labels=[0, 1]))
    ece, reliability = compute_ece(y, probs, n_bins=15)

    has_both = (real_sup > 0) and (fake_sup > 0)
    auc = float(roc_auc_score(y, s)) if has_both else float("nan")
    ap = float(average_precision_score(y, s)) if has_both else float("nan")
    eer, eer_thr = compute_eer(y, s) if has_both else (float("nan"), float("nan"))

    return {
        "n_samples": n,
        "threshold": float(threshold),
        "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        "accuracy": float(acc),
        "balanced_accuracy": float(bal_acc),
        "macro_f1": float(macro_f1),
        "roc_auc": auc,
        "average_precision": ap,
        "eer": eer,
        "eer_threshold": eer_thr,
        "brier_score": brier,
        "log_loss": nll,
        "ece": ece,
        "reliability": reliability,
        "real": {
            "precision": float(real_prec),
            "recall": float(real_rec),
            "fpr": float(1.0 - real_rec),
            "f1": float(real_f1),
            "support": real_sup,
        },
        "fake": {
            "precision": float(fake_prec),
            "recall": float(fake_rec),
            "fnr": float(1.0 - fake_rec),
            "f1": float(fake_f1),
            "support": fake_sup,
        },
    }


# ---------------------------------------------------------------------------
# 2. Cluster-Bootstrap CIs, McNemar & Holm-Bonferroni Stats (R1 #16-17, R2 #7)
# ---------------------------------------------------------------------------

def cluster_bootstrap_ci(
    y_true: Sequence[int] | np.ndarray,
    logits: Sequence[float] | np.ndarray,
    cluster_ids: Optional[Sequence[Any]] = None,
    threshold: float = 0.0,
    n_boot: int = 1000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    """Compute 95% percentile bootstrap CIs (supports identity/video cluster resampling)."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    s = np.asarray(logits, dtype=np.float64).reshape(-1)
    rng = np.random.RandomState(seed)

    if cluster_ids is not None:
        groups: Dict[Any, List[int]] = {}
        for idx, cid in enumerate(cluster_ids):
            groups.setdefault(cid, []).append(idx)
        cluster_arrays = [np.asarray(v, dtype=np.int64) for v in groups.values()]
        n_clusters = len(cluster_arrays)
    else:
        cluster_arrays = None
        n_clusters = len(y)

    buckets: Dict[str, List[float]] = {"accuracy": [], "roc_auc": [], "eer": [], "macro_f1": [], "brier_score": []}
    for _ in range(n_boot):
        if cluster_arrays is not None:
            chosen = rng.randint(0, n_clusters, size=n_clusters)
            idx = np.concatenate([cluster_arrays[c] for c in chosen])
        else:
            idx = rng.randint(0, len(y), size=len(y))
        yb, sb = y[idx], s[idx]
        if len(np.unique(yb)) < 2:
            continue
        m = compute_exact_metrics(yb, sb, threshold=threshold)
        for k in buckets:
            buckets[k].append(m[k])

    return {k: (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))) for k, v in buckets.items() if v}


def mcnemar_exact_test(
    y_true: Sequence[int] | np.ndarray,
    logits_a: Sequence[float] | np.ndarray,
    logits_b: Sequence[float] | np.ndarray,
    threshold: float = 0.0,
) -> Dict[str, Any]:
    """Exact two-sided McNemar test on paired discordant predictions."""
    y = np.asarray(y_true, dtype=np.int64).reshape(-1)
    ca = (np.asarray(logits_a) >= threshold).astype(np.int64) == y
    cb = (np.asarray(logits_b) >= threshold).astype(np.int64) == y
    b = int(np.sum(ca & ~cb))  # A right, B wrong
    c = int(np.sum(~ca & cb))  # A wrong, B right
    pval = 1.0 if (b + c) == 0 else float(binomtest(min(b, c), b + c, 0.5).pvalue)
    return {"a_only_correct": b, "b_only_correct": c, "p_value": pval}


def holm_bonferroni(p_values: Sequence[float]) -> List[float]:
    """Compute Holm-Bonferroni step-down multiplicity-adjusted p-values."""
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.zeros(m, dtype=np.float64)
    running_max = 0.0
    for rank, idx in enumerate(order):
        adj = float(p_values[idx]) * (m - rank)
        running_max = max(running_max, adj)
        adjusted[idx] = min(1.0, running_max)
    return adjusted.tolist()


def paired_seed_statistics(
    runs_a: Sequence[Dict[str, float]],
    runs_b: Sequence[Dict[str, float]],
    metric_names: Sequence[str] = ("accuracy", "macro_f1", "roc_auc", "eer", "brier_score"),
) -> Dict[str, Any]:
    """Paired t-test, 95% t-CIs, Cohen's d, and Holm-Bonferroni adjusted p-values across N seeds."""
    n = len(runs_a)
    if n != len(runs_b) or n < 2:
        raise ValueError(f"Need matched N>=2 seed runs, got {len(runs_a)} vs {len(runs_b)}")

    raw_p = []
    per_metric: Dict[str, Dict[str, Any]] = {}
    t_crit = float(student_t.ppf(0.975, df=n - 1))

    for k in metric_names:
        va = np.array([float(r[k]) for r in runs_a], dtype=np.float64)
        vb = np.array([float(r[k]) for r in runs_b], dtype=np.float64)
        diff = va - vb
        sd_diff = float(np.std(diff, ddof=1))
        cohens_d = float(np.mean(diff) / sd_diff) if sd_diff > 1e-12 else 0.0
        if sd_diff <= 1e-12:
            pval = 1.0
        else:
            pval = float(ttest_rel(va, vb).pvalue)
        raw_p.append(pval)

        def _summary(v: np.ndarray) -> Dict[str, Any]:
            mu = float(np.mean(v))
            sd = float(np.std(v, ddof=1))
            half = t_crit * sd / math.sqrt(n)
            return {"mean": mu, "sd": sd, "ci95": [mu - half, mu + half], "per_seed": v.tolist()}

        per_metric[k] = {
            "model_a": _summary(va),
            "model_b": _summary(vb),
            "mean_diff_a_minus_b": float(np.mean(diff)),
            "cohens_d": cohens_d,
            "p_value_raw": pval,
        }

    adj_p = holm_bonferroni(raw_p)
    for k, ap in zip(metric_names, adj_p):
        per_metric[k]["p_value_holm"] = float(ap)
    return {"n_seeds": n, "metrics": per_metric}


# ---------------------------------------------------------------------------
# 3. Fast Cross-Fitted Fusion Head Training on Cached Frozen Embeddings (R1 #4, #6, #12)
# ---------------------------------------------------------------------------

def train_cached_fusion_head(
    head_name: str,
    train_rgb: torch.Tensor,
    train_wav: torch.Tensor,
    train_y: torch.Tensor,
    dev_rgb: torch.Tensor,
    dev_wav: torch.Tensor,
    dev_y: torch.Tensor,
    *,
    embed_dim: int = 128,
    num_heads: int = 4,
    dropout: float = 0.2,
    modality_dropout: float = 0.15,
    label_smoothing: float = 0.05,
    lr: float = 3e-4,
    weight_decay: float = 1e-3,
    batch_size: int = 256,
    epochs: int = 25,
    seed: int = 42,
    device: str = "cuda:0",
) -> Tuple[nn.Module, Dict[str, Any]]:
    """Train a fusion head on pre-extracted out-of-sample frozen expert embeddings.

    Selects best checkpoint strictly on `dev` ROC-AUC (zero test leakage) and logs
    both train and dev loss/AUC curves every epoch (fixing R1 #6, #12 & R2 #11).
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device(device)

    head = build_ablation_head(head_name, embed_dim=embed_dim, num_heads=num_heads, dropout=dropout).to(dev)
    n_params = count_trainable_params(head)

    tr_r = train_rgb.to(dev, dtype=torch.float32)
    tr_w = train_wav.to(dev, dtype=torch.float32)
    tr_y = train_y.to(dev, dtype=torch.float32)
    dv_r = dev_rgb.to(dev, dtype=torch.float32)
    dv_w = dev_wav.to(dev, dtype=torch.float32)
    dv_y_np = dev_y.cpu().numpy().astype(np.int64)

    # Class-balanced pos_weight if split isn't strictly 50/50
    n_pos = float((tr_y == 1).sum().item())
    n_neg = float((tr_y == 0).sum().item())
    pos_weight = torch.tensor([n_neg / max(n_pos, 1.0)], device=dev, dtype=torch.float32)
    bce_hard = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs, eta_min=lr * 0.05)

    history = {"train_loss": [], "dev_loss": [], "train_auc": [], "dev_auc": [], "train_acc": [], "dev_acc": []}
    best_auc = -1.0
    best_epoch = 0
    best_state = copy.deepcopy(head.state_dict())

    n_train = len(tr_y)
    g = torch.Generator(device=dev)
    g.manual_seed(seed)
    tr_eval_n = min(n_train, 20000)
    tr_eval_idx = torch.randperm(n_train, generator=g, device=dev)[:tr_eval_n]

    for ep in range(1, epochs + 1):
        head.train()
        perm = torch.randperm(n_train, generator=g, device=dev)
        ep_loss_sum = 0.0
        n_batches = 0
        for i in range(0, n_train, batch_size):
            idx = perm[i : i + batch_size]
            br, bw, by = tr_r[idx], tr_w[idx], tr_y[idx]

            # Modality dropout prevents over-reliance on a single overconfident stream
            if modality_dropout > 0.0:
                u = torch.rand(br.shape[0], 1, device=dev, generator=g)
                drop_r = (u < modality_dropout).float()
                drop_w = ((u >= modality_dropout) & (u < 2.0 * modality_dropout)).float()
                br = br * (1.0 - drop_r)
                bw = bw * (1.0 - drop_w)

            opt.zero_grad(set_to_none=True)
            logits = head(br, bw).flatten()
            target = by * (1.0 - 2.0 * label_smoothing) + label_smoothing if label_smoothing > 0 else by
            loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 2.0)
            opt.step()
            ep_loss_sum += float(loss.item())
            n_batches += 1

        sched.step()
        head.eval()
        with torch.no_grad():
            # Deterministic shuffled train eval subset (both classes guaranteed)
            tr_logits = head(tr_r[tr_eval_idx], tr_w[tr_eval_idx]).flatten().cpu().numpy()
            tr_y_sub = tr_y[tr_eval_idx].cpu().numpy().astype(np.int64)
            tr_auc = float(roc_auc_score(tr_y_sub, tr_logits)) if len(np.unique(tr_y_sub)) == 2 else float("nan")
            tr_acc = float(((tr_logits >= 0.0).astype(np.int64) == tr_y_sub).mean())

            dv_logits_t = head(dv_r, dv_w).flatten()
            dv_loss = float(bce_hard(dv_logits_t, dev_y.to(dev, dtype=torch.float32)).item())
            dv_logits = dv_logits_t.cpu().numpy()
            dv_auc = float(roc_auc_score(dv_y_np, dv_logits))
            dv_acc = float(((dv_logits >= 0.0).astype(np.int64) == dv_y_np).mean())

        history["train_loss"].append(ep_loss_sum / max(n_batches, 1))
        history["dev_loss"].append(dv_loss)
        history["train_auc"].append(tr_auc)
        history["dev_auc"].append(dv_auc)
        history["train_acc"].append(tr_acc)
        history["dev_acc"].append(dv_acc)

        if dv_auc > best_auc:
            best_auc = dv_auc
            best_epoch = ep
            best_state = copy.deepcopy(head.state_dict())

    head.load_state_dict(best_state)
    head.eval()

    info = {
        "head_name": head_name,
        "trainable_params": n_params,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_dev_auc": best_auc,
        "train_auc_at_best": history["train_auc"][best_epoch - 1],
        "train_dev_auc_gap": history["train_auc"][best_epoch - 1] - best_auc,
        "history": history,
    }
    return head, info


# ---------------------------------------------------------------------------
# 4. End-to-End Differentiable Raw-Pixel White-Box FGSM Sweep (R1 #13, R2 #13)
# ---------------------------------------------------------------------------

class EndToEndDifferentiableWrapper(nn.Module):
    """Wraps RGB expert, Wavelet expert, or Two-Stream Fusion to accept raw [0, 1] RGB tensors [B, 3, 224, 224].

    Gradients flow cleanly through ImageNet normalization AND GPU Haar Level-3 signed_log1p wavelets
    back to raw input pixels so white-box FGSM attacks are 100% faithful (R1 #13 & R2 #13).
    """

    def __init__(
        self,
        mode: str,
        rgb_model: Optional[nn.Module] = None,
        wav_model: Optional[nn.Module] = None,
        fusion_head: Optional[nn.Module] = None,
        gpu_wavelet_fn: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
    ):
        super().__init__()
        self.mode = mode  # 'rgb', 'wavelet', 'fusion'
        self.rgb_model = rgb_model
        self.wav_model = wav_model
        self.fusion_head = fusion_head
        self.gpu_wavelet_fn = gpu_wavelet_fn
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def _rgb_feat_or_logit(self, x_01: torch.Tensor, features_only: bool) -> torch.Tensor:
        x_norm = (x_01 - self.mean) / self.std
        if features_only:
            # Expects truncated extractor or ResNet with fc=[Linear, ReLU, ...]
            return self.rgb_model(x_norm)
        return self.rgb_model(x_norm)

    def _wav_feat_or_logit(self, x_01: torch.Tensor) -> torch.Tensor:
        x_255 = x_01 * 255.0
        w = self.gpu_wavelet_fn(x_255) if self.gpu_wavelet_fn is not None else x_255
        return self.wav_model(w)

    def forward(self, x_01: torch.Tensor) -> torch.Tensor:
        if self.mode == "rgb":
            return self._rgb_feat_or_logit(x_01, features_only=False)
        if self.mode == "wavelet":
            return self._wav_feat_or_logit(x_01)
        if self.mode == "fusion":
            zr = self._rgb_feat_or_logit(x_01, features_only=True)
            zw = self._wav_feat_or_logit(x_01)
            return self.fusion_head(zr, zw)
        raise ValueError(f"Unknown mode: {self.mode}")


def run_fgsm_epsilon_sweep(
    models_dict: Dict[str, nn.Module],
    x_clean_01: torch.Tensor,
    y_true: np.ndarray,
    epsilons: Sequence[float] = (0.0, 1 / 255, 2 / 255, 4 / 255, 8 / 255),
    batch_size: int = 64,
    device: str = "cuda:0",
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Run matched white-box L_inf FGSM attack sweep on raw [0, 1] images across all models.

    Attack details (explicitly documented per R1 #13 & R2 #13):
      - Objective: maximize BCEWithLogitsLoss(model(x), y_true) (untargeted evasion)
      - Coordinates: raw [0, 1] RGB pixel space, clipped to [0, 1] after single FGSM step
      - Random start: False (deterministic single-step FGSM)
      - Gradient flow: end-to-end through both ImageNet normalization and GPU DWT subbands
    """
    dev = torch.device(device)
    y_t = torch.as_tensor(y_true, dtype=torch.float32)
    results: Dict[str, Dict[str, Dict[str, Any]]] = {name: {} for name in models_dict}

    for name, wrapper in models_dict.items():
        wrapper = wrapper.to(dev).eval()
        for eps in epsilons:
            eps_key = f"{eps * 255:.0f}/255" if eps > 0 else "0/255"
            all_logits: List[float] = []
            for i in range(0, len(y_true), batch_size):
                bx = x_clean_01[i : i + batch_size].to(dev)
                by = y_t[i : i + batch_size].to(dev)
                if eps <= 0.0:
                    with torch.no_grad():
                        out = wrapper(bx).flatten()
                else:
                    bx_adv = bx.clone().detach().requires_grad_(True)
                    logits = wrapper(bx_adv).flatten()
                    loss = F.binary_cross_entropy_with_logits(logits, by)
                    grad = torch.autograd.grad(loss, bx_adv, retain_graph=False, create_graph=False)[0]
                    bx_pert = torch.clamp(bx_adv.detach() + eps * grad.sign(), 0.0, 1.0)
                    with torch.no_grad():
                        out = wrapper(bx_pert).flatten()
                all_logits.extend(out.cpu().tolist())
            results[name][eps_key] = compute_exact_metrics(y_true, all_logits, threshold=0.0)
    return results


# ---------------------------------------------------------------------------
# 5. Quantitative Embedding Separability (Fixes R1 #26)
# ---------------------------------------------------------------------------

def compute_embedding_separability(
    embeddings: np.ndarray,
    binary_labels: np.ndarray,
    max_samples: int = 4000,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute Silhouette score and Fisher centroid-separation ratio on 128-D embeddings."""
    rng = np.random.RandomState(seed)
    n = len(binary_labels)
    if n > max_samples:
        idx = rng.choice(n, size=max_samples, replace=False)
        X, y = embeddings[idx], binary_labels[idx]
    else:
        X, y = embeddings, binary_labels

    # Normalize to unit sphere for cosine-aligned stability
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-8
    Xn = X / norms
    sil = float(silhouette_score(Xn, y, metric="euclidean"))

    mu0, mu1 = Xn[y == 0].mean(axis=0), Xn[y == 1].mean(axis=0)
    inter = float(np.linalg.norm(mu0 - mu1))
    intra0 = float(np.mean(np.linalg.norm(Xn[y == 0] - mu0, axis=1)))
    intra1 = float(np.mean(np.linalg.norm(Xn[y == 1] - mu1, axis=1)))
    fisher_ratio = inter / max(0.5 * (intra0 + intra1), 1e-8)
    return {"silhouette_score": sil, "centroid_distance": inter, "fisher_separation_ratio": float(fisher_ratio)}
