"""Unit tests for Paper 1 Revision capacity-matched heads, exact metrics, calibration, and FGSM sweep."""
import unittest
import numpy as np
import torch
import torch.nn as nn

from models.mha.ablation_heads import build_ablation_head, count_trainable_params
from tools.paper1_revision_suite import (
    EndToEndDifferentiableWrapper,
    cluster_bootstrap_ci,
    compute_embedding_separability,
    compute_exact_metrics,
    holm_bonferroni,
    mcnemar_exact_test,
    paired_seed_statistics,
    run_fgsm_epsilon_sweep,
    train_cached_fusion_head,
)


class TestPaper1RevisionSuite(unittest.TestCase):
    def test_ablation_heads_capacity_matching_and_shapes(self):
        """Verify capacity-matched heads match TokenAttentionFusion (~199K params) within 1%."""
        mha = build_ablation_head("mha_128", embed_dim=128, num_heads=4, dropout=0.2)
        p_mha = count_trainable_params(mha)
        self.assertEqual(p_mha, 199425)

        for name in ("concat_matched", "no_attn_ffn", "linear_mix"):
            h = build_ablation_head(name, embed_dim=128, dropout=0.2)
            p = count_trainable_params(h)
            ratio = p / p_mha
            self.assertGreater(ratio, 0.98, msg=f"{name} params={p} vs MHA={p_mha}")
            self.assertLess(ratio, 1.02, msg=f"{name} params={p} vs MHA={p_mha}")

        zr, zw = torch.randn(4, 128), torch.randn(4, 128)
        for name in ("mha_128", "gated", "fusion_128", "concat_matched", "no_attn_ffn", "linear_mix"):
            h = build_ablation_head(name, embed_dim=128, dropout=0.2).eval()
            out = h(zr, zw)
            self.assertEqual(tuple(out.shape), (4, 1))
            if name == "mha_128":
                self.assertEqual(tuple(h.last_attention.shape), (4, 4, 3, 3))
            if name == "gated":
                self.assertEqual(tuple(h.last_gate.shape), (4, 2))

    def test_exact_metrics_consistency_and_calibration(self):
        """Confusion counts must reproduce accuracy, precision, recall, and Macro-F1 identically."""
        y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])
        logits = np.array([-3.0, -1.5, -0.2, +0.5, -0.4, +0.2, +1.5, +2.5, +3.0, +4.0])
        m = compute_exact_metrics(y, logits, threshold=0.0)
        cm = m["confusion_matrix"]
        self.assertEqual((cm["tn"], cm["fp"], cm["fn"], cm["tp"]), (3, 1, 1, 5))
        self.assertAlmostEqual(m["accuracy"], (3 + 5) / 10, places=6)
        self.assertAlmostEqual(m["real"]["precision"], 3 / 4, places=6)
        self.assertAlmostEqual(m["real"]["recall"], 3 / 4, places=6)
        self.assertAlmostEqual(m["fake"]["precision"], 5 / 6, places=6)
        self.assertAlmostEqual(m["fake"]["recall"], 5 / 6, places=6)
        self.assertAlmostEqual(m["macro_f1"], 0.5 * (m["real"]["f1"] + m["fake"]["f1"]), places=6)
        self.assertGreaterEqual(m["ece"], 0.0)
        self.assertGreaterEqual(m["brier_score"], 0.0)

        ci = cluster_bootstrap_ci(y, logits, cluster_ids=[0, 0, 1, 1, 2, 2, 3, 3, 4, 4], n_boot=50)
        self.assertIn("roc_auc", ci)
        mc = mcnemar_exact_test(y, logits, -logits)
        self.assertIn("p_value", mc)
        adj = holm_bonferroni([0.01, 0.04, 0.03, 0.20])
        self.assertEqual(len(adj), 4)
        self.assertAlmostEqual(adj[0], 0.04, places=6)

    def test_cached_fusion_training_paired_stats_and_fgsm(self):
        """Test cross-fitted cached training, paired seed stats, separability, and end-to-end FGSM sweep."""
        torch.manual_seed(0)
        n_tr, n_dv = 120, 40
        tr_y = torch.cat([torch.zeros(n_tr // 2), torch.ones(n_tr // 2)])
        dv_y = torch.cat([torch.zeros(n_dv // 2), torch.ones(n_dv // 2)])
        tr_r = torch.randn(n_tr, 128) + tr_y.unsqueeze(1) * 1.2
        tr_w = torch.randn(n_tr, 128) + tr_y.unsqueeze(1) * 1.2
        dv_r = torch.randn(n_dv, 128) + dv_y.unsqueeze(1) * 1.2
        dv_w = torch.randn(n_dv, 128) + dv_y.unsqueeze(1) * 1.2

        runs_a, runs_b = [], []
        for s in (42, 100):
            ha, info_a = train_cached_fusion_head(
                "mha_128", tr_r, tr_w, tr_y, dv_r, dv_w, dv_y, epochs=3, batch_size=32, seed=s, device="cpu"
            )
            hb, _ = train_cached_fusion_head(
                "concat_matched", tr_r, tr_w, tr_y, dv_r, dv_w, dv_y, epochs=3, batch_size=32, seed=s, device="cpu"
            )
            self.assertEqual(len(info_a["history"]["dev_auc"]), 3)
            with torch.no_grad():
                runs_a.append(compute_exact_metrics(dv_y.numpy(), ha(dv_r, dv_w).flatten().numpy()))
                runs_b.append(compute_exact_metrics(dv_y.numpy(), hb(dv_r, dv_w).flatten().numpy()))

        stats = paired_seed_statistics(runs_a, runs_b)
        self.assertIn("p_value_holm", stats["metrics"]["roc_auc"])
        sep = compute_embedding_separability(dv_r.numpy(), dv_y.numpy().astype(int))
        self.assertGreater(sep["silhouette_score"], 0.0)

        # Tiny differentiable end-to-end FGSM check
        dummy_rgb = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 128))
        dummy_wav = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(3, 128))
        head = build_ablation_head("mha_128", embed_dim=128)
        wrap = EndToEndDifferentiableWrapper("fusion", rgb_model=dummy_rgb, wav_model=dummy_wav, fusion_head=head)
        x_clean = torch.rand(8, 3, 32, 32)
        y_clean = np.array([0, 0, 0, 0, 1, 1, 1, 1])
        fgsm = run_fgsm_epsilon_sweep({"MHA_128": wrap}, x_clean, y_clean, epsilons=[0.0, 4 / 255], batch_size=4, device="cpu")
        self.assertIn("4/255", fgsm["MHA_128"])
        self.assertIn("real", fgsm["MHA_128"]["4/255"])


if __name__ == "__main__":
    unittest.main()
