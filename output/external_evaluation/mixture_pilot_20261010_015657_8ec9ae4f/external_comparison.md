# External RGB pilot comparison

Thresholds frozen on shared development data; no external threshold fitting.

| Arm | Threshold rule | Threshold | Accuracy | Balanced accuracy | AUC | Real recall | Fake recall | Macro F1 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| arm_a | fixed_0_5 | 0.50000000 | 51.65% | 51.65% | 0.7748 | 3.29% | 100.00% | 0.3689 |
| arm_a | frozen_dev | 0.82718241 | 53.88% | 53.88% | 0.7748 | 7.79% | 99.97% | 0.4144 |
| arm_b | fixed_0_5 | 0.50000000 | 50.27% | 50.27% | 0.8180 | 0.53% | 100.00% | 0.3392 |
| arm_b | frozen_dev | 0.42735213 | 50.23% | 50.23% | 0.8180 | 0.46% | 100.00% | 0.3384 |
