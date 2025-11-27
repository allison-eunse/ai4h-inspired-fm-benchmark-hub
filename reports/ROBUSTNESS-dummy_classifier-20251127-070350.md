# Robustness Evaluation Report: ROBUSTNESS-dummy_classifier-20251127-070350

## Overview

- **Model**: dummy_classifier
- **Benchmark**: Robustness Testing (ITU AI4H DEL3 Aligned)
- **Date**: 2025-11-27 07:03 UTC

## Aggregate Robustness Scores

| Metric | Score |
|--------|-------|
| **dropout_rAUC** | 0.7758 |
| **expression_rAUC** | 0.7835 |
| **masking_rAUC** | 0.7663 |
| **noise_rAUC** | 0.7740 |
| **perm_equivariance** | 0.7783 |
| **robustness_score** | 0.7749 |

## Probe Results

### Dropout

**Grid**: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

**Logit rAUC**: 0.7758

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 0.0 | 0.7737 |
| 0.05 | 0.7779 |
| 0.1 | 0.7833 |
| 0.2 | 0.7456 |
| 0.3 | 0.7908 |
| 0.5 | 0.7774 |

### Noise

**Grid**: [40.0, 30.0, 20.0, 10.0, 5.0, 0.0]

**Logit rAUC**: 0.7740

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 40.0 | 0.8030 |
| 30.0 | 0.7750 |
| 20.0 | 0.7581 |
| 10.0 | 0.7880 |
| 5.0 | 0.7671 |
| 0.0 | 0.7479 |

### Permutation

**Mean Similarity**: 0.7783 (±0.0209)

### Masking

**Grid**: [0.0, 0.05, 0.1, 0.15, 0.2, 0.3]

**Logit rAUC**: 0.7663

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 0.0 | 0.7615 |
| 0.05 | 0.7806 |
| 0.1 | 0.7309 |
| 0.15 | 0.7966 |
| 0.2 | 0.7547 |
| 0.3 | 0.7773 |

### Expression

**Grid**: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

**Logit rAUC**: 0.7835

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 0.0 | 0.7688 |
| 0.05 | 0.7933 |
| 0.1 | 0.7686 |
| 0.2 | 0.7923 |
| 0.3 | 0.7912 |
| 0.5 | 0.7732 |

## Interpretation Guide

- **rAUC (Reverse Area Under Curve)**: Measures output stability as perturbation increases. Higher is better (0-1 scale).
- **Cosine Similarity**: How similar outputs remain after perturbation. 1.0 = identical, 0.0 = orthogonal.
- **Robustness Score**: Aggregate mean of all rAUC values across probes.

## References

- ITU FG-AI4H DEL3: AI4H Requirement Specifications
- brainaug-lab: Reproducible EEG Augmentation Lab
