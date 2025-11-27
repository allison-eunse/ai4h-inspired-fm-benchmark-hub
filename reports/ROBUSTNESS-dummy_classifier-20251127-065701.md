# Robustness Evaluation Report: ROBUSTNESS-dummy_classifier-20251127-065701

## Overview

- **Model**: dummy_classifier
- **Benchmark**: Robustness Testing (ITU AI4H DEL3 Aligned)
- **Date**: 2025-11-27 06:57 UTC

## Aggregate Robustness Scores

| Metric | Score |
|--------|-------|
| **dropout_rAUC** | 0.7760 |
| **line_noise_rAUC** | 0.7737 |
| **noise_rAUC** | 0.7867 |
| **perm_equivariance** | 0.7819 |
| **robustness_score** | 0.7810 |
| **shift_rAUC** | 0.7874 |
| **shift_sensitivity** | 0.7897 |

## Probe Results

### Dropout

**Grid**: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

**Logit rAUC**: 0.7760

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 0.0 | 0.7984 |
| 0.05 | 0.7683 |
| 0.1 | 0.7519 |
| 0.2 | 0.7972 |
| 0.3 | 0.7663 |
| 0.5 | 0.7858 |

### Noise

**Grid**: [40.0, 30.0, 20.0, 10.0, 5.0, 0.0]

**Logit rAUC**: 0.7867

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 40.0 | 0.8163 |
| 30.0 | 0.7794 |
| 20.0 | 0.7670 |
| 10.0 | 0.7865 |
| 5.0 | 0.8143 |
| 0.0 | 0.7815 |

### Line Noise

**Grid**: [40.0, 30.0, 20.0, 10.0, 5.0]

**Logit rAUC**: 0.7737

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 40.0 | 0.7845 |
| 30.0 | 0.7791 |
| 20.0 | 0.7633 |
| 10.0 | 0.7741 |
| 5.0 | 0.7710 |

### Permutation

**Mean Similarity**: 0.7819 (±0.0218)

### Shift

**Logit rAUC**: 0.7874

**Mean Similarity**: 0.7897 (±0.0000)

## Interpretation Guide

- **rAUC (Reverse Area Under Curve)**: Measures output stability as perturbation increases. Higher is better (0-1 scale).
- **Cosine Similarity**: How similar outputs remain after perturbation. 1.0 = identical, 0.0 = orthogonal.
- **Robustness Score**: Aggregate mean of all rAUC values across probes.

## References

- ITU FG-AI4H DEL3: AI4H Requirement Specifications
- brainaug-lab: Reproducible EEG Augmentation Lab
