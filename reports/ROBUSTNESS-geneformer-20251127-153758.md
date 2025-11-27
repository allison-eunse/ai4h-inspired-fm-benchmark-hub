# Robustness Evaluation Report: ROBUSTNESS-geneformer-20251127-153758

## Overview

- **Model**: geneformer
- **Benchmark**: Robustness Testing (ITU AI4H DEL3 Aligned)
- **Date**: 2025-11-27 15:37 UTC

## Aggregate Robustness Scores

| Metric | Score |
|--------|-------|
| **dropout_rAUC** | 0.9995 |
| **line_noise_rAUC** | 0.9995 |
| **noise_rAUC** | 0.9995 |
| **perm_equivariance** | 0.9995 |
| **robustness_score** | 0.9995 |
| **shift_rAUC** | 0.9995 |
| **shift_sensitivity** | 0.9995 |

## Probe Results

### Dropout

**Grid**: [0.0, 0.05, 0.1, 0.2, 0.3, 0.5]

**Logit rAUC**: 0.9995

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 0.0 | 0.9996 |
| 0.05 | 0.9994 |
| 0.1 | 0.9995 |
| 0.2 | 0.9995 |
| 0.3 | 0.9995 |
| 0.5 | 0.9995 |

### Noise

**Grid**: [40.0, 30.0, 20.0, 10.0, 5.0, 0.0]

**Logit rAUC**: 0.9995

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 40.0 | 0.9997 |
| 30.0 | 0.9995 |
| 20.0 | 0.9995 |
| 10.0 | 0.9995 |
| 5.0 | 0.9995 |
| 0.0 | 0.9994 |

### Line Noise

**Grid**: [40.0, 30.0, 20.0, 10.0, 5.0]

**Logit rAUC**: 0.9995

| Perturbation | Cosine Similarity |
|--------------|------------------|
| 40.0 | 0.9996 |
| 30.0 | 0.9995 |
| 20.0 | 0.9994 |
| 10.0 | 0.9995 |
| 5.0 | 0.9995 |

### Permutation

**Mean Similarity**: 0.9995 (±0.0001)

### Shift

**Logit rAUC**: 0.9995

**Mean Similarity**: 0.9995 (±0.0000)

## Interpretation Guide

- **rAUC (Reverse Area Under Curve)**: Measures output stability as perturbation increases. Higher is better (0-1 scale).
- **Cosine Similarity**: How similar outputs remain after perturbation. 1.0 = identical, 0.0 = orthogonal.
- **Robustness Score**: Aggregate mean of all rAUC values across probes.

## References

- ITU FG-AI4H DEL3: AI4H Requirement Specifications
- brainaug-lab: Reproducible EEG Augmentation Lab
