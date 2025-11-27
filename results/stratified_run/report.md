# Evaluation Report: SUITE-TOY-CLASS-dummy_classifier-20251127-064428

- **Suite**: SUITE-TOY-CLASS
- **Benchmark**: BM-TOY-CLASS
- **Model**: dummy_classifier
- **Dataset**: DS-TOY-FMRI-CLASS
- **Task**: classification

## Performance Metrics

### Overall
- **AUROC**: 0.5597
- **Accuracy**: 0.575
- **F1-Score**: 0.5732

### Stratified Performance

#### By Site

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| SiteA | 0.5663 | 0.5714 | 0.5675 | 63 |
| SiteB | 0.578 | 0.6286 | 0.6081 | 70 |
| SiteC | 0.5107 | 0.5224 | 0.5214 | 67 |

#### By Sex

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5487 | 0.5577 | 0.5562 | 104 |
| M | 0.5545 | 0.5938 | 0.5772 | 96 |

#### By Age Group

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.5679 | 0.5862 | 0.5735 | 58 |
| age_40-60 | 0.5209 | 0.5172 | 0.5149 | 58 |
| age_60-80 | 0.5846 | 0.6094 | 0.6093 | 64 |
| age_80-100 | 0.61 | 0.6 | 0.596 | 20 |

## Execution Details
- **Date**: 2025-11-27
- **Runner**: fmbench (classification)
