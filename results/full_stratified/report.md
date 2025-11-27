# Evaluation Report: SUITE-TOY-CLASS-dummy_classifier-20251127-064456

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
| SiteA | 0.4201 | 0.5139 | 0.5093 | 72 |
| SiteB | 0.6305 | 0.6316 | 0.6298 | 57 |
| SiteC | 0.6348 | 0.5915 | 0.5912 | 71 |

#### By Sex

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5021 | 0.5326 | 0.5326 | 92 |
| M | 0.6061 | 0.6111 | 0.6045 | 108 |

#### By Scanner

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| GE | 0.6373 | 0.6286 | 0.6274 | 70 |
| Philips | 0.4662 | 0.5205 | 0.5147 | 73 |
| Siemens | 0.5844 | 0.5789 | 0.5788 | 57 |

#### By Disease Stage

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| AD | 0.4955 | 0.5833 | 0.5804 | 60 |
| CN | 0.5559 | 0.5429 | 0.5414 | 70 |
| MCI | 0.6085 | 0.6 | 0.5987 | 70 |

#### By Age Group

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.5819 | 0.5741 | 0.5668 | 54 |
| age_40-60 | 0.481 | 0.5692 | 0.5513 | 65 |
| age_60-80 | 0.5943 | 0.5857 | 0.5788 | 70 |
| age_80-100 | 0.6 | 0.5455 | 0.5299 | 11 |

## Execution Details
- **Date**: 2025-11-27
- **Runner**: fmbench (classification)
