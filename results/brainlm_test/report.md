# Evaluation Report: SUITE-TOY-CLASS-brainlm-20251127-153750

- **Suite**: SUITE-TOY-CLASS
- **Benchmark**: BM-TOY-CLASS
- **Model**: brainlm
- **Dataset**: DS-TOY-FMRI-CLASS
- **Task**: classification

## Performance Metrics

### Overall
- **AUROC**: 0.5193
- **Accuracy**: 0.51
- **F1-Score**: 0.51

### Stratified Performance

#### By Site

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| SiteA | 0.5791 | 0.5417 | 0.5394 | 72 |
| SiteB | 0.4643 | 0.4737 | 0.4722 | 57 |
| SiteC | 0.4944 | 0.507 | 0.5035 | 71 |

#### By Sex

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5513 | 0.5217 | 0.5208 | 92 |
| M | 0.4881 | 0.5 | 0.4985 | 108 |

#### By Scanner

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| GE | 0.4087 | 0.4714 | 0.4687 | 70 |
| Philips | 0.6226 | 0.5479 | 0.5476 | 73 |
| Siemens | 0.5099 | 0.5088 | 0.5086 | 57 |

#### By Disease Stage

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| AD | 0.4649 | 0.45 | 0.4462 | 60 |
| CN | 0.4286 | 0.4571 | 0.4571 | 70 |
| MCI | 0.6372 | 0.6143 | 0.6136 | 70 |

#### By Ethnicity

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| Asian | 0.6667 | 0.5862 | 0.5817 | 29 |
| Black | 0.6165 | 0.6316 | 0.6306 | 38 |
| Hispanic | 0.463 | 0.5116 | 0.5106 | 43 |
| Other | 0.2857 | 0.3077 | 0.3077 | 13 |
| White | 0.4717 | 0.4545 | 0.4522 | 77 |

#### By Age Group

| Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.4528 | 0.4444 | 0.4444 | 54 |
| age_40-60 | 0.573 | 0.5538 | 0.543 | 65 |
| age_60-80 | 0.5357 | 0.5286 | 0.5238 | 70 |
| age_80-100 | 0.5667 | 0.4545 | 0.45 | 11 |

## Execution Details
- **Date**: 2025-11-27
- **Runner**: fmbench (classification)
