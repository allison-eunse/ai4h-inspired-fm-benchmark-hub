# Benchmark Leaderboards

Automated leaderboards generated from repository metadata, aligned with **ITU FG-AI4H** standards.

## Cross-Domain

### Foundation Model Robustness Evaluation

**Health Topic**: Model Reliability and Artifact Resilience | **AI Task**: Robustness Assessment

**

**Clinical Relevance**: Clinical deployment of AI models requires robustness to real-world data variability including sensor noise, signal artifacts, and acquisition differences. This benchmark evaluates model stability under controlled perturbations that simulate common data quality issues.


| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| dummy_classifier (candidate) | - | **robustness_score**: 0.4554<br>**dropout_rAUC**: 0.5186<br>**line_noise_rAUC**: 0.4901<br>**noise_rAUC**: 0.4003<br>**perm_equivariance**: 0.7819<br>**shift_rAUC**: 0.4126 | Completed | 2025-11-27 |

## Genomics

### Cell Type Annotation

**Health Topic**: Single-cell Transcriptomics | **AI Task**: Classification

*Predicting cell types from single-cell RNA-seq data.*

**Clinical Relevance**: Automated characterization of immune cell populations.

| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| Geneformer (genetics_fm) | PBMC 68k | **Accuracy**: 0.91<br>**F1-Score**: 0.85 | Completed | 2023-11-01 |

## Neurology

### Alzheimer's Disease Classification using Brain MRI

**Health Topic**: Alzheimer's Disease | **AI Task**: Classification

*Binary classification of AD vs CN using structural MRI data.*

**Clinical Relevance**: Automated screening for AD to assist radiological workflow.

| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| UNI (brain_fm) | Alzheimer's Disease Neuroimaging Initiative (ADNI) | **AUROC**: 0.92<br>**Accuracy**: 0.88 | Completed | 2023-10-27 |

### Brain Time-Series Modeling

**Health Topic**: Functional Brain Connectivity | **AI Task**: Reconstruction

*Evaluating ability to reconstruct masked fMRI voxel time-series.*

**Clinical Relevance**: Foundation for understanding functional connectivity patterns.

| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| BrainLM (brain_fm) | UK Biobank fMRI tensors | **MSE**: 0.45<br>**Correlation**: 0.78 | Completed | 2025-11-15 |

### Toy Classification Benchmark

**Health Topic**: N/A | **AI Task**: Classification

*A toy benchmark for testing the pipeline.*

| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| dummy_classifier (candidate) | Toy fMRI Classification | **AUROC**: 0.5597<br>**Accuracy**: 0.575<br>**F1-Score**: 0.5732 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | Toy fMRI Classification | **AUROC**: 0.5597<br>**Accuracy**: 0.575<br>**F1-Score**: 0.5732 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | Toy fMRI Classification | **AUROC**: 0.5597<br>**Accuracy**: 0.575<br>**F1-Score**: 0.5732 | Completed | 2025-11-27 |

