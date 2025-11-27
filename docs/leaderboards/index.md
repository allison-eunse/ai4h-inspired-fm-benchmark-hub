# Benchmark Leaderboards

Automated leaderboards generated from repository metadata, aligned with **ITU FG-AI4H** standards.

## Cross-Domain

### Clinical Report Generation Quality

**Health Topic**: Automated Clinical Reporting | **AI Task**: Generation

**

**Clinical Relevance**: Foundation models increasingly generate clinical reports, radiology  interpretations, and patient summaries. Quality metrics must capture both linguistic fluency and clinical accuracy/safety.


| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| Flamingo (candidate) | mimic_cxr_reports | **clinical_accuracy**: 0.89<br>**finding_recall**: 0.85<br>**bertscore**: 0.87<br>**bleu**: 38.5<br>**finding_precision**: 0.91<br>**flesch_kincaid**: 10.2 | Completed | 2024-01-20 |

### Foundation Model Robustness Evaluation

**Health Topic**: Model Reliability and Artifact Resilience | **AI Task**: Robustness Assessment

**

**Clinical Relevance**: Clinical deployment of AI models requires robustness to real-world data variability including sensor noise, signal artifacts, and acquisition differences. This benchmark evaluates model stability under controlled perturbations that simulate common data quality issues.


| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| dummy_classifier (candidate) | - | **robustness_score**: 0.7749<br>**dropout_rAUC**: 0.7758<br>**expression_rAUC**: 0.7835<br>**masking_rAUC**: 0.7663<br>**noise_rAUC**: 0.774<br>**perm_equivariance**: 0.7783 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | - | **robustness_score**: 0.781<br>**dropout_rAUC**: 0.776<br>**line_noise_rAUC**: 0.7737<br>**noise_rAUC**: 0.7867<br>**perm_equivariance**: 0.7819<br>**shift_rAUC**: 0.7874 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | - | **robustness_score**: 0.4554<br>**dropout_rAUC**: 0.5186<br>**line_noise_rAUC**: 0.4901<br>**noise_rAUC**: 0.4003<br>**perm_equivariance**: 0.7819<br>**shift_rAUC**: 0.4126 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | - | **robustness_score**: 0.781<br>**dropout_rAUC**: 0.776<br>**line_noise_rAUC**: 0.7737<br>**noise_rAUC**: 0.7867<br>**perm_equivariance**: 0.7819<br>**shift_rAUC**: 0.7874 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | - | **robustness_score**: 0.781<br>**dropout_rAUC**: 0.776<br>**line_noise_rAUC**: 0.7737<br>**noise_rAUC**: 0.7867<br>**perm_equivariance**: 0.7819<br>**shift_rAUC**: 0.7874 | Completed | 2025-11-27 |

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
| dummy_classifier (candidate) | Toy fMRI Classification | **AUROC**: 0.5597<br>**Accuracy**: 0.575<br>**F1-Score**: 0.5732 | Completed | 2025-11-27 |
| dummy_classifier (candidate) | Toy fMRI Classification | **AUROC**: 0.5597<br>**Accuracy**: 0.575<br>**F1-Score**: 0.5732 | Completed | 2025-11-27 |

#### Granular Performance Breakdown


<details>
<summary>📊 <strong>dummy_classifier</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| GE | 0.6373 | 0.6286 | 0.6274 | 70 |
| Philips | 0.4662 | 0.5205 | 0.5147 | 73 |
| Siemens | 0.5844 | 0.5789 | 0.5788 | 57 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Site</summary>

| Site | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| SiteA | 0.4201 | 0.5139 | 0.5093 | 72 |
| SiteB | 0.6305 | 0.6316 | 0.6298 | 57 |
| SiteC | 0.6348 | 0.5915 | 0.5912 | 71 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Disease Stage</summary>

| Disease Stage | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| AD | 0.4955 | 0.5833 | 0.5804 | 60 |
| CN | 0.5559 | 0.5429 | 0.5414 | 70 |
| MCI | 0.6085 | 0.6 | 0.5987 | 70 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Sex</summary>

| Sex | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5021 | 0.5326 | 0.5326 | 92 |
| M | 0.6061 | 0.6111 | 0.6045 | 108 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Age Group</summary>

| Age Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.5819 | 0.5741 | 0.5668 | 54 |
| age_40-60 | 0.481 | 0.5692 | 0.5513 | 65 |
| age_60-80 | 0.5943 | 0.5857 | 0.5788 | 70 |
| age_80-100 | 0.6 | 0.5455 | 0.5299 | 11 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Site</summary>

| Site | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| SiteA | 0.5663 | 0.5714 | 0.5675 | 63 |
| SiteB | 0.578 | 0.6286 | 0.6081 | 70 |
| SiteC | 0.5107 | 0.5224 | 0.5214 | 67 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Sex</summary>

| Sex | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5487 | 0.5577 | 0.5562 | 104 |
| M | 0.5545 | 0.5938 | 0.5772 | 96 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Age Group</summary>

| Age Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.5679 | 0.5862 | 0.5735 | 58 |
| age_40-60 | 0.5209 | 0.5172 | 0.5149 | 58 |
| age_60-80 | 0.5846 | 0.6094 | 0.6093 | 64 |
| age_80-100 | 0.61 | 0.6 | 0.596 | 20 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| GE | 0.6373 | 0.6286 | 0.6274 | 70 |
| Philips | 0.4662 | 0.5205 | 0.5147 | 73 |
| Siemens | 0.5844 | 0.5789 | 0.5788 | 57 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Site</summary>

| Site | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| SiteA | 0.4201 | 0.5139 | 0.5093 | 72 |
| SiteB | 0.6305 | 0.6316 | 0.6298 | 57 |
| SiteC | 0.6348 | 0.5915 | 0.5912 | 71 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Disease Stage</summary>

| Disease Stage | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| AD | 0.4955 | 0.5833 | 0.5804 | 60 |
| CN | 0.5559 | 0.5429 | 0.5414 | 70 |
| MCI | 0.6085 | 0.6 | 0.5987 | 70 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Sex</summary>

| Sex | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5021 | 0.5326 | 0.5326 | 92 |
| M | 0.6061 | 0.6111 | 0.6045 | 108 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Age Group</summary>

| Age Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.5819 | 0.5741 | 0.5668 | 54 |
| age_40-60 | 0.481 | 0.5692 | 0.5513 | 65 |
| age_60-80 | 0.5943 | 0.5857 | 0.5788 | 70 |
| age_80-100 | 0.6 | 0.5455 | 0.5299 | 11 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Ethnicity</summary>

| Ethnicity | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| Asian | 0.3952 | 0.5172 | 0.5167 | 29 |
| Black | 0.5625 | 0.5789 | 0.5682 | 38 |
| Hispanic | 0.5457 | 0.5349 | 0.5346 | 43 |
| Other | 0.4524 | 0.5385 | 0.5125 | 13 |
| White | 0.6296 | 0.6234 | 0.6224 | 77 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| GE | 0.6373 | 0.6286 | 0.6274 | 70 |
| Philips | 0.4662 | 0.5205 | 0.5147 | 73 |
| Siemens | 0.5844 | 0.5789 | 0.5788 | 57 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Site</summary>

| Site | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| SiteA | 0.4201 | 0.5139 | 0.5093 | 72 |
| SiteB | 0.6305 | 0.6316 | 0.6298 | 57 |
| SiteC | 0.6348 | 0.5915 | 0.5912 | 71 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Disease Stage</summary>

| Disease Stage | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| AD | 0.4955 | 0.5833 | 0.5804 | 60 |
| CN | 0.5559 | 0.5429 | 0.5414 | 70 |
| MCI | 0.6085 | 0.6 | 0.5987 | 70 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Sex</summary>

| Sex | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| F | 0.5021 | 0.5326 | 0.5326 | 92 |
| M | 0.6061 | 0.6111 | 0.6045 | 108 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Age Group</summary>

| Age Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| age_20-40 | 0.5819 | 0.5741 | 0.5668 | 54 |
| age_40-60 | 0.481 | 0.5692 | 0.5513 | 65 |
| age_60-80 | 0.5943 | 0.5857 | 0.5788 | 70 |
| age_80-100 | 0.6 | 0.5455 | 0.5299 | 11 |

</details>

<details>
<summary>📊 <strong>dummy_classifier</strong> by Ethnicity</summary>

| Ethnicity | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| Asian | 0.3952 | 0.5172 | 0.5167 | 29 |
| Black | 0.5625 | 0.5789 | 0.5682 | 38 |
| Hispanic | 0.5457 | 0.5349 | 0.5346 | 43 |
| Other | 0.4524 | 0.5385 | 0.5125 | 13 |
| White | 0.6296 | 0.6234 | 0.6224 | 77 |

</details>

### fMRI Foundation Model Benchmark (Granular)

**Health Topic**: Functional Brain Imaging Analysis | **AI Task**: Classification/Reconstruction

**

**Clinical Relevance**: Foundation models for fMRI must generalize across diverse acquisition  parameters, scanner manufacturers, and preprocessing pipelines. This benchmark provides granular rankings to identify optimal model-data matches.


| Model | Dataset | Metrics | Status | Date |
| :--- | :--- | :--- | :--- | :--- |
| BrainLM (candidate) | hcp_1200 | **AUROC**: 0.91<br>**Accuracy**: 0.87<br>**F1-Score**: 0.86<br>**MSE**: 0.42<br>**Correlation**: 0.81 | Completed | 2024-01-15 |

#### Granular Performance Breakdown


<details>
<summary>📊 <strong>BrainLM</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| GE | 0.88 | 0.84 | 0.83 | 380 |
| Philips | 0.9 | 0.86 | 0.85 | 370 |
| Siemens | 0.93 | 0.89 | 0.88 | 450 |

</details>

<details>
<summary>📊 <strong>BrainLM</strong> by Site</summary>

| Site | AUROC | Accuracy | N |
|---|---|---|---|
| MGH | 0.92 | 0.88 | 200 |
| Oxford | 0.91 | 0.87 | 200 |
| UCLA | 0.9 | 0.86 | 180 |
| UMinn | 0.89 | 0.85 | 200 |
| WashU | 0.93 | 0.89 | 220 |

</details>

<details>
<summary>📊 <strong>BrainLM</strong> by Acquisition Type</summary>

| Acquisition Type | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| language | 0.91 | 0.87 | - | 100 |
| motor | 0.88 | 0.84 | - | 150 |
| resting_state | 0.92 | 0.88 | 0.87 | 600 |
| task_based | 0.89 | 0.85 | 0.84 | 400 |
| working_memory | 0.9 | 0.86 | - | 150 |

</details>

<details>
<summary>📊 <strong>BrainLM</strong> by Preprocessing</summary>

| Preprocessing | AUROC | Accuracy | N |
|---|---|---|---|
| fmriprep | 0.92 | 0.88 | 500 |
| hcp | 0.91 | 0.87 | 400 |
| minimal | 0.85 | 0.81 | 300 |

</details>

