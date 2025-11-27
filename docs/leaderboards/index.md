# 🏆 Foundation Model Leaderboards

!!! success "Benchmark Hub Stats"
    🎯 **7** Benchmarks | 🤖 **5** Models Evaluated | 📊 **15** Total Evaluations

Welcome to the **AI4H-Inspired FM Benchmark Hub**! Rankings below show **all submitted models** from best to developing, helping you find the right model for your use case.

## 🧭 Quick Navigation

- [🌐 Cross-Domain](#cross-domain)
- [🧬 Genomics](#genomics)
- [🧠 Neurology](#neurology)

---

## 🌐 Cross-Domain

### 🌐 Clinical Report Generation Quality

✍️ **Task**: Generation | 🏥 **Health Topic**: Automated Clinical Reporting

!!! info "Clinical Relevance"
    Foundation models increasingly generate clinical reports, radiology  interpretations, and patient summaries. Quality metrics must capture both linguistic fluency and clinical accuracy/safety.


#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **Flamingo** 👑 | 0.8400 | mimic_cxr_reports | report_quality_score | 2024-01-20 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 Flamingo**

| Metric | Value |
|---|---|
| bertscore | 0.8700 |
| bleu | 38.5000 |
| clinical_accuracy | 0.8900 |
| finding_precision | 0.9100 |
| finding_recall | 0.8500 |
| flesch_kincaid | 10.2000 |
| hallucination_rate | 0.0600 |
| harmful_content | 0.0010 |
| meteor | 0.5400 |
| omission_rate | 0.0900 |
| report_quality_score | 0.8400 |
| rouge_l | 0.6200 |
| structure_score | 0.8800 |
| uncertainty_calibration | 0.8200 |

</details>

#### 📊 Granular Performance Breakdown

Expand sections below to see how models perform across different conditions:


<details>
<summary>📄 <strong>Flamingo</strong> by Report Type</summary>

| Report Type | clinical_accuracy | finding_recall | bertscore | N |
|---|---|---|---|---|
| 🥇 chest_xray | 0.9100 | 0.8700 | 0.8800 | 2000 |
| 🥈 brain_mri | 0.8800 | 0.8400 | 0.8600 | 600 |
| ct_abdomen | 0.8600 | 0.8200 | 0.8500 | 800 |

</details>

<details>
<summary>📊 <strong>Flamingo</strong> by Complexity</summary>

| Complexity | clinical_accuracy | hallucination_rate | N |
|---|---|---|---|
| 🥇 simple | 0.9400 | 0.0300 | 1500 |
| 🥈 moderate | 0.8800 | 0.0600 | 1200 |
| complex | 0.8200 | 0.1000 | 700 |

</details>

---
*Ranked by report_quality_score. Higher is better. Last updated from 1 evaluation(s).*

### 🌐 Foundation Model Robustness Evaluation

🛡️ **Task**: Robustness Assessment | 🏥 **Health Topic**: Model Reliability and Artifact Resilience

!!! info "Clinical Relevance"
    Clinical deployment of AI models requires robustness to real-world data variability including sensor noise, signal artifacts, and acquisition differences. This benchmark evaluates model stability under controlled perturbations that simulate common data quality issues.


#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **dummy_classifier** 👑 | 0.7810 | - | robustness_score | 2025-11-27 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 dummy_classifier**

| Metric | Value |
|---|---|
| dropout_rAUC | 0.7760 |
| line_noise_rAUC | 0.7737 |
| noise_rAUC | 0.7867 |
| perm_equivariance | 0.7819 |
| robustness_score | 0.7810 |
| shift_rAUC | 0.7874 |
| shift_sensitivity | 0.7897 |

</details>

---
*Ranked by robustness_score. Higher is better. Last updated from 5 evaluation(s).*

## 🧬 Genomics

### 🧬 Cell Type Annotation

🎯 **Task**: Classification | 🏥 **Health Topic**: Single-cell Transcriptomics

*Predicting cell types from single-cell RNA-seq data.*

!!! info "Clinical Relevance"
    Automated characterization of immune cell populations.

#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **Geneformer** 👑 | 0.9100 | PBMC 68k | Accuracy | 2023-11-01 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 Geneformer**

| Metric | Value |
|---|---|
| Accuracy | 0.9100 |
| F1-Score | 0.8500 |

</details>

---
*Ranked by Accuracy. Higher is better. Last updated from 1 evaluation(s).*

## 🧠 Neurology

### 🧠 Alzheimer's Disease Classification using Brain MRI

🎯 **Task**: Classification | 🏥 **Health Topic**: Alzheimer's Disease

*Binary classification of AD vs CN using structural MRI data.*

!!! info "Clinical Relevance"
    Automated screening for AD to assist radiological workflow.

#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **UNI** 👑 | 0.9200 | Alzheimer's Disease Neuroimaging Initiative (ADNI) | AUROC | 2023-10-27 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 UNI**

| Metric | Value |
|---|---|
| AUROC | 0.9200 |
| Accuracy | 0.8800 |

</details>

---
*Ranked by AUROC. Higher is better. Last updated from 1 evaluation(s).*

### 🧠 Brain Time-Series Modeling

🔄 **Task**: Reconstruction | 🏥 **Health Topic**: Functional Brain Connectivity

*Evaluating ability to reconstruct masked fMRI voxel time-series.*

!!! info "Clinical Relevance"
    Foundation for understanding functional connectivity patterns.

#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **BrainLM** 👑 | 0.7800 | UK Biobank fMRI tensors | Correlation | 2025-11-15 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 BrainLM**

| Metric | Value |
|---|---|
| Correlation | 0.7800 |
| MSE | 0.4500 |

</details>

---
*Ranked by Correlation. Higher is better. Last updated from 1 evaluation(s).*

### 🧠 Toy Classification Benchmark

🎯 **Task**: Classification | 🏥 **Health Topic**: N/A

*A toy benchmark for testing the pipeline.*

#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **dummy_classifier** 👑 | 0.5597 | Toy fMRI Classification | AUROC | 2025-11-27 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 dummy_classifier**

| Metric | Value |
|---|---|
| AUROC | 0.5597 |
| Accuracy | 0.5750 |
| F1-Score | 0.5732 |

</details>

#### 📊 Granular Performance Breakdown

Expand sections below to see how models perform across different conditions:


<details>
<summary>🔬 <strong>dummy_classifier</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 GE | 0.6373 | 0.6286 | 0.6274 | 70 |
| 🥈 Siemens | 0.5844 | 0.5789 | 0.5788 | 57 |
| Philips | 0.4662 | 0.5205 | 0.5147 | 73 |

</details>

<details>
<summary>🏥 <strong>dummy_classifier</strong> by Site</summary>

| Site | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 SiteC | 0.6348 | 0.5915 | 0.5912 | 71 |
| 🥈 SiteB | 0.6305 | 0.6316 | 0.6298 | 57 |
| SiteA | 0.4201 | 0.5139 | 0.5093 | 72 |

</details>

<details>
<summary>🩺 <strong>dummy_classifier</strong> by Disease Stage</summary>

| Disease Stage | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 MCI | 0.6085 | 0.6000 | 0.5987 | 70 |
| 🥈 CN | 0.5559 | 0.5429 | 0.5414 | 70 |
| AD | 0.4955 | 0.5833 | 0.5804 | 60 |

</details>

<details>
<summary>👤 <strong>dummy_classifier</strong> by Sex</summary>

| Sex | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 M | 0.6061 | 0.6111 | 0.6045 | 108 |
| F | 0.5021 | 0.5326 | 0.5326 | 92 |

</details>

<details>
<summary>📅 <strong>dummy_classifier</strong> by Age Group</summary>

| Age Group | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 age_80-100 | 0.6000 | 0.5455 | 0.5299 | 11 |
| 🥈 age_60-80 | 0.5943 | 0.5857 | 0.5788 | 70 |
| 🥉 age_20-40 | 0.5819 | 0.5741 | 0.5668 | 54 |
| age_40-60 | 0.4810 | 0.5692 | 0.5513 | 65 |

</details>

---
*Ranked by AUROC. Higher is better. Last updated from 5 evaluation(s).*

### 🧠 fMRI Foundation Model Benchmark (Granular)

📋 **Task**: Classification/Reconstruction | 🏥 **Health Topic**: Functional Brain Imaging Analysis

!!! info "Clinical Relevance"
    Foundation models for fMRI must generalize across diverse acquisition  parameters, scanner manufacturers, and preprocessing pipelines. This benchmark provides granular rankings to identify optimal model-data matches.


#### 🏆 Leaderboard

| Rank | Model | Score | Dataset | Details |
| :---: | :--- | :---: | :--- | :--- |
| 🥇 | **BrainLM** 👑 | 0.9100 | hcp_1200 | AUROC | 2024-01-15 |

<details>
<summary>📋 <strong>Full Metrics for All Models</strong></summary>

**🥇 BrainLM**

| Metric | Value |
|---|---|
| AUROC | 0.9100 |
| Accuracy | 0.8700 |
| Correlation | 0.8100 |
| F1-Score | 0.8600 |
| MSE | 0.4200 |

</details>

#### 📊 Granular Performance Breakdown

Expand sections below to see how models perform across different conditions:


<details>
<summary>🔬 <strong>BrainLM</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 Siemens | 0.9300 | 0.8900 | 0.8800 | 450 |
| 🥈 Philips | 0.9000 | 0.8600 | 0.8500 | 370 |
| GE | 0.8800 | 0.8400 | 0.8300 | 380 |

</details>

<details>
<summary>🏥 <strong>BrainLM</strong> by Site</summary>

| Site | AUROC | Accuracy | N |
|---|---|---|---|
| 🥇 WashU | 0.9300 | 0.8900 | 220 |
| 🥈 MGH | 0.9200 | 0.8800 | 200 |
| 🥉 Oxford | 0.9100 | 0.8700 | 200 |
| UCLA | 0.9000 | 0.8600 | 180 |
| UMinn | 0.8900 | 0.8500 | 200 |

</details>

<details>
<summary>📡 <strong>BrainLM</strong> by Acquisition Type</summary>

| Acquisition Type | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 resting_state | 0.9200 | 0.8800 | 0.8700 | 600 |
| 🥈 language | 0.9100 | 0.8700 | - | 100 |
| 🥉 working_memory | 0.9000 | 0.8600 | - | 150 |
| task_based | 0.8900 | 0.8500 | 0.8400 | 400 |
| motor | 0.8800 | 0.8400 | - | 150 |

</details>

<details>
<summary>⚙️ <strong>BrainLM</strong> by Preprocessing</summary>

| Preprocessing | AUROC | Accuracy | N |
|---|---|---|---|
| 🥇 fmriprep | 0.9200 | 0.8800 | 500 |
| 🥈 hcp | 0.9100 | 0.8700 | 400 |
| minimal | 0.8500 | 0.8100 | 300 |

</details>

<details>
<summary>🧲 <strong>BrainLM</strong> by Field Strength</summary>

| Field Strength | AUROC | Accuracy | N |
|---|---|---|---|
| 🥇 7T | 0.9400 | 0.9100 | 100 |
| 3T | 0.9100 | 0.8700 | 900 |

</details>

---
*Ranked by AUROC. Higher is better. Last updated from 1 evaluation(s).*

---

## 🚀 Submit Your Model

Want to see your Foundation Model on these leaderboards?

1. 📥 **Download** the benchmark suite: `pip install -e .`
2. 🧪 **Run** evaluations: `python -m fmbench run-robustness --help`
3. 📤 **Submit** via Pull Request - see [Submission Guide](../contributing/submission_guide.md)

*Aligned with [ITU/WHO FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards for healthcare AI evaluation.*
