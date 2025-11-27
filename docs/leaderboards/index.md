# 🏆 Foundation Model Leaderboards

!!! success "Benchmark Hub Stats"
    🎯 **7** Benchmarks | 🤖 **9** Models Evaluated | 📊 **19** Total Evaluations

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

```
         🥇          
     [Flamingo]    
    ┌─────────┐     
 🥈 │         │ 🥉  
[Med-Flamin]│         │[RadBERT]
────┴─────────┴────
```

**All 3 models ranked by report_quality_score:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **Flamingo** 👑 | 0.8400 |  | mimic_cxr_reports | 2024-01-20 |
| 🥈 | **Med-Flamingo** 🌟 | 0.7800 |  | mimic_cxr_reports | 2024-01-18 |
| 🥉 | **RadBERT** ✨ | 0.6900 |  | mimic_cxr_reports | 2024-01-12 |


#### 📖 Ranking Explanation

!!! abstract "Why These Rankings?"
    **🥇 Flamingo** leads with report_quality_score=0.8400

    - Gap to 🥈 **Med-Flamingo**: +0.0600 (7.7% better)
    - Score range across all models: 0.1500


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `report_quality_score`**

> Composite score of linguistic fluency + clinical accuracy (0.0-1.0)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **generation tasks**, we prioritize:
1. `report_quality_score` – composite clinical + linguistic quality
2. `clinical_accuracy` – correctness of medical content
3. `bertscore` – semantic similarity
4. `hallucination_rate` – safety-critical (lower is better)

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | report_quality_score | clinical_accuracy | bertscore | bleu | finding_recall | hallucination_rate | finding_precision | flesch_kincaid |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇  | Flamingo | **0.8400** | 0.8900 | 0.8700 | 38.5000 | 0.8500 | 0.0600 | 0.9100 | 10.2000 |
| 🥈  | Med-Flamingo | **0.7800** | 0.8200 | 0.8200 | 32.5000 | 0.7900 | 0.0900 | 0.8500 | 11.5000 |
| 🥉  | RadBERT | **0.6900** | 0.7200 | 0.7400 | 24.2000 | 0.6800 | 0.1500 | 0.7500 | 13.2000 |

!!! tip "Legend"
    📊 **Primary metric**: report_quality_score (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

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

<details>
<summary>📄 <strong>Med-Flamingo</strong> by Report Type</summary>

| Report Type | clinical_accuracy | finding_recall | bertscore | N |
|---|---|---|---|---|
| 🥇 chest_xray | 0.8500 | 0.8100 | 0.8300 | 2000 |
| 🥈 brain_mri | 0.8000 | 0.7600 | 0.8000 | 600 |
| ct_abdomen | 0.7800 | 0.7400 | 0.7900 | 800 |

</details>

---
*Ranked by **report_quality_score** (higher is better). Last updated from 3 evaluation(s).*

### 🌐 Foundation Model Robustness Evaluation

🛡️ **Task**: Robustness Assessment | 🏥 **Health Topic**: Model Reliability and Artifact Resilience

!!! info "Clinical Relevance"
    Clinical deployment of AI models requires robustness to real-world data variability including sensor noise, signal artifacts, and acquisition differences. This benchmark evaluates model stability under controlled perturbations that simulate common data quality issues.


#### 🏆 Leaderboard

**All 1 models ranked by robustness_score:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **dummy_classifier** 👑 | 0.7810 | 🔶 Fair | - | 2025-11-27 |


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `robustness_score`**

> Average performance retention under data perturbations (0.0-1.0)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **robustness testing**, we prioritize:
1. `robustness_score` – overall perturbation resilience
2. Individual probe scores (dropout, noise, shift, etc.)
3. `perm_equivariance` – consistency under input reordering

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | robustness_score | dropout_rAUC | line_noise_rAUC | noise_rAUC | perm_equivariance | shift_rAUC | shift_sensitivity |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 🥇 🔶 | dummy_classifier | **0.7810** | 0.7760 | 0.7737 | 0.7867 | 0.7819 | 0.7874 | 0.7897 |

!!! tip "Legend"
    📊 **Primary metric**: robustness_score (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

---
*Ranked by **robustness_score** (higher is better). Last updated from 5 evaluation(s).*

## 🧬 Genomics

### 🧬 Cell Type Annotation

🎯 **Task**: Classification | 🏥 **Health Topic**: Single-cell Transcriptomics

*Predicting cell types from single-cell RNA-seq data.*

!!! info "Clinical Relevance"
    Automated characterization of immune cell populations.

#### 🏆 Leaderboard

**All 1 models ranked by Accuracy:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **Geneformer** 👑 | 0.9100 | ⭐ Excellent | PBMC 68k | 2023-11-01 |


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `Accuracy`**

> Proportion of correct predictions (0.0-1.0)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **classification/regression tasks**, we prioritize:
1. `AUROC` – best for imbalanced medical data (DEL3 recommended)
2. `Accuracy` – overall correctness rate
3. `F1-Score` – precision-recall balance
4. `Sensitivity/Specificity` – for diagnostic screening

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | Accuracy | F1-Score |
|:---:|:---|:---:|:---:|
| 🥇 ⭐ | Geneformer | **0.9100** | 0.8500 |

!!! tip "Legend"
    📊 **Primary metric**: Accuracy (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

---
*Ranked by **Accuracy** (higher is better). Last updated from 1 evaluation(s).*

## 🧠 Neurology

### 🧠 Alzheimer's Disease Classification using Brain MRI

🎯 **Task**: Classification | 🏥 **Health Topic**: Alzheimer's Disease

*Binary classification of AD vs CN using structural MRI data.*

!!! info "Clinical Relevance"
    Automated screening for AD to assist radiological workflow.

#### 🏆 Leaderboard

**All 1 models ranked by AUROC:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **UNI** 👑 | 0.9200 | ⭐ Excellent | Alzheimer's Disease Neuroimaging Initiative (ADNI) | 2023-10-27 |


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `AUROC`**

> Area Under ROC Curve - measures discrimination ability (0.5 = random, 1.0 = perfect)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **classification/regression tasks**, we prioritize:
1. `AUROC` – best for imbalanced medical data (DEL3 recommended)
2. `Accuracy` – overall correctness rate
3. `F1-Score` – precision-recall balance
4. `Sensitivity/Specificity` – for diagnostic screening

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | AUROC | Accuracy |
|:---:|:---|:---:|:---:|
| 🥇 ⭐ | UNI | **0.9200** | 0.8800 |

!!! tip "Legend"
    📊 **Primary metric**: AUROC (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

---
*Ranked by **AUROC** (higher is better). Last updated from 1 evaluation(s).*

### 🧠 Brain Time-Series Modeling

🔄 **Task**: Reconstruction | 🏥 **Health Topic**: Functional Brain Connectivity

*Evaluating ability to reconstruct masked fMRI voxel time-series.*

!!! info "Clinical Relevance"
    Foundation for understanding functional connectivity patterns.

#### 🏆 Leaderboard

**All 1 models ranked by Correlation:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **BrainLM** 👑 | 0.7800 |  | UK Biobank fMRI tensors | 2025-11-15 |


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `Correlation`**

> Pearson correlation between predicted and actual values (-1 to 1)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **classification/regression tasks**, we prioritize:
1. `AUROC` – best for imbalanced medical data (DEL3 recommended)
2. `Accuracy` – overall correctness rate
3. `F1-Score` – precision-recall balance
4. `Sensitivity/Specificity` – for diagnostic screening

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | Correlation | MSE |
|:---:|:---|:---:|:---:|
| 🥇  | BrainLM | **0.7800** | 0.4500 |

!!! tip "Legend"
    📊 **Primary metric**: Correlation (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

---
*Ranked by **Correlation** (higher is better). Last updated from 1 evaluation(s).*

### 🧠 Toy Classification Benchmark

🎯 **Task**: Classification | 🏥 **Health Topic**: N/A

*A toy benchmark for testing the pipeline.*

#### 🏆 Leaderboard

**All 1 models ranked by AUROC:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **dummy_classifier** 👑 | 0.5597 | 📈 Developing | Toy fMRI Classification | 2025-11-27 |


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `AUROC`**

> Area Under ROC Curve - measures discrimination ability (0.5 = random, 1.0 = perfect)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **classification/regression tasks**, we prioritize:
1. `AUROC` – best for imbalanced medical data (DEL3 recommended)
2. `Accuracy` – overall correctness rate
3. `F1-Score` – precision-recall balance
4. `Sensitivity/Specificity` – for diagnostic screening

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | AUROC | Accuracy | F1-Score |
|:---:|:---|:---:|:---:|:---:|
| 🥇 📈 | dummy_classifier | **0.5597** | 0.5750 | 0.5732 |

!!! tip "Legend"
    📊 **Primary metric**: AUROC (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

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
*Ranked by **AUROC** (higher is better). Last updated from 5 evaluation(s).*

### 🧠 fMRI Foundation Model Benchmark (Granular)

📋 **Task**: Classification/Reconstruction | 🏥 **Health Topic**: Functional Brain Imaging Analysis

!!! info "Clinical Relevance"
    Foundation models for fMRI must generalize across diverse acquisition  parameters, scanner manufacturers, and preprocessing pipelines. This benchmark provides granular rankings to identify optimal model-data matches.


#### 🏆 Leaderboard

```
         🥇          
     [BrainLM]    
    ┌─────────┐     
 🥈 │         │ 🥉  
[BrainBERT]│         │[NeuroCLIP]
────┴─────────┴────
```

**All 3 models ranked by AUROC:**

| Rank | Model | Score | Performance | Dataset | Date |
| :---: | :--- | :---: | :---: | :--- | :---: |
| 🥇 | **BrainLM** 👑 | 0.9100 | ⭐ Excellent | hcp_1200 | 2024-01-15 |
| 🥈 | **BrainBERT** 🌟 | 0.8700 | ✅ Good | hcp_1200 | 2024-01-10 |
| 🥉 | **NeuroCLIP** ✨ | 0.8300 | ✅ Good | hcp_1200 | 2024-01-05 |


#### 📖 Ranking Explanation

!!! abstract "Why These Rankings?"
    **🥇 BrainLM** leads with AUROC=0.9100

    - Gap to 🥈 **BrainBERT**: +0.0400 (4.6% better)
    - Score range across all models: 0.0800
    - Performance distribution: ⭐ 1 excellent, ✅ 2 good


#### 📐 Scoring Methodology

<details>
<summary>🔍 <strong>How are models scored? (ITU/WHO AI4H Aligned)</strong></summary>

!!! note "ITU/WHO FG-AI4H Alignment"
    This evaluation framework follows [ITU-T FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards:

    - **DEL3**: Performance metrics per System Requirement Specifications (SyRS)
    - **DEL0.1**: Standardized terminology (AI Solution, Benchmarking Run)
    - **DEL10.x**: Topic Description Documents for health domains

**Primary Ranking Metric: `AUROC`**

> Area Under ROC Curve - measures discrimination ability (0.5 = random, 1.0 = perfect)

**How is the primary metric chosen?** *(per DEL3 Section 6)*

For **classification/regression tasks**, we prioritize:
1. `AUROC` – best for imbalanced medical data (DEL3 recommended)
2. `Accuracy` – overall correctness rate
3. `F1-Score` – precision-recall balance
4. `Sensitivity/Specificity` – for diagnostic screening

**Score Interpretation** *(Clinical Deployment Readiness)*

| Range | Tier | DEL3 Deployment Level | Clinical Guidance |
|:---:|:---:|:---:|:---|
| ≥ 0.90 | ⭐ Excellent | **Production Ready** | Suitable for clinical decision support with monitoring |
| 0.80-0.89 | ✅ Good | **Pilot/Validation** | Promising; requires prospective validation study |
| 0.70-0.79 | 🔶 Fair | **Research Only** | Research use; not for patient-facing applications |
| < 0.70 | 📈 Developing | **Development** | Requires significant improvement before deployment |

**Generalizability Analysis** *(DEL3 Section 4.3)*

Models are evaluated across demographic and technical strata:

- 👤 **Demographics**: Age groups, sex, ethnicity
- 🔬 **Technical**: Scanner manufacturer, acquisition parameters
- 🏥 **Clinical**: Disease stage, comorbidities, site

Sub-group performance gaps > 10% are flagged for fairness review.

**Ranking Rules**

1. Models ranked by **primary metric** (descending, higher = better)
2. Ties broken by secondary metrics in priority order
3. Each model's **best evaluation run** is used
4. Scores reported to 4 decimal places for precision
5. Statistical significance assessed via bootstrap CI (when available)

</details>


#### 📋 Complete Metrics Comparison

| Rank | Model | AUROC | Accuracy | F1-Score | Correlation | MSE |
|:---:|:---|:---:|:---:|:---:|:---:|:---:|
| 🥇 ⭐ | BrainLM | **0.9100** | 0.8700 | 0.8600 | 0.8100 | 0.4200 |
| 🥈 ✅ | BrainBERT | **0.8700** | 0.8200 | 0.8100 | 0.7600 | 0.5100 |
| 🥉 ✅ | NeuroCLIP | **0.8300** | 0.7900 | 0.7800 | 0.7200 | 0.5800 |

!!! tip "Legend"
    📊 **Primary metric**: AUROC (bold) | ⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)

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

<details>
<summary>🔬 <strong>BrainBERT</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 Siemens | 0.8900 | 0.8400 | 0.8300 | 450 |
| 🥈 GE | 0.8600 | 0.8100 | 0.8000 | 380 |
| Philips | 0.8500 | 0.8000 | 0.7900 | 370 |

</details>

<details>
<summary>📡 <strong>BrainBERT</strong> by Acquisition Type</summary>

| Acquisition Type | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 resting_state | 0.8800 | 0.8300 | 0.8200 | 600 |
| task_based | 0.8500 | 0.8000 | 0.7900 | 400 |

</details>

<details>
<summary>🔬 <strong>NeuroCLIP</strong> by Scanner</summary>

| Scanner | AUROC | Accuracy | F1-Score | N |
|---|---|---|---|---|
| 🥇 Siemens | 0.8500 | 0.8100 | 0.8000 | 450 |
| 🥈 GE | 0.8200 | 0.7800 | 0.7700 | 380 |
| Philips | 0.8100 | 0.7700 | 0.7600 | 370 |

</details>

---
*Ranked by **AUROC** (higher is better). Last updated from 3 evaluation(s).*

---

## 🚀 Get Your Model on the Leaderboard

Want to see your Foundation Model ranked here?

1. 📥 **Download** the benchmark suite and run locally
2. 🧪 **Evaluate** your model: `python -m fmbench run --help`
3. 📤 **Submit** your results via [GitHub Issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=benchmark_submission.md)

💡 **Propose new evaluation protocols** via [Issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=protocol_proposal.md)

!!! note "Curated Benchmark Hub"
    All submissions are reviewed before being added. See [Submission Guide](../contributing/submission_guide.md) for details.

*Aligned with [ITU/WHO FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards for healthcare AI evaluation.*
