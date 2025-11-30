# 🏆 Foundation Model Leaderboards

!!! success "Benchmark Hub Overview"
    📊 **7** Benchmarks | 🤖 **21** Models | 📈 **38** Evaluations


> **What is this?** This page ranks AI models for healthcare applications. 
> Higher-ranked models perform better on standardized tests.
> 
> **How to read it:** Each table shows models from best (🥇) to developing (📈).
> Click "How are scores calculated?" for details on what the numbers mean.

## 🧭 Jump To

- [🌐 Overall Rankings](#overall-rankings-all-modalities) — Best across all categories
- [🧬 Genomics](#genomics)
- [🧠 Brain Imaging (MRI/fMRI)](#brain-imaging-mrifmri)

---

## 🌐 Overall Rankings (All Modalities)

*Best score per model across all benchmarks*

| Rank | Model | Best Score | Benchmark | Modality |
|:---:|:---|:---:|:---|:---|
| 🥇 | **geneformer** 👑 | 0.9995 | Foundation Model Robustne | 📊 Other |
| 🥈 | **Brain-JEPA** | 0.9350 | Alzheimer's Disease Class | 🧠 Brain Imaging ( |
| 🥉 | **Evo 2** | 0.9250 | Cell Type Annotation | 🧬 Genomics |
| 🏅 | UNI | 0.9200 | Alzheimer's Disease Class | 🧠 Brain Imaging ( |
| 🏅 | Geneformer | 0.9100 | Cell Type Annotation | 🧬 Genomics |
| 🎖️ | BrainLM | 0.9100 | fMRI Foundation Model Ben | 🧠 Brain Imaging ( |
| 🎖️ | SWIFT | 0.8950 | Cell Type Annotation | 🧬 Genomics |
| 🎖️ | Caduceus | 0.8850 | Cell Type Annotation | 🧬 Genomics |
| 🎖️ | Me-LLaMA | 0.8750 | Clinical Report Generatio | 🧬 Genomics |
| 🎖️ | BrainBERT | 0.8700 | fMRI Foundation Model Ben | 🧠 Brain Imaging ( |
| #11 | HyenaDNA | 0.8700 | Cell Type Annotation | 🧬 Genomics |
| #12 | M3FM | 0.8600 | Clinical Report Generatio | 🧬 Genomics |
| #13 | DNABERT-2 | 0.8500 | Cell Type Annotation | 🧬 Genomics |
| #14 | BrainMT | 0.8500 | fMRI Foundation Model Ben | 🧠 Brain Imaging ( |
| #15 | BrainHarmony | 0.8450 | Foundation Model Robustne | 📊 Other |
| #16 | OpenFlamingo | 0.8400 | Clinical Report Generatio | 🧬 Genomics |
| #17 | NeuroClips | 0.8300 | fMRI Foundation Model Ben | 🧠 Brain Imaging ( |
| #18 | TITAN | 0.8100 | Clinical Report Generatio | 🧬 Genomics |
| #19 | Baseline (Random/Majority) | 0.7810 | Foundation Model Robustne | 📊 Other |
| #20 | Med-Flamingo | 0.7800 | Clinical Report Generatio | 🧬 Genomics |
| #21 | RadBERT | 0.6900 | Clinical Report Generatio | 🧬 Genomics |

!!! abstract "Performance Distribution"
    ⭐ 6 Excellent | ✅ 12 Good | 🔶 2 Fair | 📈 1 Developing

---

## 🧬 Genomics

### 🎯 Classification

#### Cell Type Annotation

*Predicting cell types from single-cell RNA-seq data.*


<div align="center">

```
                    🏆                    
                                          
              🥇   Evo 2                 
                 (0.925)                 
             ╔═══════════════╗             
             ║               ║             
   🥈 Geneformer   ║               ║   🥉   SWIFT      
      (0.910)      ║               ║      (0.895)      
  ╔═══════════╝               ╚═══════════╗  
  ║                                       ║  
══╩═══════════════════════════════════════╩══
```

</div>

**6 models ranked by `Accuracy`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **Evo 2** 👑 | 0.9250 | ⭐ Excellent | PBMC 68k, 2024-02-01 |
| 🥈 | **Geneformer** | 0.9100 | ⭐ Excellent | PBMC 68k, 2023-11-01 |
| 🥉 | **SWIFT** | 0.8950 | ✅ Good | PBMC 68k, 2024-01-15 |
| 🏅 | Caduceus | 0.8850 | ✅ Good | PBMC 68k, 2024-01-12 |
| 🏅 | HyenaDNA | 0.8700 | ✅ Good | PBMC 68k, 2024-01-08 |
| 🎖️ | DNABERT-2 | 0.8500 | ✅ Good | PBMC 68k, 2024-01-05 |

!!! tip "Quick Comparison"
    **🥇 Evo 2** leads with Accuracy = **0.9250**

    - Gap to 🥈 Geneformer: +0.0150
    - Score spread (best to worst): 0.0750


<details>
<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>

---

### 🎯 What We Measure: `Accuracy`

> **Accuracy**
>
> Percentage of correct predictions
>
> 📏 Range: 0% → 100% (or 0.0 → 1.0)

---

### 📊 What Do Scores Mean?

| Score | Rating | What It Means |
|:---:|:---:|:---|
| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |
| **0.80-0.89** | ✅ Good | Promising, needs more testing |
| **0.70-0.79** | 🔶 Fair | Research use only |
| **< 0.70** | 📈 Developing | Needs more work |

---

### 📏 How We Rank

1. **Higher score = Better ranking** (except for error metrics)
2. If scores tie, we look at secondary metrics
3. Only the best run from each model counts

---

!!! info "Standards Alignment"
    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.

</details>

---

### ✍️ Generation

#### Clinical Report Generation Quality


<div align="center">

```
                    🏆                    
                                          
              🥇   Me-LLaMA                
                 (0.875)                 
             ╔═══════════════╗             
             ║               ║             
   🥈     M3FM       ║               ║   🥉 OpenFlamingo   
      (0.860)      ║               ║      (0.840)      
  ╔═══════════╝               ╚═══════════╗  
  ║                                       ║  
══╩═══════════════════════════════════════╩══
```

</div>

**6 models ranked by `report_quality_score`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **Me-LLaMA** 👑 | 0.8750 | ✅ Good | mimic_cxr_reports, 2024-02-05 |
| 🥈 | **M3FM** | 0.8600 | ✅ Good | mimic_cxr_reports, 2024-01-28 |
| 🥉 | **OpenFlamingo** | 0.8400 | ✅ Good | mimic_cxr_reports, 2024-01-20 |
| 🏅 | TITAN | 0.8100 | ✅ Good | mimic_cxr_reports, 2024-01-25 |
| 🏅 | Med-Flamingo | 0.7800 | 🔶 Fair | mimic_cxr_reports, 2024-01-18 |
| 🎖️ | RadBERT | 0.6900 | 📈 Developing | mimic_cxr_reports, 2024-01-12 |

!!! tip "Quick Comparison"
    **🥇 Me-LLaMA** leads with report_quality_score = **0.8750**

    - Gap to 🥈 M3FM: +0.0150
    - Score spread (best to worst): 0.1850


<details>
<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>

---

### 🎯 What We Measure: `report_quality_score`

> **Report Quality Score**
>
> Overall quality of generated medical reports
>
> 📏 Range: 0.0 (poor) → 1.0 (excellent)

---

### 📊 What Do Scores Mean?

| Score | Rating | What It Means |
|:---:|:---:|:---|
| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |
| **0.80-0.89** | ✅ Good | Promising, needs more testing |
| **0.70-0.79** | 🔶 Fair | Research use only |
| **< 0.70** | 📈 Developing | Needs more work |

---

### 📏 How We Rank

1. **Higher score = Better ranking** (except for error metrics)
2. If scores tie, we look at secondary metrics
3. Only the best run from each model counts

---

!!! info "Standards Alignment"
    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.

</details>

---

## 🧠 Brain Imaging (MRI/fMRI)

### 🎯 Classification

#### Toy Classification Benchmark

*A toy benchmark for testing the pipeline.*

**2 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **Baseline (Random/Majority)** 👑 | 0.5597 | 📈 Developing | Toy fMRI Classificat, 2025-11-27 |
| 🥈 | **BrainLM** | 0.5193 | 📈 Developing | Toy fMRI Classificat, 2025-11-27 |

!!! tip "Quick Comparison"
    **🥇 Baseline (Random/Majority)** leads with AUROC = **0.5597**

    - Gap to 🥈 BrainLM: +0.0404


<details>
<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>

---

### 🎯 What We Measure: `AUROC`

> **Area Under ROC Curve**
>
> How well the model distinguishes between classes
>
> 📏 Range: 0.5 (random guess) → 1.0 (perfect)

---

### 📊 What Do Scores Mean?

| Score | Rating | What It Means |
|:---:|:---:|:---|
| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |
| **0.80-0.89** | ✅ Good | Promising, needs more testing |
| **0.70-0.79** | 🔶 Fair | Research use only |
| **< 0.70** | 📈 Developing | Needs more work |

---

### 📏 How We Rank

1. **Higher score = Better ranking** (except for error metrics)
2. If scores tie, we look at secondary metrics
3. Only the best run from each model counts

---

!!! info "Standards Alignment"
    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.

</details>

---

#### Alzheimer's Disease Classification using Brain MRI

*Binary classification of AD vs CN using structural MRI data.*


<div align="center">

```
                    🏆                    
                                          
              🥇 Brain-JEPA              
                 (0.935)                 
             ╔═══════════════╗             
             ║               ║             
   🥈    UNI       ║               ║   🥉  BrainLM     
      (0.920)      ║               ║      (0.910)      
  ╔═══════════╝               ╚═══════════╗  
  ║                                       ║  
══╩═══════════════════════════════════════╩══
```

</div>

**3 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **Brain-JEPA** 👑 | 0.9350 | ⭐ Excellent | ADNI, 2024-01-20 |
| 🥈 | **UNI** | 0.9200 | ⭐ Excellent | Alzheimer's Disease , 2023-10-27 |
| 🥉 | **BrainLM** | 0.9100 | ⭐ Excellent | ADNI, 2024-01-15 |

!!! tip "Quick Comparison"
    **🥇 Brain-JEPA** leads with AUROC = **0.9350**

    - Gap to 🥈 UNI: +0.0150
    - Score spread (best to worst): 0.0250


<details>
<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>

---

### 🎯 What We Measure: `AUROC`

> **Area Under ROC Curve**
>
> How well the model distinguishes between classes
>
> 📏 Range: 0.5 (random guess) → 1.0 (perfect)

---

### 📊 What Do Scores Mean?

| Score | Rating | What It Means |
|:---:|:---:|:---|
| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |
| **0.80-0.89** | ✅ Good | Promising, needs more testing |
| **0.70-0.79** | 🔶 Fair | Research use only |
| **< 0.70** | 📈 Developing | Needs more work |

---

### 📏 How We Rank

1. **Higher score = Better ranking** (except for error metrics)
2. If scores tie, we look at secondary metrics
3. Only the best run from each model counts

---

!!! info "Standards Alignment"
    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.

</details>

---

### 📋 Classification/Reconstruction

#### fMRI Foundation Model Benchmark (Granular)


<div align="center">

```
                    🏆                    
                                          
              🥇 Brain-JEPA              
                 (0.925)                 
             ╔═══════════════╗             
             ║               ║             
   🥈  BrainLM     ║               ║   🥉 BrainBERT    
      (0.910)      ║               ║      (0.870)      
  ╔═══════════╝               ╚═══════════╗  
  ║                                       ║  
══╩═══════════════════════════════════════╩══
```

</div>

**5 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **Brain-JEPA** 👑 | 0.9250 | ⭐ Excellent | hcp_1200, 2024-01-22 |
| 🥈 | **BrainLM** | 0.9100 | ⭐ Excellent | hcp_1200, 2024-01-15 |
| 🥉 | **BrainBERT** | 0.8700 | ✅ Good | hcp_1200, 2024-01-10 |
| 🏅 | BrainMT | 0.8500 | ✅ Good | hcp_1200, 2024-01-18 |
| 🏅 | NeuroClips | 0.8300 | ✅ Good | hcp_1200, 2024-01-05 |

!!! tip "Quick Comparison"
    **🥇 Brain-JEPA** leads with AUROC = **0.9250**

    - Gap to 🥈 BrainLM: +0.0150
    - Score spread (best to worst): 0.0950


<details>
<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>

---

### 🎯 What We Measure: `AUROC`

> **Area Under ROC Curve**
>
> How well the model distinguishes between classes
>
> 📏 Range: 0.5 (random guess) → 1.0 (perfect)

---

### 📊 What Do Scores Mean?

| Score | Rating | What It Means |
|:---:|:---:|:---|
| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |
| **0.80-0.89** | ✅ Good | Promising, needs more testing |
| **0.70-0.79** | 🔶 Fair | Research use only |
| **< 0.70** | 📈 Developing | Needs more work |

---

### 📏 How We Rank

1. **Higher score = Better ranking** (except for error metrics)
2. If scores tie, we look at secondary metrics
3. Only the best run from each model counts

---

!!! info "Standards Alignment"
    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.

</details>

---

### 🔄 Reconstruction

#### Brain Time-Series Modeling

*Evaluating ability to reconstruct masked fMRI voxel time-series.*

**1 models ranked by `Correlation`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **BrainLM** 👑 | 0.7800 | 🔶 Fair | UK Biobank fMRI tens, 2025-11-15 |


<details>
<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>

---

### 🎯 What We Measure: `Correlation`

> **Correlation**: Performance measure

---

### 📊 What Do Scores Mean?

| Score | Rating | What It Means |
|:---:|:---:|:---|
| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |
| **0.80-0.89** | ✅ Good | Promising, needs more testing |
| **0.70-0.79** | 🔶 Fair | Research use only |
| **< 0.70** | 📈 Developing | Needs more work |

---

### 📏 How We Rank

1. **Higher score = Better ranking** (except for error metrics)
2. If scores tie, we look at secondary metrics
3. Only the best run from each model counts

---

!!! info "Standards Alignment"
    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.

</details>

---

## 📋 Other Benchmarks

### Foundation Model Robustness Evaluation

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **geneformer** 👑 | 0.9995 | ⭐ Excellent | -, 2025-11-27 |
| 🥈 | **Brain-JEPA** | 0.8650 | ✅ Good | DS-TOY-NEURO-ROBUSTN, 2024-01-20 |
| 🥉 | **BrainHarmony** | 0.8450 | ✅ Good | DS-TOY-NEURO-ROBUSTN, 2024-01-18 |
| 🏅 | Geneformer | 0.8350 | ✅ Good | DS-TOY-GENOMICS, 2024-01-10 |
| 🏅 | BrainLM | 0.8250 | ✅ Good | DS-TOY-NEURO-ROBUSTN, 2024-01-16 |
| 🎖️ | HyenaDNA | 0.7950 | 🔶 Fair | DS-TOY-GENOMICS, 2024-01-12 |
| 🎖️ | Baseline (Random/Majority) | 0.7810 | 🔶 Fair | -, 2025-11-27 |
| 🎖️ | Baseline (Random/Majority) | 0.7810 | 🔶 Fair | -, 2025-11-27 |
| 🎖️ | Baseline (Random/Majority) | 0.7810 | 🔶 Fair | -, 2025-11-27 |
| 🎖️ | Baseline (Random/Majority) | 0.7749 | 🔶 Fair | -, 2025-11-27 |
| #11 | Baseline (Random/Majority) | 0.4554 | 📈 Developing | -, 2025-11-27 |

---


## 🚀 Add Your Model

Want your model on this leaderboard?

1. **Download** the benchmark toolkit
2. **Run locally** on your model (your code stays private!)
3. **Submit results** via [GitHub Issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=benchmark_submission.md)

[📥 Get Started](../index.md){ .md-button .md-button--primary }
[📖 Submission Guide](../contributing/submission_guide.md){ .md-button }

---

*Aligned with [ITU/WHO FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards for healthcare AI evaluation.*
