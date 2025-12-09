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

### 📖 Understanding This Leaderboard

This section explains how we measure and compare AI models. Don't worry if you're new to AI metrics — we'll break it down step by step.

---

### 🎯 The Main Metric: `Accuracy`

**Accuracy**

**In simple terms:** The percentage of predictions the model got right

**How it works:** This is the most intuitive metric: out of all the predictions the model made, how many were correct? For example, if a model makes 100 predictions and 90 are correct, the accuracy is 90% (or 0.90). While easy to understand, accuracy can be misleading when classes are imbalanced (e.g., if 95% of cases are healthy, a model that always predicts 'healthy' would have 95% accuracy but miss all diseases).

**Score range:** 0.0 (all wrong) → 1.0 (all correct)

💡 **Example:** An accuracy of 0.92 means the model correctly classified 92 out of every 100 samples.

---

### 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:

- For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand
  how reliably the model separates different outcome groups. In addition to raw accuracy,
  we recommend also looking at metrics like AUROC and F1 Score, especially when classes are
  imbalanced (for example, when positive cases are rare).

---

### 📊 Performance Tiers: What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses:

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |
| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |
| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |
| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |

!!! tip "Important Context"
    These thresholds are general guidelines. The acceptable score depends on the specific clinical application, risk level, and whether the AI assists or replaces human judgment. Always consult domain experts when evaluating fitness for a particular use case.

---

### 📏 How We Determine Rankings

Models are ranked following these principles:

1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. For metrics where lower is better (like error rates), the lowest score wins.

2. **Ties are broken by secondary metrics** — If two models have identical primary scores, we look at other relevant metrics to determine which performs better overall.

3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), only its best result appears on the leaderboard. This ensures fair comparison.

4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, dataset used, and configuration to ensure transparency.

---

### 🏥 Why This Matters for Healthcare AI

Healthcare AI has higher stakes than many other AI applications. A model that works 95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. That's why we:

- Use **multiple metrics** to capture different aspects of performance
- Test **robustness** to real-world data quality issues
- Require **transparency** about evaluation conditions
- Follow **international standards** for healthcare AI assessment

---

### 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. This ensures our evaluations are:

- **Rigorous** — Following established scientific methodology
- **Comparable** — Using standardized metrics across different models
- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI

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

### 📖 Understanding This Leaderboard

This section explains how we measure and compare AI models. Don't worry if you're new to AI metrics — we'll break it down step by step.

---

### 🎯 The Main Metric: `report_quality_score`

**Report Quality Score**

**In simple terms:** An overall measure of how good the AI-generated medical reports are

**How it works:** This composite score combines multiple aspects of report quality: clinical accuracy (are the findings correct?), completeness (are important findings mentioned?), language quality (is it well-written?), and safety (no harmful content). It provides a single number to compare models, though looking at individual components gives more insight into specific strengths and weaknesses.

**Score range:** 0.0 (poor quality) → 1.0 (excellent quality)

💡 **Example:** A score of 0.85 indicates the model generates reports that are mostly accurate, complete, and well-structured.

---

### 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:

- For **report generation**, we care not only about language quality but also clinical safety.
  This metric is usually combined with others (e.g., clinical accuracy, hallucination rate,
  and completeness of findings) to judge whether the generated report is both readable **and**
  medically reliable.

---

### 📊 Performance Tiers: What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses:

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |
| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |
| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |
| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |

!!! tip "Important Context"
    These thresholds are general guidelines. The acceptable score depends on the specific clinical application, risk level, and whether the AI assists or replaces human judgment. Always consult domain experts when evaluating fitness for a particular use case.

---

### 📏 How We Determine Rankings

Models are ranked following these principles:

1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. For metrics where lower is better (like error rates), the lowest score wins.

2. **Ties are broken by secondary metrics** — If two models have identical primary scores, we look at other relevant metrics to determine which performs better overall.

3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), only its best result appears on the leaderboard. This ensures fair comparison.

4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, dataset used, and configuration to ensure transparency.

---

### 🏥 Why This Matters for Healthcare AI

Healthcare AI has higher stakes than many other AI applications. A model that works 95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. That's why we:

- Use **multiple metrics** to capture different aspects of performance
- Test **robustness** to real-world data quality issues
- Require **transparency** about evaluation conditions
- Follow **international standards** for healthcare AI assessment

---

### 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. This ensures our evaluations are:

- **Rigorous** — Following established scientific methodology
- **Comparable** — Using standardized metrics across different models
- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI

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

### 📖 Understanding This Leaderboard

This section explains how we measure and compare AI models. Don't worry if you're new to AI metrics — we'll break it down step by step.

---

### 🎯 The Main Metric: `AUROC`

**Area Under ROC Curve (AUROC)**

**In simple terms:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)

**How it works:** Think of it like this: if you randomly pick one positive case and one negative case, AUROC tells you the probability that the model correctly identifies which is which. A score of 0.5 means the model is just guessing randomly (like flipping a coin), while 1.0 means it perfectly separates all cases.

**Score range:** 0.5 (random guessing) → 1.0 (perfect separation)

💡 **Example:** An AUROC of 0.85 means the model correctly ranks a positive case higher than a negative case 85% of the time.

---

### 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:

- For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand
  how reliably the model separates different outcome groups. In addition to raw accuracy,
  we recommend also looking at metrics like AUROC and F1 Score, especially when classes are
  imbalanced (for example, when positive cases are rare).

---

### 📊 Performance Tiers: What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses:

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |
| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |
| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |
| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |

!!! tip "Important Context"
    These thresholds are general guidelines. The acceptable score depends on the specific clinical application, risk level, and whether the AI assists or replaces human judgment. Always consult domain experts when evaluating fitness for a particular use case.

---

### 📏 How We Determine Rankings

Models are ranked following these principles:

1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. For metrics where lower is better (like error rates), the lowest score wins.

2. **Ties are broken by secondary metrics** — If two models have identical primary scores, we look at other relevant metrics to determine which performs better overall.

3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), only its best result appears on the leaderboard. This ensures fair comparison.

4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, dataset used, and configuration to ensure transparency.

---

### 🏥 Why This Matters for Healthcare AI

Healthcare AI has higher stakes than many other AI applications. A model that works 95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. That's why we:

- Use **multiple metrics** to capture different aspects of performance
- Test **robustness** to real-world data quality issues
- Require **transparency** about evaluation conditions
- Follow **international standards** for healthcare AI assessment

---

### 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. This ensures our evaluations are:

- **Rigorous** — Following established scientific methodology
- **Comparable** — Using standardized metrics across different models
- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI

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

### 📖 Understanding This Leaderboard

This section explains how we measure and compare AI models. Don't worry if you're new to AI metrics — we'll break it down step by step.

---

### 🎯 The Main Metric: `AUROC`

**Area Under ROC Curve (AUROC)**

**In simple terms:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)

**How it works:** Think of it like this: if you randomly pick one positive case and one negative case, AUROC tells you the probability that the model correctly identifies which is which. A score of 0.5 means the model is just guessing randomly (like flipping a coin), while 1.0 means it perfectly separates all cases.

**Score range:** 0.5 (random guessing) → 1.0 (perfect separation)

💡 **Example:** An AUROC of 0.85 means the model correctly ranks a positive case higher than a negative case 85% of the time.

---

### 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:

- For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand
  how reliably the model separates different outcome groups. In addition to raw accuracy,
  we recommend also looking at metrics like AUROC and F1 Score, especially when classes are
  imbalanced (for example, when positive cases are rare).

---

### 📊 Performance Tiers: What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses:

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |
| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |
| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |
| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |

!!! tip "Important Context"
    These thresholds are general guidelines. The acceptable score depends on the specific clinical application, risk level, and whether the AI assists or replaces human judgment. Always consult domain experts when evaluating fitness for a particular use case.

---

### 📏 How We Determine Rankings

Models are ranked following these principles:

1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. For metrics where lower is better (like error rates), the lowest score wins.

2. **Ties are broken by secondary metrics** — If two models have identical primary scores, we look at other relevant metrics to determine which performs better overall.

3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), only its best result appears on the leaderboard. This ensures fair comparison.

4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, dataset used, and configuration to ensure transparency.

---

### 🏥 Why This Matters for Healthcare AI

Healthcare AI has higher stakes than many other AI applications. A model that works 95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. That's why we:

- Use **multiple metrics** to capture different aspects of performance
- Test **robustness** to real-world data quality issues
- Require **transparency** about evaluation conditions
- Follow **international standards** for healthcare AI assessment

---

### 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. This ensures our evaluations are:

- **Rigorous** — Following established scientific methodology
- **Comparable** — Using standardized metrics across different models
- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI

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

### 📖 Understanding This Leaderboard

This section explains how we measure and compare AI models. Don't worry if you're new to AI metrics — we'll break it down step by step.

---

### 🎯 The Main Metric: `AUROC`

**Area Under ROC Curve (AUROC)**

**In simple terms:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)

**How it works:** Think of it like this: if you randomly pick one positive case and one negative case, AUROC tells you the probability that the model correctly identifies which is which. A score of 0.5 means the model is just guessing randomly (like flipping a coin), while 1.0 means it perfectly separates all cases.

**Score range:** 0.5 (random guessing) → 1.0 (perfect separation)

💡 **Example:** An AUROC of 0.85 means the model correctly ranks a positive case higher than a negative case 85% of the time.

---

### 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:

- For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand
  how reliably the model separates different outcome groups. In addition to raw accuracy,
  we recommend also looking at metrics like AUROC and F1 Score, especially when classes are
  imbalanced (for example, when positive cases are rare).

---

### 📊 Performance Tiers: What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses:

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |
| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |
| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |
| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |

!!! tip "Important Context"
    These thresholds are general guidelines. The acceptable score depends on the specific clinical application, risk level, and whether the AI assists or replaces human judgment. Always consult domain experts when evaluating fitness for a particular use case.

---

### 📏 How We Determine Rankings

Models are ranked following these principles:

1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. For metrics where lower is better (like error rates), the lowest score wins.

2. **Ties are broken by secondary metrics** — If two models have identical primary scores, we look at other relevant metrics to determine which performs better overall.

3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), only its best result appears on the leaderboard. This ensures fair comparison.

4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, dataset used, and configuration to ensure transparency.

---

### 🏥 Why This Matters for Healthcare AI

Healthcare AI has higher stakes than many other AI applications. A model that works 95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. That's why we:

- Use **multiple metrics** to capture different aspects of performance
- Test **robustness** to real-world data quality issues
- Require **transparency** about evaluation conditions
- Follow **international standards** for healthcare AI assessment

---

### 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. This ensures our evaluations are:

- **Rigorous** — Following established scientific methodology
- **Comparable** — Using standardized metrics across different models
- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI

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

### 📖 Understanding This Leaderboard

This section explains how we measure and compare AI models. Don't worry if you're new to AI metrics — we'll break it down step by step.

---

### 🎯 The Main Metric: `Correlation`

**Correlation**

**In simple terms:** How closely the model's predictions match the actual values

**How it works:** Correlation measures the strength and direction of the relationship between predicted and actual values. A correlation of 1.0 means perfect positive agreement (when actual goes up, prediction goes up proportionally), while 0 means no relationship at all. This is commonly used for reconstruction tasks where we want to see how well the model can recreate the original signal.

**Score range:** -1.0 (perfect inverse) → 0 (no relationship) → 1.0 (perfect match)

💡 **Example:** A correlation of 0.78 means the model's outputs track reasonably well with the true values.

---

### 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:

- For **regression / continuous prediction** tasks, this metric captures how closely the model's
  predicted values track the true values over a range (e.g., symptom severity, signal amplitude).
  We are usually interested in both overall fit (correlation) and error magnitude.

---

### 📊 Performance Tiers: What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses:

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |
| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |
| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |
| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |

!!! tip "Important Context"
    These thresholds are general guidelines. The acceptable score depends on the specific clinical application, risk level, and whether the AI assists or replaces human judgment. Always consult domain experts when evaluating fitness for a particular use case.

---

### 📏 How We Determine Rankings

Models are ranked following these principles:

1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. For metrics where lower is better (like error rates), the lowest score wins.

2. **Ties are broken by secondary metrics** — If two models have identical primary scores, we look at other relevant metrics to determine which performs better overall.

3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), only its best result appears on the leaderboard. This ensures fair comparison.

4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, dataset used, and configuration to ensure transparency.

---

### 🏥 Why This Matters for Healthcare AI

Healthcare AI has higher stakes than many other AI applications. A model that works 95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. That's why we:

- Use **multiple metrics** to capture different aspects of performance
- Test **robustness** to real-world data quality issues
- Require **transparency** about evaluation conditions
- Follow **international standards** for healthcare AI assessment

---

### 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. This ensures our evaluations are:

- **Rigorous** — Following established scientific methodology
- **Comparable** — Using standardized metrics across different models
- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI

</details>

---

## 📋 Other Benchmarks

### Foundation Model Robustness Evaluation

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **geneformer** 👑 | 0.9995 | ⭐ Excellent | neuro/robustness, 2025-11-27 |
| 🥈 | **Brain-JEPA** | 0.8650 | ✅ Good | DS-TOY-NEURO-ROBUSTN, 2024-01-20 |
| 🥉 | **BrainHarmony** | 0.8450 | ✅ Good | DS-TOY-NEURO-ROBUSTN, 2024-01-18 |
| 🏅 | Geneformer | 0.8350 | ✅ Good | DS-TOY-GENOMICS, 2024-01-10 |
| 🏅 | BrainLM | 0.8250 | ✅ Good | DS-TOY-NEURO-ROBUSTN, 2024-01-16 |
| 🎖️ | HyenaDNA | 0.7950 | 🔶 Fair | DS-TOY-GENOMICS, 2024-01-12 |
| 🎖️ | Baseline (Random/Majority) | 0.7810 | 🔶 Fair | neuro/robustness, 2025-11-27 |

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
