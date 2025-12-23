# 🏆 Foundation Model Leaderboards

!!! success "Benchmark Hub Overview"
    📊 **7** Benchmarks | 🤖 **12** Models | 📈 **35** Evaluations


> **What is this?** This page ranks AI models for healthcare applications. 
> Higher-ranked models perform better on standardized tests.
> 
> **How to read it:** Each table shows models from best (🥇) to developing (📈).
> Click "How are scores calculated?" for details on what the numbers mean.

## Example: what a real submission looks like

This is a **real, end-to-end** run using the built-in baseline model. Your submission should look like this: a local run that produces `report.md` + `eval.yaml`.

| Model ID | Suite / Benchmark | Task | <abbr title="Area Under the Receiver Operating Characteristic curve">AUROC</abbr> | <abbr title="Reverse area-under-curve for channel dropout robustness">dropout rAUC</abbr> | <abbr title="Reverse area-under-curve for Gaussian noise robustness">noise rAUC</abbr> |
|:---|:---|:---|---:|---:|---:|
| `dummy_classifier` | `SUITE-TOY-CLASS` / `BM-TOY-CLASS` | Toy fMRI-like classification | 0.5597 | 0.7760 | 0.7867 |

**Artifacts:** [Example classification eval.yaml](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/blob/main/evals/SUITE-TOY-CLASS-dummy_classifier-20251127-071011.yaml) · [Example classification report.md](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/blob/main/reports/SUITE-TOY-CLASS-dummy_classifier-20251127-071011.md) · [Example robustness eval.yaml](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/blob/main/evals/ROBUSTNESS-dummy_classifier-20251127-071004.yaml) · [Example robustness report.md](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/blob/main/reports/ROBUSTNESS-dummy_classifier-20251127-071004.md)

---

## 📐 Metric Cheat Sheet

Use this as a general reference for the metrics that appear on the leaderboards.

### Area Under ROC Curve (AUROC)

- **What it measures:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)
- **Typical range:** 0.5 (random guessing) → 1.0 (perfect separation)
- **Example:** An AUROC of 0.85 means the model correctly ranks a positive case higher than a negative case 85% of the time.

### Accuracy

- **What it measures:** The percentage of predictions the model got right
- **Typical range:** 0.0 (all wrong) → 1.0 (all correct)
- **Example:** An accuracy of 0.92 means the model correctly classified 92 out of every 100 samples.

### F1 Score

- **What it measures:** A balanced measure that considers both false alarms and missed cases
- **Typical range:** 0.0 (poor) → 1.0 (perfect balance of precision and recall)
- **Example:** An F1 of 0.85 indicates the model has a good balance between catching real cases and avoiding false alarms.

### Correlation

- **What it measures:** How closely the model's predictions match the actual values
- **Typical range:** -1.0 (perfect inverse) → 0 (no relationship) → 1.0 (perfect match)
- **Example:** A correlation of 0.78 means the model's outputs track reasonably well with the true values.

### Robustness Score

- **What it measures:** How stable and reliable the model is when data quality isn't perfect
- **Typical range:** 0.0 (performance collapses with any noise) → 1.0 (completely stable)
- **Example:** A robustness score of 0.82 means the model maintains most of its accuracy even when data has noise or missing values.

### Report Quality Score

- **What it measures:** An overall measure of how good the AI-generated medical reports are
- **Typical range:** 0.0 (poor quality) → 1.0 (excellent quality)
- **Example:** A score of 0.85 indicates the model generates reports that are mostly accurate, complete, and well-structured.

### Clinical Accuracy

- **What it measures:** Are the medical findings in the generated report actually correct?
- **Typical range:** 0.0 (all findings wrong) → 1.0 (all findings correct)
- **Example:** A clinical accuracy of 0.92 means 92% of the medical findings in the report are verified as correct.

### Hallucination Rate

- **What it measures:** How often the AI makes up information that isn't supported by the input data
- **Typical range:** 0.0 (no hallucinations — ideal) → 1.0 (everything is made up)
- **Example:** A hallucination rate of 0.05 means only 5% of generated content is unsupported by the input — quite good!

### BERTScore

- **What it measures:** How similar the generated text is to the reference text in meaning (not just exact words)
- **Typical range:** 0.0 (completely different meaning) → 1.0 (semantically identical)
- **Example:** A BERTScore of 0.87 indicates the generated report conveys very similar clinical meaning to the expert reference.

---

## 🧭 Jump To

- [🧬 Genomics](#genomics)
- [🧠 Brain Imaging (MRI/fMRI)](#brain-imaging-mrifmri)

---

## 🧬 Genomics

### 🎯 Classification

#### DNA Promoter Classification

*Benchmark for classifying DNA sequences as promoters or non-promoters.
Promoters are regulatory regions at transcription start sites (TSS).
This benchmark focuses on non-TATA promoters, which lack the canonical
TATA box and represent ~75% of human promoters.
*


<div align="center">

```
                    🏆                    
                                          
              🥇 HyenaDNA              
                 (0.872)                 
             ╔═══════════════╗             
             ║               ║             
   🥈 Caduceus   ║               ║   🥉  Evo 2     
      (0.859)      ║               ║      (0.859)      
  ╔═══════════╝               ╚═══════════╗  
  ║                                       ║  
══╩═══════════════════════════════════════╩══
```

</div>

**6 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **HyenaDNA** 👑 | 0.8720 | ✅ Good | DS-DNA-PROMOTER, 2025-12-18T21:03:12.030852 |
| 🥈 | **Caduceus** | 0.8594 | ✅ Good | Human Non-TATA Promo, 2025-12-19T12:00:12.829913 |
| 🥉 | **Evo 2** | 0.8594 | ✅ Good | Human Non-TATA Promo, 2025-12-19T12:00:13.671201 |
| 🏅 | kmer_k6 | 0.8357 | ✅ Good | Human Non-TATA Promo, 2025-12-18T18:44:10.847321 |
| 🏅 | DNABERT-2 | 0.8357 | ✅ Good | Human Non-TATA Promo, 2025-12-18T18:44:27.391206 |
| 🎖️ | HyenaDNA | 0.8357 | ✅ Good | Human Non-TATA Promo, 2025-12-18T18:44:19.651418 |

!!! tip "Quick Comparison"
    **🥇 HyenaDNA** leads with AUROC = **0.8720**

    - Gap to 🥈 Caduceus: +0.0126
    - Score spread (best to worst): 0.0363


<details class="score-details" markdown="1">
<summary>📐 <strong>How are scores calculated for this benchmark?</strong> (click to expand)</summary>

## 📂 What this leaderboard measures

- **Benchmark:** `BM-DNA-PROMOTER` — DNA Promoter Classification
- **Domain:** Genetics, Genomics - Promoter Prediction
- **Task type:** Classification
- **Datasets used in the table above:**
  - `DS-DNA-PROMOTER` — DS-DNA-PROMOTER
  - `DS-DNA-PROMOTERS-NONTATA` — Human Non-TATA Promoters (EPD)
- **Typical sample size in these runs:** ~6250 samples (train + test combined)
- **Primary ranking metric:** `AUROC` (the score column in the table)

<br>

---

## 🎯 Primary metric for this leaderboard

- **Metric:** `AUROC`
- **What it measures:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)
- **Typical range:** 0.5 (random guessing) → 1.0 (perfect separation)

> 🔎 For a full explanation of this and other metrics, see the **Metric Cheat Sheet** near the top of this page.

<br>

---

## 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance.

**Here's how this metric should be interpreted for this benchmark:**

<br>

For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand how reliably the model separates different outcome groups.

> 💡 **Tip:** In addition to raw accuracy, look at metrics like **AUROC** and **F1 Score**, especially when classes are imbalanced (when positive cases are rare).

<br>

---

## 📊 Performance Tiers

### What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses.

<br>

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier, consistently reliable | Clinical pilots (with oversight) |
| **0.80 – 0.89** | ✅ Good | Strong performance, real promise | Validation studies |
| **0.70 – 0.79** | 🔶 Fair | Moderate, has limitations | Research only |
| **< 0.70** | 📈 Developing | Needs improvement | Early research |

<br>

!!! warning "Important Context"
    These thresholds are **general guidelines**.

    The acceptable score depends on:

    - The specific clinical application
    - Risk level of the use case
    - Whether AI assists or replaces human judgment

    **Always consult domain experts** when evaluating fitness for a particular use case.

<br>

---

## 📏 How We Determine Rankings

Models are ranked following these principles:

<br>

### 1️⃣ Primary metric determines rank

The model with the highest score in the main metric ranks first.

> For metrics where **lower is better** (like error rates), the lowest score wins.

<br>

### 2️⃣ Ties are broken by secondary metrics

If two models have identical primary scores, we look at other relevant metrics.

<br>

### 3️⃣ Best run per model

If a model was evaluated multiple times (e.g., with different settings), only its **best result** appears on the leaderboard.

<br>

### 4️⃣ Reproducibility required

All results must be reproducible. We record:

- Evaluation date
- Dataset used
- Configuration details

<br>

---

## 🏥 Why This Matters for Healthcare AI

Healthcare AI has **higher stakes** than many other AI applications.

> A model that works 95% of the time might sound good, but that 5% could mean **missed diagnoses** or **incorrect treatments**.

<br>

**That's why we:**

✅ Use **multiple metrics** to capture different aspects of performance

✅ Test **robustness** to real-world data quality issues

✅ Require **transparency** about evaluation conditions

✅ Follow **international standards** for healthcare AI assessment

<br>

---

## 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework.

<br>

This ensures our evaluations are:

| Quality | What it means |
|:--------|:--------------|
| **Rigorous** | Following established scientific methodology |
| **Comparable** | Using standardized metrics across models |
| **Trustworthy** | Aligned with WHO/ITU recommendations |

<br>

</details>

---

#### Cell Type Annotation

*Predicting cell types from single-cell RNA-seq data.*

**2 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **Baseline (Random/Majority)** 👑 | 0.0000 | 📈 Developing | PBMC 3k (processed, , 2025-12-18 |
| 🥈 | **geneformer** | 0.0000 | 📈 Developing | PBMC 3k (processed, , 2025-12-18 |

!!! tip "Quick Comparison"
    **🥇 Baseline (Random/Majority)** leads with AUROC = **0.0000**

    - Gap to 🥈 geneformer: +0.0000


<details class="score-details" markdown="1">
<summary>📐 <strong>How are scores calculated for this benchmark?</strong> (click to expand)</summary>

## 📂 What this leaderboard measures

- **Benchmark:** `BM-002` — Cell Type Annotation
- **Domain:** Genomics, Single-cell Transcriptomics
- **Task type:** Classification
- **Datasets used in the table above:**
  - `DS-PBMC` — PBMC 3k (processed, with cell type labels)
- **Primary ranking metric:** `AUROC` (the score column in the table)

<br>

---

## 🎯 Primary metric for this leaderboard

- **Metric:** `AUROC`
- **What it measures:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)
- **Typical range:** 0.5 (random guessing) → 1.0 (perfect separation)

> 🔎 For a full explanation of this and other metrics, see the **Metric Cheat Sheet** near the top of this page.

<br>

---

## 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance.

**Here's how this metric should be interpreted for this benchmark:**

<br>

For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand how reliably the model separates different outcome groups.

> 💡 **Tip:** In addition to raw accuracy, look at metrics like **AUROC** and **F1 Score**, especially when classes are imbalanced (when positive cases are rare).

<br>

---

## 📊 Performance Tiers

### What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses.

<br>

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier, consistently reliable | Clinical pilots (with oversight) |
| **0.80 – 0.89** | ✅ Good | Strong performance, real promise | Validation studies |
| **0.70 – 0.79** | 🔶 Fair | Moderate, has limitations | Research only |
| **< 0.70** | 📈 Developing | Needs improvement | Early research |

<br>

!!! warning "Important Context"
    These thresholds are **general guidelines**.

    The acceptable score depends on:

    - The specific clinical application
    - Risk level of the use case
    - Whether AI assists or replaces human judgment

    **Always consult domain experts** when evaluating fitness for a particular use case.

<br>

---

## 📏 How We Determine Rankings

Models are ranked following these principles:

<br>

### 1️⃣ Primary metric determines rank

The model with the highest score in the main metric ranks first.

> For metrics where **lower is better** (like error rates), the lowest score wins.

<br>

### 2️⃣ Ties are broken by secondary metrics

If two models have identical primary scores, we look at other relevant metrics.

<br>

### 3️⃣ Best run per model

If a model was evaluated multiple times (e.g., with different settings), only its **best result** appears on the leaderboard.

<br>

### 4️⃣ Reproducibility required

All results must be reproducible. We record:

- Evaluation date
- Dataset used
- Configuration details

<br>

---

## 🏥 Why This Matters for Healthcare AI

Healthcare AI has **higher stakes** than many other AI applications.

> A model that works 95% of the time might sound good, but that 5% could mean **missed diagnoses** or **incorrect treatments**.

<br>

**That's why we:**

✅ Use **multiple metrics** to capture different aspects of performance

✅ Test **robustness** to real-world data quality issues

✅ Require **transparency** about evaluation conditions

✅ Follow **international standards** for healthcare AI assessment

<br>

---

## 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework.

<br>

This ensures our evaluations are:

| Quality | What it means |
|:--------|:--------------|
| **Rigorous** | Following established scientific methodology |
| **Comparable** | Using standardized metrics across models |
| **Trustworthy** | Aligned with WHO/ITU recommendations |

<br>

</details>

---

#### DNA Enhancer Classification

*Benchmark for classifying DNA sequences as enhancers or non-enhancers.
Enhancers are distal regulatory elements that activate gene expression.
Accurate enhancer prediction is critical for understanding gene regulation
and identifying disease-associated variants.
*


<div align="center">

```
                    🏆                    
                                          
              🥇 HyenaDNA              
                 (0.788)                 
             ╔═══════════════╗             
             ║               ║             
   🥈 Caduceus   ║               ║   🥉  Evo 2     
      (0.745)      ║               ║      (0.745)      
  ╔═══════════╝               ╚═══════════╗  
  ║                                       ║  
══╩═══════════════════════════════════════╩══
```

</div>

**6 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **HyenaDNA** 👑 | 0.7883 | 🔶 Fair | DS-DNA-ENHANCER, 2025-12-18T21:03:03.285801 |
| 🥈 | **Caduceus** | 0.7453 | 🔶 Fair | Human Enhancers (Coh, 2025-12-19T12:00:12.636691 |
| 🥉 | **Evo 2** | 0.7453 | 🔶 Fair | Human Enhancers (Coh, 2025-12-19T12:00:13.160707 |
| 🏅 | kmer_k6 | 0.7365 | 🔶 Fair | Human Enhancers (Coh, 2025-12-18T18:44:08.075706 |
| 🏅 | DNABERT-2 | 0.7365 | 🔶 Fair | Human Enhancers (Coh, 2025-12-18T18:44:24.678525 |
| 🎖️ | HyenaDNA | 0.7365 | 🔶 Fair | Human Enhancers (Coh, 2025-12-18T18:44:17.006557 |

!!! tip "Quick Comparison"
    **🥇 HyenaDNA** leads with AUROC = **0.7883**

    - Gap to 🥈 Caduceus: +0.0430
    - Score spread (best to worst): 0.0518


<details class="score-details" markdown="1">
<summary>📐 <strong>How are scores calculated for this benchmark?</strong> (click to expand)</summary>

## 📂 What this leaderboard measures

- **Benchmark:** `BM-DNA-ENHANCER` — DNA Enhancer Classification
- **Domain:** Genetics, Genomics - Regulatory Element Prediction
- **Task type:** Classification
- **Datasets used in the table above:**
  - `DS-DNA-ENHANCER` — DS-DNA-ENHANCER
  - `DS-DNA-ENHANCERS-COHN` — Human Enhancers (Cohn et al.)
- **Typical sample size in these runs:** ~6250 samples (train + test combined)
- **Primary ranking metric:** `AUROC` (the score column in the table)

<br>

---

## 🎯 Primary metric for this leaderboard

- **Metric:** `AUROC`
- **What it measures:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)
- **Typical range:** 0.5 (random guessing) → 1.0 (perfect separation)

> 🔎 For a full explanation of this and other metrics, see the **Metric Cheat Sheet** near the top of this page.

<br>

---

## 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance.

**Here's how this metric should be interpreted for this benchmark:**

<br>

For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand how reliably the model separates different outcome groups.

> 💡 **Tip:** In addition to raw accuracy, look at metrics like **AUROC** and **F1 Score**, especially when classes are imbalanced (when positive cases are rare).

<br>

---

## 📊 Performance Tiers

### What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses.

<br>

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier, consistently reliable | Clinical pilots (with oversight) |
| **0.80 – 0.89** | ✅ Good | Strong performance, real promise | Validation studies |
| **0.70 – 0.79** | 🔶 Fair | Moderate, has limitations | Research only |
| **< 0.70** | 📈 Developing | Needs improvement | Early research |

<br>

!!! warning "Important Context"
    These thresholds are **general guidelines**.

    The acceptable score depends on:

    - The specific clinical application
    - Risk level of the use case
    - Whether AI assists or replaces human judgment

    **Always consult domain experts** when evaluating fitness for a particular use case.

<br>

---

## 📏 How We Determine Rankings

Models are ranked following these principles:

<br>

### 1️⃣ Primary metric determines rank

The model with the highest score in the main metric ranks first.

> For metrics where **lower is better** (like error rates), the lowest score wins.

<br>

### 2️⃣ Ties are broken by secondary metrics

If two models have identical primary scores, we look at other relevant metrics.

<br>

### 3️⃣ Best run per model

If a model was evaluated multiple times (e.g., with different settings), only its **best result** appears on the leaderboard.

<br>

### 4️⃣ Reproducibility required

All results must be reproducible. We record:

- Evaluation date
- Dataset used
- Configuration details

<br>

---

## 🏥 Why This Matters for Healthcare AI

Healthcare AI has **higher stakes** than many other AI applications.

> A model that works 95% of the time might sound good, but that 5% could mean **missed diagnoses** or **incorrect treatments**.

<br>

**That's why we:**

✅ Use **multiple metrics** to capture different aspects of performance

✅ Test **robustness** to real-world data quality issues

✅ Require **transparency** about evaluation conditions

✅ Follow **international standards** for healthcare AI assessment

<br>

---

## 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework.

<br>

This ensures our evaluations are:

| Quality | What it means |
|:--------|:--------------|
| **Rigorous** | Following established scientific methodology |
| **Comparable** | Using standardized metrics across models |
| **Trustworthy** | Aligned with WHO/ITU recommendations |

<br>

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


<details class="score-details" markdown="1">
<summary>📐 <strong>How are scores calculated for this benchmark?</strong> (click to expand)</summary>

## 📂 What this leaderboard measures

- **Benchmark:** `BM-TOY-CLASS` — Toy Classification Benchmark
- **Domain:** Neurology
- **Task type:** Classification
- **Datasets used in the table above:**
  - `DS-TOY-FMRI-CLASS` — Toy fMRI Classification
- **Primary ranking metric:** `AUROC` (the score column in the table)

<br>

---

## 🎯 Primary metric for this leaderboard

- **Metric:** `AUROC`
- **What it measures:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)
- **Typical range:** 0.5 (random guessing) → 1.0 (perfect separation)

> 🔎 For a full explanation of this and other metrics, see the **Metric Cheat Sheet** near the top of this page.

<br>

---

## 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance.

**Here's how this metric should be interpreted for this benchmark:**

<br>

For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand how reliably the model separates different outcome groups.

> 💡 **Tip:** In addition to raw accuracy, look at metrics like **AUROC** and **F1 Score**, especially when classes are imbalanced (when positive cases are rare).

<br>

---

## 📊 Performance Tiers

### What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses.

<br>

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier, consistently reliable | Clinical pilots (with oversight) |
| **0.80 – 0.89** | ✅ Good | Strong performance, real promise | Validation studies |
| **0.70 – 0.79** | 🔶 Fair | Moderate, has limitations | Research only |
| **< 0.70** | 📈 Developing | Needs improvement | Early research |

<br>

!!! warning "Important Context"
    These thresholds are **general guidelines**.

    The acceptable score depends on:

    - The specific clinical application
    - Risk level of the use case
    - Whether AI assists or replaces human judgment

    **Always consult domain experts** when evaluating fitness for a particular use case.

<br>

---

## 📏 How We Determine Rankings

Models are ranked following these principles:

<br>

### 1️⃣ Primary metric determines rank

The model with the highest score in the main metric ranks first.

> For metrics where **lower is better** (like error rates), the lowest score wins.

<br>

### 2️⃣ Ties are broken by secondary metrics

If two models have identical primary scores, we look at other relevant metrics.

<br>

### 3️⃣ Best run per model

If a model was evaluated multiple times (e.g., with different settings), only its **best result** appears on the leaderboard.

<br>

### 4️⃣ Reproducibility required

All results must be reproducible. We record:

- Evaluation date
- Dataset used
- Configuration details

<br>

---

## 🏥 Why This Matters for Healthcare AI

Healthcare AI has **higher stakes** than many other AI applications.

> A model that works 95% of the time might sound good, but that 5% could mean **missed diagnoses** or **incorrect treatments**.

<br>

**That's why we:**

✅ Use **multiple metrics** to capture different aspects of performance

✅ Test **robustness** to real-world data quality issues

✅ Require **transparency** about evaluation conditions

✅ Follow **international standards** for healthcare AI assessment

<br>

---

## 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework.

<br>

This ensures our evaluations are:

| Quality | What it means |
|:--------|:--------------|
| **Rigorous** | Following established scientific methodology |
| **Comparable** | Using standardized metrics across models |
| **Trustworthy** | Aligned with WHO/ITU recommendations |

<br>

</details>

---

### 📋 Classification/Reconstruction

#### fMRI Foundation Model Benchmark (Granular)

**2 models ranked by `AUROC`:**

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **BrainLM** 👑 | 1.0000 | ⭐ Excellent | DS-TOY-FMRI, 2025-12-19T12:00:49.423857 |
| 🥈 | **Brain-JEPA** | 1.0000 | ⭐ Excellent | DS-TOY-FMRI, 2025-12-19T12:00:49.427678 |

!!! tip "Quick Comparison"
    **🥇 BrainLM** leads with AUROC = **1.0000**

    - Gap to 🥈 Brain-JEPA: +0.0000


<details class="score-details" markdown="1">
<summary>📐 <strong>How are scores calculated for this benchmark?</strong> (click to expand)</summary>

## 📂 What this leaderboard measures

- **Benchmark:** `BM-FMRI-GRANULAR` — fMRI Foundation Model Benchmark (Granular)
- **Domain:** Neurology, Functional Brain Imaging Analysis
- **Task type:** Classification/Reconstruction
- **Datasets used in the table above:**
  - `DS-TOY-FMRI` — DS-TOY-FMRI
- **Typical sample size in these runs:** ~200 samples (train + test combined)
- **Primary ranking metric:** `AUROC` (the score column in the table)

<br>

---

## 🎯 Primary metric for this leaderboard

- **Metric:** `AUROC`
- **What it measures:** Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)
- **Typical range:** 0.5 (random guessing) → 1.0 (perfect separation)

> 🔎 For a full explanation of this and other metrics, see the **Metric Cheat Sheet** near the top of this page.

<br>

---

## 🧠 How This Metric Fits This Task

Different tasks emphasize different aspects of performance.

**Here's how this metric should be interpreted for this benchmark:**

<br>

For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand how reliably the model separates different outcome groups.

> 💡 **Tip:** In addition to raw accuracy, look at metrics like **AUROC** and **F1 Score**, especially when classes are imbalanced (when positive cases are rare).

<br>

---

## 📊 Performance Tiers

### What Do the Scores Mean?

We group models into performance tiers to help you quickly understand how ready they are for different uses.

<br>

| Score Range | Rating | Interpretation | Suitable For |
|:---:|:---:|:---|:---|
| **≥ 0.90** | ⭐ Excellent | Top-tier, consistently reliable | Clinical pilots (with oversight) |
| **0.80 – 0.89** | ✅ Good | Strong performance, real promise | Validation studies |
| **0.70 – 0.79** | 🔶 Fair | Moderate, has limitations | Research only |
| **< 0.70** | 📈 Developing | Needs improvement | Early research |

<br>

!!! warning "Important Context"
    These thresholds are **general guidelines**.

    The acceptable score depends on:

    - The specific clinical application
    - Risk level of the use case
    - Whether AI assists or replaces human judgment

    **Always consult domain experts** when evaluating fitness for a particular use case.

<br>

---

## 📏 How We Determine Rankings

Models are ranked following these principles:

<br>

### 1️⃣ Primary metric determines rank

The model with the highest score in the main metric ranks first.

> For metrics where **lower is better** (like error rates), the lowest score wins.

<br>

### 2️⃣ Ties are broken by secondary metrics

If two models have identical primary scores, we look at other relevant metrics.

<br>

### 3️⃣ Best run per model

If a model was evaluated multiple times (e.g., with different settings), only its **best result** appears on the leaderboard.

<br>

### 4️⃣ Reproducibility required

All results must be reproducible. We record:

- Evaluation date
- Dataset used
- Configuration details

<br>

---

## 🏥 Why This Matters for Healthcare AI

Healthcare AI has **higher stakes** than many other AI applications.

> A model that works 95% of the time might sound good, but that 5% could mean **missed diagnoses** or **incorrect treatments**.

<br>

**That's why we:**

✅ Use **multiple metrics** to capture different aspects of performance

✅ Test **robustness** to real-world data quality issues

✅ Require **transparency** about evaluation conditions

✅ Follow **international standards** for healthcare AI assessment

<br>

---

## 🌍 Standards Alignment

This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) framework.

<br>

This ensures our evaluations are:

| Quality | What it means |
|:--------|:--------------|
| **Rigorous** | Following established scientific methodology |
| **Comparable** | Using standardized metrics across models |
| **Trustworthy** | Aligned with WHO/ITU recommendations |

<br>

</details>

---

### 🔄 Reconstruction

#### Brain Time-Series Modeling

*Evaluating ability to reconstruct masked fMRI voxel time-series.*

!!! warning "No submissions yet"
    Be the first! See [Submission Guide](../contributing/submission_guide.md)

## 📋 Other Benchmarks

### Foundation Model Robustness Evaluation

| Rank | Model | Score | Level | Details |
|:---:|:---|:---:|:---:|:---|
| 🥇 | **geneformer** 👑 | 0.9995 | ⭐ Excellent | neuro/robustness, 2025-11-27 |
| 🥈 | **BrainLM** | 0.9451 | ⭐ Excellent | DS-TOY-FMRI-ROBUSTNE, 2025-12-19T12:01:52.781177 |
| 🥉 | **Brain-JEPA** | 0.9377 | ⭐ Excellent | DS-TOY-FMRI-ROBUSTNE, 2025-12-19T12:01:52.789369 |
| 🏅 | SWIFT | 0.9234 | ⭐ Excellent | DS-TOY-FMRI-ROBUSTNE, 2025-12-18T21:25:36.388271 |
| 🏅 | Baseline (Random/Majority) | 0.7810 | 🔶 Fair | neuro/robustness, 2025-11-27 |

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
