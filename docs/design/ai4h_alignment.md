# AI4H Alignment

## Overview

This benchmark hub is explicitly designed to align with the **ITU/WHO Focus Group on Artificial Intelligence for Health (FG-AI4H)** standards and deliverables. This page documents how our framework implements and extends these standards for foundation model evaluation.

---

## Quick mapping: deliverables → concrete implementation

| AI4H deliverable / concept | What it means (short) | Where it shows up in this hub |
|---|---|---|
| **DEL3 — System Requirement Specifications (SyRS)** | Define what an evaluation must do and how it’s validated | **Benchmark definitions** in `benchmarks/*.yaml` (functional requirements + metrics), and **runner outputs** (`report.md`, `eval.yaml`) produced by `fmbench` |
| **DEL0.1 — Common unified terms** | Shared vocabulary so runs are comparable | Consistent IDs/terms across YAMLs: `benchmark_id`, `dataset_id`, `model_id`, `eval_id` used in `benchmarks/`, `datasets/`, `models/`, `evals/` and the docs |
| **DEL10.8 — Neurology TDD** | Neurology benchmark structure: topic, scope, inputs, metrics | Neurology-style benchmark schemas reflected in `benchmarks/*.yaml` and supported by stratified metrics in `evals/*.yaml` (`metrics.stratified`) |
| **DEL7.x / test specifications (in practice)** | A runnable test suite definition | **Suites** in `tests/suite_*.yaml` (e.g., `SUITE-TOY-CLASS`) that define how to run + what artifacts should be produced |

## ITU FG-AI4H Background

The [ITU/WHO Focus Group on AI for Health](https://www.itu.int/en/ITU-T/focusgroups/ai4h/Pages/default.aspx) was established to develop international standards for AI in healthcare, focusing on:

- **Safety**: Ensuring AI systems are safe for clinical use
- **Effectiveness**: Establishing evidence-based validation methods
- **Transparency**: Promoting explainability and interpretability
- **Ethics**: Addressing fairness, bias, and patient rights
- **Interoperability**: Enabling cross-system compatibility

## Key Deliverables Used

### DEL0.1: Common Unified Terms

**Reference**: [ITU-T FG-AI4H-DEL0.1 (2022)](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2022-1-PDF-E.pdf)

**Purpose**: Establish standardized terminology for AI4H systems

**Our Implementation**:

| AI4H Term | Our Usage |
|-----------|-----------|
| **AI Solution** | Foundation models (e.g., BrainLM, Geneformer) |
| **Benchmarking Run** | Evaluation instance (`eval_id` in results) |
| **Reference Implementation** | Baseline models (logistic regression, random forest) |
| **Health Topic** | Domain area (e.g., "Functional Brain Imaging", "Genomics") |
| **AI Task** | ML task type (classification, reconstruction, regression) |
| **Test Dataset** | Standardized evaluation data (e.g., PBMC 3k, HCP fMRI) |

**Example from our schema**:

```yaml
# From benchmarks/bm_fmri_granular.yaml
benchmark_id: BM-FMRI-GRANULAR
health_topic: Functional Brain Imaging Analysis
ai_task: Classification/Reconstruction
```

---

### DEL3: AI4H Requirement Specifications

**Reference**: [ITU-T FG-AI4H-DEL3 (2023)](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf)

**Purpose**: Define System Requirements Specification (SyRS) framework for AI4H systems

**Our Implementation**:

#### 1. Functional Requirements (DEL3 Section 4)

We define functional requirements for each benchmark:

```yaml
# Example: Cell type annotation benchmark
benchmark_id: CELL-TYPE-ANNOTATION
functional_requirements:
  - input: Single-cell RNA-seq count matrix
  - output: Cell type labels from standardized ontology
  - performance_threshold: F1 > 0.80 (vs random baseline)
```

#### 2. Performance Requirements (DEL3 Section 6)

Our leaderboards track multiple performance dimensions:

- **Accuracy metrics**: AUROC, F1-score, balanced accuracy
- **Robustness**: rAUC scores under perturbations
- **Resource usage**: Inference time, memory footprint
- **Fairness**: Stratified performance by demographic groups

**Example**:

```python
# From fmbench/runners.py
metrics = {
    'auroc': roc_auc_score(y_true, y_prob),
    'accuracy': accuracy_score(y_true, y_pred),
    'f1_score': f1_score(y_true, y_pred, average='weighted'),
    'stratified': {
        'by_age': compute_stratified_metrics(y_true, y_pred, age_groups),
        'by_sex': compute_stratified_metrics(y_true, y_pred, sex_groups),
    }
}
```

#### 3. Data Requirements (DEL3 Section 4.2)

We enforce standardized data formats:

- **Neuroimaging**: NIfTI, preprocessed per [fMRI specs](../integration/modality_features/fmri.md)
- **Genomics**: AnnData (scRNA-seq), FASTA (DNA), VCF (variants)
- **Metadata**: YAML schema with required fields

```yaml
# From datasets/*.yaml
dataset_id: pbmc_68k
name: PBMC 68k
modality: scRNA-seq
n_samples: 68579
preprocessing: scanpy_1.9.1
quality_control:
  min_genes_per_cell: 200
  max_pct_mito: 5
```

#### 4. Validation & Verification (DEL3 Section 5)

Our framework includes:

- ✅ **Cross-validation**: Stratified k-fold for robust estimates
- ✅ **Baseline comparison**: Always compare to random, majority, linear baselines
- ✅ **Statistical significance**: Permutation testing, confidence intervals
- ✅ **Confound control**: Partial correlations, matched controls

See our [analysis recipes](../integration/analysis_recipes/cca_permutation.md).

---

### DEL10.8: Topic Description Document for Neurology

**Reference**: [ITU-T FG-AI4H-DEL10.8 (2023)](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf)

**Purpose**: Define neurology-specific evaluation standards for the TG-Neuro topic group

**Our Implementation**:

#### Benchmark Structure (Following TDD Template)

Each neurology benchmark follows the TDD structure:

```yaml
# Example: bm_fmri_granular.yaml
benchmark_id: BM-FMRI-GRANULAR
name: fMRI Foundation Model Benchmark (Granular)

# 1. Health Topic (TDD Section 2)
health_topic: Functional Brain Imaging Analysis
health_domain: Neurology

# 2. Scope (TDD Section 3)
scope:
  clinical_context: "Evaluating FM robustness and representation quality"
  population: General population
  
# 3. Input Data (TDD Section 4)
inputs:
  dataset:
    modality: fMRI
    sequence: BOLD
    preprocessing: fMRIPrep or HCP Pipelines
  
# 4. Evaluation Metrics (TDD Section 5)
metrics:
  primary: AUROC
  secondary:
    - Accuracy
    - F1-Score
    - Robustness rAUC
  stratification:
    - scanner
    - preprocessing_pipeline
    - acquisition_type

# 5. Clinical Relevance (TDD Section 6)
clinical_relevance:
  use_case: FM robustness and generalization testing
  impact: Reliable brain imaging AI systems
```

#### Stratified Evaluation

DEL10.8 emphasizes evaluation across patient subgroups. Our framework automatically computes stratified metrics:

```python
# From fmbench/runners.py
def compute_stratified_metrics(y_true, y_pred, groups):
    """
    Compute metrics for each subgroup.
    
    Aligns with DEL10.8 Section 5.3: Subgroup analysis.
    """
    stratified = {}
    
    for group_name in np.unique(groups):
        mask = groups == group_name
        
        if mask.sum() > 10:  # Minimum sample size
            stratified[group_name] = {
                'accuracy': accuracy_score(y_true[mask], y_pred[mask]),
                'n_samples': mask.sum(),
            }
    
    return stratified
```

**Example output**:

```yaml
metrics:
  accuracy: 0.85
  auroc: 0.91
  stratified:
    by_age_group:
      '55-65': {accuracy: 0.88, n_samples: 120}
      '65-75': {accuracy: 0.85, n_samples: 200}
      '75+': {accuracy: 0.80, n_samples: 80}
    by_sex:
      M: {accuracy: 0.84, n_samples: 180}
      F: {accuracy: 0.86, n_samples: 220}
```

---

## Extensions Beyond AI4H Standards

While we align with AI4H deliverables, we extend the framework to address foundation model-specific challenges:

### 1. Robustness Testing

**Motivation**: Foundation models must handle real-world data variability (noise, artifacts, missing data).

**Our Framework**: 
- Inspired by [brainaug-lab](https://github.com/anonymous/brainaug-lab) methodology
- Tests model resilience to controlled perturbations
- Produces **rAUC (Reverse Area Under Curve)** scores

**Probes**:
- Channel dropout (missing sensors)
- Gaussian noise (SNR variation)
- Line noise (50/60 Hz artifacts)
- Channel permutation (equivariance test)
- Temporal shift (timing jitter)

```bash
# Run robustness evaluation
python -m fmbench run-robustness \
    --model configs/model_brainlm.yaml \
    --data toy_data/neuro/robustness \
    --out results/robustness_eval
```

See: [Robustness documentation](../index.md#robustness-testing)

### 2. Multi-Modal Evaluation

**Challenge**: Many foundation models integrate multiple data types (e.g., imaging + genomics).

**Our Approach**:
- CCA-based cross-modal alignment testing
- Multi-modal fusion benchmarks
- Modality-specific and joint performance metrics

See: [CCA & Permutation Testing](../integration/analysis_recipes/cca_permutation.md)

### 3. Interpretability & Explainability

**Planned Features** (aligned with DEL3 Section 8):
- Attention map visualization
- Feature attribution (SHAP, Integrated Gradients)
- Embedding space interpretability

---

## Compliance Checklist

Use this checklist to verify AI4H alignment for new benchmarks:

- [ ] **Terminology**: Uses standardized AI4H terms (DEL0.1)
- [ ] **Health Topic**: Clearly defined clinical context (DEL10.8 Section 2)
- [ ] **Input Specs**: Documented data format and preprocessing (DEL3 Section 4.2)
- [ ] **Metrics**: Primary and secondary metrics defined (DEL3 Section 6)
- [ ] **Baselines**: Comparison to reference implementations (DEL3 Section 7)
- [ ] **Stratification**: Performance across relevant subgroups (DEL10.8 Section 5.3)
- [ ] **Clinical Relevance**: Justification for clinical use case (DEL10.8 Section 6)
- [ ] **Reproducibility**: Code, data, and results publicly available

---

## Governance & Contribution

### Adding New Benchmarks

To propose a new benchmark aligned with AI4H standards:

1. **Define the Health Topic** (following DEL10.8 template)
2. **Specify Input/Output** (following DEL3 Section 4)
3. **Choose Metrics** (primary + secondary, with clinical justification)
4. **Implement Reference Baselines** (see [Prediction Baselines](../integration/analysis_recipes/prediction_baselines.md))
5. **Document Clinical Relevance**
6. **Submit PR** with benchmark YAML + documentation

### Citing AI4H Deliverables

When publishing results from this benchmark hub, please cite:

```bibtex
@techreport{itu_ai4h_del3_2023,
  title = {AI4H requirement specifications},
  author = {{ITU-T Focus Group on AI for Health}},
  year = {2023},
  institution = {International Telecommunication Union},
  number = {FG-AI4H-DEL3},
  url = {https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf}
}

@techreport{itu_ai4h_del10_8_2023,
  title = {Topic Description Document for the Topic Group on AI for neurological disorders (TG-Neuro)},
  author = {{ITU-T Focus Group on AI for Health}},
  year = {2023},
  institution = {International Telecommunication Union},
  number = {FG-AI4H-DEL10.8},
  url = {https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf}
}
```

---

## References

1. ITU-T Focus Group on AI for Health. (2022). *Common unified terms in artificial intelligence for health* (DEL0.1). [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2022-1-PDF-E.pdf)

2. ITU-T Focus Group on AI for Health. (2023). *AI4H requirement specifications* (DEL3). [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf)

3. ITU-T Focus Group on AI for Health. (2023). *Topic Description Document for the Topic Group on AI for neurological disorders (TG-Neuro)* (DEL10.8). [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf)

4. Wiegand, T., et al. (2019). WHO and ITU establish benchmarking process for artificial intelligence in health. *The Lancet*, 394(10192), 9-11.

---

## Contact & Feedback

For questions about AI4H alignment or to suggest improvements:

- **GitHub Issues**: [Report an issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues)
- **ITU FG-AI4H**: [Official website](https://www.itu.int/en/ITU-T/focusgroups/ai4h/Pages/default.aspx)

---

*© ITU 2025. AI4H deliverables are available under the Creative Commons Attribution-Non Commercial-Share Alike 3.0 IGO licence (CC BY-NC-SA 3.0 IGO).*
