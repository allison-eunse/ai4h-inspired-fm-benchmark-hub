# AI4H-Inspired FM Benchmarks

**Standardized Evaluation for Neurogenomic Foundation Models**

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://allison-eunse.github.io/ai4h-inspired-fm-benchmark-hub)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## Mission

This repository hosts an open benchmarking hub for **Genetics** and **Brain Imaging** foundation models. Our framework is directly inspired by the principles and deliverables of the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H), specifically aiming for rigorous, transparent, and clinically relevant evaluation standards.

We gratefully acknowledge the work of the ITU-T Focus Group on AI for Health as the foundation for our evaluation methodology.

## Features

### 1. Interactive Evaluation Tool
We provide a downloadable evaluation suite that allows researchers to test their models against standardized baselines, following the **AI4H System Requirement Specifications (DEL3)**.

*   **Toy Samples**: The tool provides representative "toy" datasets for initial testing and debugging.
*   **Local Evaluation**: Run evaluations on your own infrastructure to ensure data privacy and reproducibility.
*   **Automated Reporting**: The tool analyzes your model's outputs and generates a detailed performance report, characterizing your model's functions and capabilities.

### 2. Comprehensive Leaderboards
We maintain leaderboards for existing open-source Foundation Models (FMs), ranking them not just by a single score, but by specific task performance and granular parameters defined in the **Topic Description Documents (DEL10.x)**.

*   **Task-Specific Rankings**: Separate leaderboards for distinct tasks like fMRI classification, single-cell variant interpretation, etc.
*   **Granular Metrics**:
    *   **Data Sub-types**: Performance breakdowns by specific data characteristics (e.g., specific fMRI scanner types, sequencing depth).
    *   **Output Quality**: Metrics for report generation quality, accuracy, and clinical utility.
    *   **Resource Usage**: Inference time and computational cost.

### 3. Robustness Testing

Built-in robustness probes (inspired by brainaug-lab methodology) test foundation model resilience against realistic perturbations:

*   **Channel Dropout**: Simulates sensor failures and missing data
*   **Gaussian Noise**: Tests performance under various SNR conditions
*   **Line Interference**: Evaluates robustness to powerline artifacts (50/60 Hz)
*   **Channel Permutation**: Tests equivariance to electrode/channel ordering
*   **Temporal Shift**: Measures sensitivity to timing jitter

This produces **rAUC (Reverse Area Under Curve)** scores that quantify how stable model outputs remain as perturbations increase.

### 4. Domain Focus

*   🧬 **Genomics**: Single-cell analysis, variant interpretation, gene expression modeling.
*   🧠 **Neurology**: Brain MRI/fMRI/EEG analysis, neurodegenerative disease classification, reconstruction.

## Alignment with ITU/WHO FG-AI4H Standards

This project explicitly references and adapts the following ITU-T Focus Group deliverables:

*   **[DEL3: AI4H Requirement Specifications](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf)**: We adopt the **System Requirement Specifications (SyRS)** framework for defining functional, operational, and performance requirements for benchmarked models.
*   **[DEL0.1: Common Unified Terms](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2022-1-PDF-E.pdf)**: We utilize the standard terminology (e.g., "AI Solution", "Benchmarking Run") to ensure consistency across the healthcare AI domain.
*   **[DEL10.8: Topic Description Document for Neurology](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf)**: Our Neurology benchmarks are structured according to the TDD template, defining the health topic, scope, input data, and evaluation metrics (e.g., for TG-Neuro).

## Getting Started

For full documentation, including installation instructions, benchmark definitions, and current leaderboards, please visit our [Documentation Site](https://allison-eunse.github.io/ai4h-inspired-fm-benchmark-hub).

### Quick Start

```bash
# Clone the repository
git clone https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub.git
cd ai4h-inspired-fm-benchmark-hub

# Install dependencies
pip install -r requirements.txt

# 1. Generate toy data (includes robustness test data)
python -m fmbench generate-toy-data

# 2. Run a benchmark suite (using the built-in dummy model)
python -m fmbench run \
    --suite SUITE-TOY-CLASS \
    --model configs/model_dummy_classifier.yaml \
    --out results/my_first_run

# 3. Build and view the leaderboard
python -m fmbench build-leaderboard
# Open docs/leaderboards/index.md to see your results!
```

### Robustness Testing

Test how your model handles noise, artifacts, and perturbations:

```bash
# Run robustness evaluation (no additional dependencies needed!)
python -m fmbench run-robustness \
    --model configs/model_dummy_classifier.yaml \
    --data toy_data/neuro/robustness \
    --out results/robustness_eval \
    --probes dropout,noise,line_noise,permutation,shift

# Check the robustness report
cat results/robustness_eval/report.md
```

The robustness evaluation produces:
- **rAUC scores** for each perturbation type (higher = more robust)
- **Similarity curves** showing output stability vs perturbation strength  
- **Aggregate robustness score** for overall comparison

## Credits & Attribution

The methodology and framework of this benchmark suite are derived from the public deliverables of the **ITU/WHO Focus Group on Artificial Intelligence for Health (FG-AI4H)**.

*   **DEL3**: *AI4H requirement specifications* (03/2023) - [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf)
*   **DEL0.1**: *Common unified terms in artificial intelligence for health* (2022) - [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2022-1-PDF-E.pdf)
*   **DEL10.8**: *Topic Description Document for the Topic Group on AI for neurological disorders (TG-Neuro)* (2023) - [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf)

(c) ITU 2025. These publications are available under the Creative Commons Attribution-Non Commercial-Share Alike 3.0 IGO licence (CC BY-NC-SA 3.0 IGO).
