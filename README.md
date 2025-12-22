# AI4H-Inspired FM Benchmarks

**Standardized Evaluation for Neurogenomic Foundation Models**

[![Documentation](https://img.shields.io/badge/docs-mkdocs-blue)](https://allison-eunse.github.io/ai4h-inspired-fm-benchmark-hub)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Leaderboard](https://img.shields.io/badge/leaderboard-live-success)](https://allison-eunse.github.io/ai4h-inspired-fm-benchmark-hub/leaderboards/)

<p align="center">
  <strong>🧬 Genomics</strong> • <strong>🧠 Brain Imaging</strong> • <strong>🔒 Privacy-Preserving</strong> • <strong>🤖 Fully Automated</strong>
</p>

---

## 🚀 Quick Start (30 seconds)

```bash
git clone https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub.git
cd ai4h-inspired-fm-benchmark-hub
pip install -e .
python -m fmbench generate-toy-data
python -m fmbench run --suite SUITE-TOY-CLASS --model configs/model_dummy_classifier.yaml --out results/test
```

**That's it!** You now have `results/test/eval.yaml` ready for submission.

---

## Mission

This repository hosts an open benchmarking hub for **Genetics** and **Brain Imaging** foundation models. Our framework is directly inspired by the principles and deliverables of the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H), specifically aiming for rigorous, transparent, and clinically relevant evaluation standards.

We gratefully acknowledge the work of the ITU-T Focus Group on AI for Health as the foundation for our evaluation methodology.

## Features

### 1. Interactive Evaluation Tool
We provide a downloadable evaluation suite that allows researchers to test their models against standardized baselines, following the **AI4H System Requirement Specifications (DEL3)**.

*   **Toy Samples**: The tool provides representative "toy" datasets for initial testing and debugging. See [Toy Data Disclaimer](#️-toy-data-disclaimer) for details on sample sizes.
*   **Local Evaluation**: Run evaluations on your own infrastructure to ensure data privacy and reproducibility.
*   **Automated Reporting**: The tool analyzes your model's outputs and generates a detailed performance report, characterizing your model's functions and capabilities.

### 2. Comprehensive Leaderboards
We maintain leaderboards for existing open-source Foundation Models (FMs), ranking them not just by a single score, but by specific task performance and granular parameters defined in the **Topic Description Documents (DEL10.x)**.

*   **Task-Specific Rankings**: Separate leaderboards for distinct tasks like fMRI classification, single-cell variant interpretation, etc.
*   **Granular Metrics**:
    *   **Data Sub-types**: Performance breakdowns by specific data characteristics (e.g., specific fMRI scanner types, sequencing depth).
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

*   🧬 **Genomics**: Single-cell analysis, DNA sequence modeling, gene expression analysis.
*   🧠 **Neurology**: Brain fMRI/sMRI analysis, robustness testing, representation quality evaluation.

## Alignment with ITU/WHO FG-AI4H Standards

This project explicitly references and adapts the following ITU-T Focus Group deliverables:

*   **[DEL3: AI4H Requirement Specifications](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf)**: We adopt the **System Requirement Specifications (SyRS)** framework for defining functional, operational, and performance requirements for benchmarked models.
*   **[DEL0.1: Common Unified Terms](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2022-1-PDF-E.pdf)**: We utilize the standard terminology (e.g., "AI Solution", "Benchmarking Run") to ensure consistency across the healthcare AI domain.
*   **[DEL10.8: Topic Description Document for Neurology](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf)**: Our Neurology benchmarks are structured according to the TDD template, defining the health topic, scope, input data, and evaluation metrics (e.g., for TG-Neuro).

## 🔒 Privacy-Preserving Evaluation (AI4H DEL3 Aligned)

**You don't need to share your model to appear on our leaderboard!**

Our framework follows the [ITU/WHO FG-AI4H DEL3](https://www.itu.int/pub/T-FG-AI4H) principle of **local evaluation with standardized result reporting**:

```
┌─────────────────────────────────────────┐
│  YOUR MACHINE (Private)                 │
│                                         │
│  1. Download fmbench toolkit            │
│  2. Wrap your model (simple interface)  │
│  3. Run benchmarks locally              │
│  4. Submit ONLY metrics (eval.yaml)     │
│                                         │
└────────────────┬────────────────────────┘
                 │  Submit metrics only
                 ▼
┌─────────────────────────────────────────┐
│  LEADERBOARD (Public)                   │
│                                         │
│  ✅ Receives: metrics, metadata         │
│  ❌ NOT received: weights, code, data   │
│                                         │
└─────────────────────────────────────────┘
```

### Quick Start for Researchers

```bash
# 1. Install toolkit
pip install -e .

# 2. Generate test data
python -m fmbench generate-toy-data

# 3. Create a simple wrapper for your model
cat > my_model.py << 'EOF'
import numpy as np

class MyModelWrapper:
    def __init__(self):
        # Load YOUR model here (stays private)
        pass
    
    def predict(self, X):
        # Your inference code
        return predictions
    
    def predict_proba(self, X):
        # Your probability predictions
        return probabilities
EOF

# 4. Create config
cat > my_config.yaml << 'EOF'
model_id: my_awesome_model
type: python_class
import_path: "my_model:MyModelWrapper"
EOF

# 5. Run benchmark locally
python -m fmbench run --suite SUITE-TOY-CLASS --model my_config.yaml --out results/

# 6. Submit results/eval.yaml via GitHub Issue
```

**Your model weights and code NEVER leave your machine!**

---

## ⚠️ Important: Data Disclaimer

> **This repository does NOT include production-scale datasets.**
> 
> We provide only **toy data** (small subsamples) for testing your pipeline. Full-scale benchmarking requires you to download datasets from their original sources.

### What's Included vs What You Need to Download

| What | Included in Repo? | What to Expect |
|------|-------------------|----------------|
| **Toy data** | ✅ Yes | Small samples (100–27,000) for pipeline validation |
| **Full genomics datasets** | ❌ No | Download from HuggingFace (links below) |
| **Brain imaging data** | ❌ No | Requires institutional access (UK Biobank, HCP, etc.) |
| **Evaluation framework** | ✅ Yes | CLI tools, adapters, metrics, leaderboard |

### Expected Outcomes

| Using Toy Data | Using Full Data |
|----------------|-----------------|
| ✅ Verify your model wrapper works | ✅ Get publishable benchmark scores |
| ✅ Test the evaluation pipeline | ✅ Compare fairly against leaderboard |
| ✅ Debug integration issues | ✅ Produce stable, reproducible metrics |
| ⚠️ Metrics have high variance | ⚠️ Requires more compute time |
| ⚠️ Not suitable for publication | ⚠️ May require downloading 1–10GB |

### Genomics Data Sources

| Dataset | Toy Samples | Full Size | Download From |
|---------|-------------|-----------|---------------|
| `enhancers_cohn` | 20,843 ✅ | 20,843 (complete) | [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks) |
| `promoters_nontata` | 27,097 ✅ | 27,097 (complete) | [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks) |
| `nucleotide_transformer` | 1,500 | 493,242 | [InstaDeepAI](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised) |
| `regulatory_ensembl` | 1,500 | 231,348 | [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks) |
| `open_chromatin` | 1,000 | 139,804 | [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks) |

### Brain Imaging Data

| Dataset | Included Samples | Notes |
|---------|------------------|-------|
| `fmri_classification` | 200 (synthetic) | Real fMRI requires UK Biobank or HCP access |
| `eeg_classification` | 150 (synthetic) | For pipeline testing only |
| `robustness_probes` | 100 (synthetic) | For robustness testing |

> **Why no real brain imaging data?**  
> Datasets like UK Biobank and HCP have strict Data Use Agreements (DUAs) that prohibit redistribution. If you have institutional access, you can run this framework on your own data locally.

### Recommended Workflow

```
1. VALIDATE PIPELINE (toy data, included)
   └─► python -m fmbench run --suite SUITE-TOY-CLASS ...
   └─► Expected: Quick run, verify outputs work
   
2. FULL BENCHMARKING (external data, download yourself)
   └─► Download full datasets from HuggingFace
   └─► Point fmbench to your local data directory
   └─► Expected: Production-quality metrics for publication
```

### Reproducibility

- All toy data subsampling uses **fixed random seed (42)**
- Re-running produces identical subsamples
- Full datasets available from original sources (linked above)

---

## 🛠️ Installation

### Option A: pip (quick start)

```bash
git clone https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub.git
cd ai4h-inspired-fm-benchmark-hub
pip install -e .
```

### Option B: conda (recommended for Geneformer/genomics FMs)

```bash
git clone https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub.git
cd ai4h-inspired-fm-benchmark-hub

# Create environment with all dependencies
conda env create -f environment.yml
conda activate fmbench

# Or use the activation script (sets all environment variables)
source scripts/activate_fmbench.sh
```

The conda environment includes:
- Python 3.11 (compatible with numba/Geneformer)
- PyTorch, Transformers, Scanpy
- All genomics FM dependencies

---

## Getting Started

For full documentation, visit our [Documentation Site](https://allison-eunse.github.io/ai4h-inspired-fm-benchmark-hub).

### Run Your First Benchmark

```bash

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

## 📤 Submit Your Model (Fully Automated)

Our leaderboard updates **automatically** — no manual review delay!

```
┌─────────────────────────────────────────────────────────────────────────┐
│  YOUR MACHINE                        │  GITHUB (automated)              │
│                                      │                                  │
│  1. python -m fmbench run ...        │                                  │
│         ↓                            │                                  │
│  2. eval.yaml generated              │                                  │
│         ↓                            │                                  │
│  3. Open Issue, paste YAML    ──────►│  4. Bot validates & commits      │
│                                      │         ↓                        │
│                                      │  5. Leaderboard rebuilds         │
│                                      │         ↓                        │
│                                      │  6. Live on GitHub Pages! 🎉     │
└─────────────────────────────────────────────────────────────────────────┘
```

### How to submit

```bash
# 1. Run locally
python -m fmbench run --suite SUITE-GEN-CLASS-001 --model my_config.yaml --out results/

# 2. Open a GitHub Issue:
#    https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=benchmark_submission.md

# 3. Paste your eval.yaml in the YAML code block

# 4. Submit — done! Leaderboard updates in ~2-3 minutes
```

**Your model weights and code NEVER leave your machine!**

### Propose New Evaluation Protocols
Have ideas for new benchmarks? Open a [Discussion](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/discussions).

## Credits & Attribution

See [CREDITS.md](CREDITS.md) for full citations and acknowledgments.

### Data Sources

| Data | Source | Citation |
|------|--------|----------|
| **DNA Benchmarks** | [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks) | [Grešová et al. (2022)](https://www.biorxiv.org/content/10.1101/2022.06.08.495248) |
| **NT Benchmark** | [InstaDeepAI](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised) | [Dalla-Torre et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.01.11.523679) |
| **PBMC scRNA-seq** | [10x Genomics](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0) | 10x Genomics (2016) |
| **Regulatory Data** | [ENCODE](https://www.encodeproject.org/) | ENCODE Consortium (2012) |

### Framework

The methodology and framework of this benchmark suite are derived from the public deliverables of the **ITU/WHO Focus Group on Artificial Intelligence for Health (FG-AI4H)**.

*   **DEL3**: *AI4H requirement specifications* (03/2023) - [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-11-PDF-E.pdf)
*   **DEL0.1**: *Common unified terms in artificial intelligence for health* (2022) - [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2022-1-PDF-E.pdf)
*   **DEL10.8**: *Topic Description Document for the Topic Group on AI for neurological disorders (TG-Neuro)* (2023) - [PDF](https://www.itu.int/dms_pub/itu-t/opb/fg/T-FG-AI4H-2023-20-PDF-E.pdf)

(c) ITU 2025. These publications are available under the Creative Commons Attribution-Non Commercial-Share Alike 3.0 IGO licence (CC BY-NC-SA 3.0 IGO).
