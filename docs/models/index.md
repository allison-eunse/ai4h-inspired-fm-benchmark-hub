# Foundation Models Catalog

This page provides an overview of the foundation models evaluated in this benchmark hub. Each model is defined by a YAML configuration file in the `models/` directory.

## 🧠 Neurology / Brain Imaging Models

### BrainLM

**Model ID**: `brainlm`  
**Modality**: fMRI (Brain functional imaging)  
**Architecture**: ViT-MAE + Nystromformer encoder/decoder  
**Parameters**: 111M / 650M  
**Repository**: [github.com/vandijklab/BrainLM](https://github.com/vandijklab/BrainLM)

Masked autoencoding language model for fMRI voxel time-series. Uses ViT-MAE scaffolding with custom BrainLM embeddings that mix voxel coordinates and patched time windows to learn denoised cortical dynamics.

---

### BrainJEPA

**Model ID**: `brainjepa`  
**Modality**: fMRI, EEG  
**Repository**: [Brain-JEPA Repository](https://github.com/brain-jepa)

Joint-Embedding Predictive Architecture for brain signals. Self-supervised learning approach that learns representations by predicting latent representations rather than raw pixels/signals.

---

### BrainMT

**Model ID**: `brainmt`  
**Modality**: Multi-modal brain imaging  
**Repository**: [BrainMT Repository](https://github.com/brainmt)

Multi-modal brain transformer for integrating structural and functional brain imaging data.

---

### BrainHarmony

**Model ID**: `brainharmony`  
**Modality**: Multi-site neuroimaging  
**Repository**: [BrainHarmony Repository](https://github.com/brainharmony)

Harmonization framework for multi-site neuroimaging studies, addressing scanner and acquisition protocol variability.

---
### SwiFT

**Model ID**: `swift`  
**Modality**: fMRI (4D volumes / time series)  
**Repository**: [github.com/Transconnectome/SwiFT](https://github.com/Transconnectome/SwiFT)

Swin 4D fMRI Transformer for learning representations from spatiotemporal fMRI sequences.

---

### NeuroClips

**Model ID**: `neuroclips`  
**Modality**: fMRI → video reconstruction  
**Repository**: [github.com/gongzix/NeuroClips](https://github.com/gongzix/NeuroClips)

Framework for fMRI-to-video reconstruction; included for neuro representation work (not genomics).

---

## 🧬 Genomics / Single-Cell Models

### Geneformer

**Model ID**: `MOD-GENEFORMER`  
**Modality**: scRNA-seq (Single-cell transcriptomics)  
**Repository**: [huggingface.co/ctheodoris/Geneformer](https://huggingface.co/ctheodoris/Geneformer)

Transformer model pretrained on 30 million single cell transcriptomes. Learns context-aware gene embeddings for cell type annotation, gene regulatory network inference, and therapeutic target discovery.

---

### Caduceus

**Model ID**: `caduceus`  
**Modality**: DNA sequences  
**Repository**: [Caduceus Repository](https://github.com/caduceus)

Long-range DNA sequence model using efficient attention mechanisms for genomic variant interpretation.

---

### DNABERT-2

**Model ID**: `dnabert2`  
**Modality**: DNA sequences  
**Repository**: [DNABERT-2 Repository](https://github.com/dnabert2)

BERT-based model for DNA sequence understanding, supporting tasks like promoter prediction, splice site detection, and variant effect prediction.

---

### Evo2

**Model ID**: `evo2`  
**Modality**: DNA/RNA sequences  
**Repository**: [Evo2 Repository](https://github.com/evo2)

Evolution-inspired foundation model for sequence analysis and generation.

---

### HyenaDNA

**Model ID**: `hyenadna`  
**Modality**: Long DNA sequences  
**Repository**: [HyenaDNA Repository](https://github.com/hyenadna)

Efficient long-range sequence model using Hyena operators for genomic analysis at scale.

---

### LLM Semantic Bridge

**Model ID**: `llm_semantic_bridge`  
**Modality**: Multi-modal semantic alignment  
**Repository**: Semantic Bridge Development

Model for bridging semantic representations across different medical data modalities.

---

## 📝 Model Configuration Format

Each model is defined in a YAML file with the following structure:

```yaml
model_id: unique_identifier
name: Human-Readable Name
modality: primary_modality
upstream_repo: https://github.com/org/repo
notes: Description of the model architecture and capabilities
arch: Architecture details (optional)
params: Parameter count (optional)
```

## 🎯 Adding Your Model

To add your foundation model to the benchmark:

1. Create a model configuration YAML in `models/`
2. Implement the model interface (see `fmbench/models.py`)
3. Run the benchmark suite(s) relevant to your model's modality
4. Submit results for leaderboard inclusion

See the [contributing guide](../index.md#contributing) for more details.

## 📊 Model Performance

For detailed performance metrics and rankings, see the [Leaderboards](../leaderboards/index.md).








