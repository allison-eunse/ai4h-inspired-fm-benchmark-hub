# Free Data Sources for FM Benchmarking

This page lists the **real, publicly available datasets** included in this benchmark hub, with proper attribution to their original sources.

---

## 🧬 DNA Sequence Data (Genomic Benchmarks)

These datasets are from the **Genomic Benchmarks** collection and are ready to use in `toy_data/genomics/dna_sequences/`.

!!! info "Data Source"
    DNA sequence benchmarks are from [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks) by Grešová et al.

### Enhancer Classification (Cohn et al.)

**Real ChIP-seq derived enhancer sequences**

| Split | Sequences | Source |
|-------|-----------|--------|
| Train | 20,843 | ChIP-seq experiments |
| Test | 6,948 | ChIP-seq experiments |

```
toy_data/genomics/dna_sequences/enhancers_cohn/
├── train.tsv  (sequence, label)
└── test.tsv   (sequence, label)
```

**Run benchmark:**

```bash
python -m fmbench run --suite SUITE-DNA-CLASS --model configs/model_evo2.yaml --out results/
```

---

### Promoter Classification (EPD Database)

**Real curated promoters from Eukaryotic Promoter Database**

| Split | Sequences | Source |
|-------|-----------|--------|
| Train | 27,097 | EPD database |
| Test | 9,034 | EPD database |

```
toy_data/genomics/dna_sequences/promoters_nontata/
├── train.tsv
└── test.tsv
```

---

### Regulatory Elements (Ensembl)

**Real annotations from ENCODE + Roadmap Epigenomics**

| Split | Sequences | Classes | Notes |
|-------|-----------|---------|-------|
| Train | 1,500 | enhancer, promoter, open_chromatin | Toy version (500/class) |
| Test | 57,713 | enhancer, promoter, open_chromatin | Full test set |

!!! note "Full Dataset"
    The toy train set is a stratified subsample. Full training set (231,348 samples) available from [Genomic Benchmarks](https://huggingface.co/datasets/katielink/genomic-benchmarks).

```
toy_data/genomics/dna_sequences/regulatory_ensembl/
├── train.tsv
└── test.tsv
```

---

### Nucleotide Transformer Benchmark Suite

**The standard multi-task benchmark for DNA FMs**

| Split | Sequences | Tasks | Notes |
|-------|-----------|-------|-------|
| Train | 1,500 | 18 tasks | Toy version (500/class) |
| Test | 38,822 | 18 tasks | Full test set |

!!! note "Full Dataset"
    The toy train set is a stratified subsample. Full training set (493,242 samples) available from [InstaDeepAI](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised).

**Tasks include:**

- Enhancer prediction
- Promoter prediction (TATA / non-TATA)
- Splice site prediction (donor / acceptor)
- Histone modifications (H3, H3K4me3, H3K36me3, etc.)

```
toy_data/genomics/dna_sequences/nucleotide_transformer/
├── train.tsv  (sequence, name, label, task)
└── test.tsv
```

!!! info "Data Source"
    From [InstaDeepAI/nucleotide_transformer_downstream_tasks_revised](https://huggingface.co/datasets/InstaDeepAI/nucleotide_transformer_downstream_tasks_revised). See [Dalla-Torre et al. (2023)](https://www.biorxiv.org/content/10.1101/2023.01.11.523679).

---

## What makes this data "real"?

| Dataset | Experimental Source | Not Synthetic Because |
|---------|--------------------|-----------------------|
| Enhancers (Cohn) | ChIP-seq | Binding sites measured in cells |
| Promoters (EPD) | TSS mapping | Curated from experimental evidence |
| Regulatory (Ensembl) | ENCODE + Roadmap | 1000s of ChIP-seq/DNase-seq experiments |
| NT Benchmark | Multiple | Industry-standard benchmark from papers |

---

## 🧬 Single-Cell RNA-seq Data

### PBMC 3K Dataset

**Pre-processed with real cell type labels**

| Metric | Value |
|--------|-------|
| Cells | 2,638 |
| Genes | 1,838 |
| Classes | 8 cell types |

```
toy_data/genomics/pbmc_classification/
├── X.npy      (expression matrix)
├── y.npy      (labels)
├── metadata.csv
└── label_map.csv
```

!!! info "Data Source"
    PBMC data from [10x Genomics](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0) - "3k PBMCs from a Healthy Donor" dataset. Processed using [Scanpy](https://scanpy.readthedocs.io/) with cell type annotations from Louvain clustering.

**Run benchmark:**

```bash
python -m fmbench run --suite SUITE-GEN-CLASS-001 --model configs/model_geneformer.yaml --dataset DS-PBMC --out results/
```

---

## Additional Data Sources

### More DNA Sequence Data

| Source | URL | Data Type |
|--------|-----|-----------|
| **UCSC Genome Browser** | [hgdownload.soe.ucsc.edu](https://hgdownload.soe.ucsc.edu) | Reference genomes |
| **ENCODE Portal** | [encodeproject.org](https://www.encodeproject.org) | ChIP-seq, ATAC-seq |
| **Roadmap Epigenomics** | [roadmapepigenomics.org](http://www.roadmapepigenomics.org) | Epigenomic marks |
| **ClinVar** | [ncbi.nlm.nih.gov/clinvar](https://www.ncbi.nlm.nih.gov/clinvar/) | Pathogenic variants |

### More Single-Cell Data

| Source | URL | Data Type |
|--------|-----|-----------|
| **10x Genomics** | [10xgenomics.com/datasets](https://www.10xgenomics.com/resources/datasets) | scRNA-seq |
| **CellxGene** | [cellxgene.cziscience.com](https://cellxgene.cziscience.com) | Annotated atlases |
| **Human Cell Atlas** | [humancellatlas.org](https://www.humancellatlas.org) | Multi-tissue atlases |

### Brain Imaging Data

| Source | URL | Registration |
|--------|-----|--------------|
| **OpenNeuro** | [openneuro.org](https://openneuro.org) | None required |
| **HCP** | [humanconnectome.org](https://www.humanconnectome.org) | Required |
| **UK Biobank** | [ukbiobank.ac.uk](https://www.ukbiobank.ac.uk) | Application required |

---

## Download More Data

```bash
# Activate environment
source scripts/activate_fmbench.sh

# Download additional genomic benchmarks
python << 'EOF'
from datasets import load_dataset

# Drosophila enhancers
ds = load_dataset("katarinagresova/Genomic_Benchmarks_drosophila_enhancers_stark")

# Mouse enhancers  
ds = load_dataset("katarinagresova/Genomic_Benchmarks_dummy_mouse_enhancers_ensembl")

# Coding vs intergenic
ds = load_dataset("katarinagresova/Genomic_Benchmarks_demo_coding_vs_intergenomic_seqs")
EOF
```

---

## 📚 Citations & Acknowledgments

If you use data from this benchmark hub, please cite the original sources:

### Genomic Benchmarks (DNA Sequence Data)

> GREŠOVÁ, Katarína, et al. **Genomic Benchmarks: A Collection of Datasets for Genomic Sequence Classification.** *bioRxiv*, 2022.
> 
> URL: [https://www.biorxiv.org/content/10.1101/2022.06.08.495248](https://www.biorxiv.org/content/10.1101/2022.06.08.495248)

```bibtex
@article{gresova2022genomic,
  title={Genomic Benchmarks: A Collection of Datasets for Genomic Sequence Classification},
  author={Gresova, Katarina and Martinek, Vlastimil and Cechak, David and Simecek, Petr and Alexiou, Panagiotis},
  journal={bioRxiv},
  year={2022},
  publisher={Cold Spring Harbor Laboratory},
  url={https://www.biorxiv.org/content/10.1101/2022.06.08.495248}
}
```

### Nucleotide Transformer (NT Benchmark Data)

> DALLA-TORRE, Hugo, et al. **The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics.** *bioRxiv*, 2023.
> 
> URL: [https://www.biorxiv.org/content/10.1101/2023.01.11.523679](https://www.biorxiv.org/content/10.1101/2023.01.11.523679)

```bibtex
@article{dalla2023nucleotide,
  title={The Nucleotide Transformer: Building and Evaluating Robust Foundation Models for Human Genomics},
  author={Dalla-Torre, Hugo and Gonzalez, Liam and Mendoza-Revilla, Javier and others},
  journal={bioRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory}
}
```

### 10x Genomics PBMC 3K (Single-Cell RNA-seq Data)

> **10x Genomics.** 3k PBMCs from a Healthy Donor. 10x Genomics Datasets, 2016.
> 
> URL: [https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0](https://www.10xgenomics.com/datasets/3-k-pbm-cs-from-a-healthy-donor-1-standard-1-1-0)

### Scanpy (Single-Cell Analysis)

> WOLF, F. Alexander, ANGERER, Philipp, and THEIS, Fabian J. **SCANPY: large-scale single-cell gene expression data analysis.** *Genome Biology*, 2018.

```bibtex
@article{wolf2018scanpy,
  title={SCANPY: large-scale single-cell gene expression data analysis},
  author={Wolf, F Alexander and Angerer, Philipp and Theis, Fabian J},
  journal={Genome biology},
  volume={19},
  number={1},
  pages={1--5},
  year={2018},
  publisher={Springer}
}
```

### ENCODE Consortium (Regulatory Data)

> **The ENCODE Project Consortium.** An integrated encyclopedia of DNA elements in the human genome. *Nature*, 2012.

### Roadmap Epigenomics (Regulatory Data)

> **Roadmap Epigenomics Consortium.** Integrative analysis of 111 reference human epigenomes. *Nature*, 2015.

### Eukaryotic Promoter Database (Promoter Data)

> DREOS, René, et al. **The eukaryotic promoter database in its 30th year: focus on non-vertebrate organisms.** *Nucleic Acids Research*, 2017.

---

## License

- **Genomic Benchmarks**: Apache 2.0
- **10x Genomics PBMC**: Free for research use
- **ENCODE/Roadmap data**: Public domain

---

[Back to Home](index.md){ .md-button }
[View Benchmarks](leaderboards/index.md){ .md-button }

