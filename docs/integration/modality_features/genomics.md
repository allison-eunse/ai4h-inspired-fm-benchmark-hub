# Genomics Data Specifications

## Overview

This page defines standardized data formats and preprocessing requirements for genomics data in the benchmark hub, covering single-cell RNA-seq (scRNA-seq), bulk RNA-seq, DNA sequences, and variant data.

## Data Modalities

### 1. Single-Cell RNA-seq (scRNA-seq)

**Format**: AnnData (`.h5ad`), Loom (`.loom`), or CSV/TSV matrices  
**Shape**: `(n_cells, n_genes)`  
**Data Type**: Raw counts (integer) or normalized (float)

```python
import scanpy as sc
import numpy as np

# Load single-cell data
adata = sc.read_h5ad('pbmc_68k.h5ad')

# Inspect
print(f"Shape: {adata.shape}")  # (n_cells, n_genes)
print(f"Genes: {adata.var_names[:5]}")
print(f"Cells: {adata.obs_names[:5]}")

# Access count matrix
X = adata.X  # Sparse or dense matrix (n_cells, n_genes)
```

#### Required Metadata (adata.obs)

```python
# Cell-level metadata
adata.obs['cell_type']  # Cell type annotations
adata.obs['donor_id']  # Subject/donor identifier
adata.obs['batch']  # Batch/experiment ID
adata.obs['n_genes']  # Number of detected genes
adata.obs['n_counts']  # Total UMI counts
adata.obs['percent_mito']  # % mitochondrial gene expression
```

#### Gene Metadata (adata.var)

```python
# Gene-level metadata
adata.var['gene_ids']  # Ensembl IDs
adata.var['gene_symbols']  # HGNC symbols
adata.var['highly_variable']  # Boolean for HVG selection
adata.var['mean_counts']  # Mean expression
adata.var['dispersions']  # Dispersion (variability)
```

### 2. Bulk RNA-seq

**Format**: NumPy array, CSV, or TSV  
**Shape**: `(n_samples, n_genes)`  
**Data Type**: Raw counts, TPM, or FPKM

```python
import pandas as pd

# Load bulk RNA-seq data
gene_expression = pd.read_csv('bulk_rnaseq_tpm.csv', index_col=0)
# Rows: genes, Columns: samples

# Transpose to (samples, genes) for ML
X = gene_expression.T.values  # (n_samples, n_genes)
```

### 3. DNA Sequences

**Format**: FASTA (`.fasta`, `.fa`), plain text, or tokenized arrays  
**Alphabet**: `{A, C, G, T, N}` (N = ambiguous)

```python
from Bio import SeqIO

# Load FASTA file
sequences = []
for record in SeqIO.parse("sequences.fasta", "fasta"):
    sequences.append(str(record.seq))

# Tokenize sequences (for transformer models)
vocab = {'A': 0, 'C': 1, 'G': 2, 'T': 3, 'N': 4, '<PAD>': 5}

def tokenize_sequence(seq, vocab, max_len=512):
    """Convert DNA sequence to token IDs."""
    tokens = [vocab.get(base, vocab['N']) for base in seq.upper()]
    
    # Pad or truncate
    if len(tokens) < max_len:
        tokens += [vocab['<PAD>']] * (max_len - len(tokens))
    else:
        tokens = tokens[:max_len]
    
    return np.array(tokens)

tokenized = [tokenize_sequence(seq, vocab) for seq in sequences]
X_dna = np.array(tokenized)  # (n_sequences, max_len)
```

### 4. Variant Data (VCF)

**Format**: VCF (Variant Call Format)  
**Use**: SNP arrays, whole-genome sequencing (WGS), whole-exome sequencing (WES)

```python
import pandas as pd
import allel

# Load VCF file
vcf_path = 'cohort_variants.vcf.gz'
callset = allel.read_vcf(vcf_path)

# Extract genotypes
genotypes = callset['calldata/GT']  # (n_variants, n_samples, ploidy=2)

# Convert to dosage (0, 1, 2 for diploid)
dosage = genotypes.sum(axis=-1)  # (n_variants, n_samples)

# Transpose for ML: (samples, variants)
X_geno = dosage.T
```

## Preprocessing Workflows

### scRNA-seq Standard Preprocessing

```python
import scanpy as sc

# Load raw counts
adata = sc.read_h5ad('raw_counts.h5ad')

# 1. Quality control
sc.pp.filter_cells(adata, min_genes=200)  # Remove low-quality cells
sc.pp.filter_genes(adata, min_cells=3)  # Remove rare genes

# Calculate QC metrics
adata.var['mt'] = adata.var_names.str.startswith('MT-')
sc.pp.calculate_qc_metrics(
    adata, 
    qc_vars=['mt'], 
    percent_top=None, 
    log1p=False, 
    inplace=True
)

# Filter cells by QC
adata = adata[adata.obs.n_genes_by_counts < 2500, :]
adata = adata[adata.obs.pct_counts_mt < 5, :]

# 2. Normalization
sc.pp.normalize_total(adata, target_sum=1e4)  # CPM-like
sc.pp.log1p(adata)  # Log-transform

# 3. Feature selection
sc.pp.highly_variable_genes(adata, n_top_genes=2000)
adata = adata[:, adata.var.highly_variable]  # Keep only HVGs

# 4. Scaling (for PCA/visualization, not always for ML)
sc.pp.scale(adata, max_value=10)

# Extract preprocessed data
X_processed = adata.X  # (n_cells, 2000)
```

### Bulk RNA-seq Preprocessing

```python
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load TPM or FPKM data
X = gene_expression.values  # (n_samples, n_genes)

# 1. Log-transform (if not already)
X_log = np.log1p(X)

# 2. Filter low-expression genes
gene_means = X_log.mean(axis=0)
high_expr_genes = gene_means > 1.0  # Threshold
X_filtered = X_log[:, high_expr_genes]

# 3. Z-score normalization (optional)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_filtered)
```

### DNA Sequence Preprocessing

```python
# K-mer encoding (alternative to tokenization)
def generate_kmers(sequence, k=3):
    """Generate k-mer features from DNA sequence."""
    kmers = []
    for i in range(len(sequence) - k + 1):
        kmers.append(sequence[i:i+k])
    return kmers

# Build k-mer vocabulary
from collections import Counter

all_kmers = []
for seq in sequences:
    all_kmers.extend(generate_kmers(seq, k=3))

kmer_vocab = {kmer: i for i, kmer in enumerate(set(all_kmers))}

# Encode sequence as k-mer counts
def encode_kmers(sequence, kmer_vocab, k=3):
    """Encode sequence as k-mer count vector."""
    kmers = generate_kmers(sequence, k)
    counts = Counter(kmers)
    
    vec = np.zeros(len(kmer_vocab))
    for kmer, count in counts.items():
        if kmer in kmer_vocab:
            vec[kmer_vocab[kmer]] = count
    
    return vec
```

## Data Augmentation for scRNA-seq

### 1. Poisson Sampling (Simulates sequencing depth variation)

```python
def poisson_augment(counts, factor=0.8):
    """
    Simulate different sequencing depths via Poisson sampling.
    
    Args:
        counts: (n_cells, n_genes) - raw count matrix
        factor: Downsampling factor (0 < factor < 1)
    
    Returns:
        augmented: Downsampled count matrix
    """
    total_counts = counts.sum(axis=1, keepdims=True)
    target_counts = total_counts * factor
    
    # Multinomial sampling
    probs = counts / (total_counts + 1e-8)
    augmented = np.array([
        np.random.multinomial(int(tc), p)
        for tc, p in zip(target_counts, probs)
    ])
    
    return augmented
```

### 2. Cell Mixtures (Simulates doublets)

```python
def create_doublet(cell1, cell2):
    """Simulate doublet by averaging two cells."""
    return (cell1 + cell2) / 2
```

## Benchmark Tasks

### 1. Cell Type Annotation (scRNA-seq)

**Input**: Gene expression matrix `(n_cells, n_genes)`  
**Output**: Cell type labels `(n_cells,)`

```python
# Example: PBMC cell type classification
X_train = adata_train.X  # (n_train, n_genes)
y_train = adata_train.obs['cell_type'].values  # (n_train,)

# Labels: B cells, T cells, Monocytes, NK cells, etc.
```

### 2. Gene Expression Prediction

**Input**: Partial gene expression or other modalities  
**Output**: Full gene expression profiles

### 3. Variant Effect Prediction

**Input**: DNA sequences with variants  
**Output**: Pathogenicity scores or functional effects

### 4. Sequence Classification

**Input**: DNA sequences  
**Output**: Labels (e.g., promoter vs non-promoter, coding vs non-coding)

## Quality Control Metrics

### scRNA-seq QC

```python
# Check key QC metrics
print(f"Mean genes/cell: {adata.obs['n_genes'].mean():.0f}")
print(f"Mean UMIs/cell: {adata.obs['n_counts'].mean():.0f}")
print(f"Mean % mito: {adata.obs['percent_mito'].mean():.2f}%")

# Recommended thresholds:
# - n_genes: 200 - 5000
# - percent_mito: < 5-10%
# - n_counts: > 500
```

### Variant Data QC

```python
# Check missing rate
missing_rate = (genotypes == -1).mean()
print(f"Missing genotype rate: {missing_rate*100:.2f}%")

# Filter variants by minor allele frequency (MAF)
from allel import GenotypeArray

gt = GenotypeArray(genotypes)
allele_counts = gt.count_alleles()
maf = allele_counts[:, 1] / allele_counts.sum(axis=1)

# Keep MAF > 0.01
keep_variants = maf > 0.01
```

## Data Formats

### AnnData Structure

```python
# Recommended AnnData structure for benchmarking
adata.X  # Main data matrix (can be sparse)
adata.obs  # Cell/sample metadata (pandas DataFrame)
adata.var  # Gene/feature metadata (pandas DataFrame)
adata.layers  # Alternative representations (raw, scaled, etc.)
adata.obsm  # Multi-dimensional annotations (PCA, UMAP)
adata.varm  # Multi-dimensional gene annotations
adata.uns  # Unstructured metadata (dataset info)
```

### Example Dataset Metadata

```yaml
dataset_id: pbmc_68k
name: PBMC 68k (10x Genomics)
modality: scRNA-seq
n_cells: 68579
n_genes: 20387
technology: 10x Chromium v2
organism: Homo sapiens
tissue: Peripheral blood mononuclear cells (PBMC)
preprocessing: Scanpy 1.9.1
normalization: log1p(CPM)
feature_selection: top_2000_HVG
reference: Zheng et al. (2017)
```

## Foundation Model Input Formats

### For Geneformer and Similar Models

```python
# Geneformer expects rank-ordered gene tokens
def prepare_geneformer_input(adata, n_genes=2048):
    """
    Convert AnnData to Geneformer input format.
    
    Returns ranked gene indices per cell.
    """
    # Rank genes by expression within each cell
    cell_inputs = []
    
    for i in range(adata.n_obs):
        cell_expr = adata.X[i].toarray().ravel()
        
        # Rank genes (descending)
        ranked_genes = np.argsort(cell_expr)[::-1][:n_genes]
        
        cell_inputs.append(ranked_genes)
    
    return np.array(cell_inputs)  # (n_cells, n_genes)
```

### For DNA Sequence Models (DNABERT, HyenaDNA, etc.)

```python
# K-mer tokenization for DNABERT
def kmer_tokenize(sequence, k=6, stride=1):
    """
    Tokenize DNA sequence into k-mers.
    
    Args:
        sequence: DNA string
        k: K-mer size
        stride: Step size
    
    Returns:
        tokens: List of k-mer strings
    """
    tokens = []
    for i in range(0, len(sequence) - k + 1, stride):
        tokens.append(sequence[i:i+k])
    
    return tokens

# Example
seq = "ATCGATCGATCG"
tokens = kmer_tokenize(seq, k=6, stride=3)
print(tokens)  # ['ATCGAT', 'GATCGA', 'TCGATC']
```

## ITU AI4H Alignment

This specification aligns with:

- **DEL3 Section 4.2**: Data format requirements for AI4H benchmarks
- **DEL0.1**: Genomics terminology standardization
- **FAIR principles**: Findable, Accessible, Interoperable, Reusable data

## Tools & Libraries

### scRNA-seq
- **Scanpy**: Single-cell analysis in Python
- **Seurat**: R toolkit for scRNA-seq
- **scVI-tools**: Probabilistic models for single-cell omics

### Genomics
- **BioPython**: Sequence analysis
- **PyVCF / scikit-allel**: Variant data handling
- **pybedtools**: Genomic interval operations

### Foundation Model Libraries
- **Geneformer**: Transformer for scRNA-seq
- **DNABERT**: BERT for DNA sequences
- **Nucleotide Transformer**: Multi-species genomic foundation model

## References

1. Wolf, F. A., et al. (2018). SCANPY: large-scale single-cell gene expression data analysis. *Genome Biology*, 19(1), 15.
2. Theodoris, C. V., et al. (2023). Transfer learning enables predictions in network biology. *Nature*, 618, 616-624. (Geneformer)
3. Ji, Y., et al. (2021). DNABERT: pre-trained Bidirectional Encoder Representations from Transformers model for DNA-language in genome. *Bioinformatics*, 37(15), 2112-2120.

## Related Documentation

- [fMRI Specifications](fmri.md)
- [sMRI Specifications](smri.md)
- [Prediction Baselines](../analysis_recipes/prediction_baselines.md)




