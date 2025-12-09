# CCA & Permutation Testing

## Overview

Canonical Correlation Analysis (CCA) with permutation testing is a standard protocol for evaluating the relationship between multimodal brain data representations. This recipe provides a reproducible approach for testing whether learned representations capture meaningful biological signal beyond chance.

## Background

**Canonical Correlation Analysis (CCA)** identifies linear combinations of two sets of variables that maximize their correlation. In the context of foundation models:

- **Use Case 1**: Compare model embeddings from different modalities (e.g., fMRI vs genetics)
- **Use Case 2**: Test if model representations align with behavioral or clinical outcomes
- **Use Case 3**: Validate cross-modal alignment in multi-modal foundation models

**Permutation Testing** provides statistical significance by:
1. Computing CCA on real data
2. Shuffling one modality randomly N times
3. Computing CCA on each permutation
4. Comparing real correlation against null distribution

## Protocol

### 1. Data Preparation

```python
import numpy as np
from sklearn.cross_decomposition import CCA

# Load your model embeddings
# X: (n_samples, n_features_1) - e.g., fMRI embeddings
# Y: (n_samples, n_features_2) - e.g., genomic embeddings

X = model_fmri.encode(fmri_data)  # Shape: (N, D1)
Y = model_genomics.encode(genomic_data)  # Shape: (N, D2)

# Standardize
X = (X - X.mean(axis=0)) / X.std(axis=0)
Y = (Y - Y.mean(axis=0)) / Y.std(axis=0)
```

### 2. Run CCA

```python
# Number of canonical components to compute
n_components = min(X.shape[1], Y.shape[1], 20)

cca = CCA(n_components=n_components)
X_c, Y_c = cca.fit_transform(X, Y)

# Compute canonical correlations
canonical_corrs = [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] 
                   for i in range(n_components)]

print(f"Top 5 canonical correlations: {canonical_corrs[:5]}")
```

### 3. Permutation Test

```python
from tqdm import tqdm

n_permutations = 1000
perm_corrs = []

for _ in tqdm(range(n_permutations)):
    # Shuffle one modality
    idx = np.random.permutation(len(Y))
    Y_perm = Y[idx]
    
    # Run CCA on permuted data
    cca_perm = CCA(n_components=n_components)
    X_c_perm, Y_c_perm = cca_perm.fit_transform(X, Y_perm)
    
    # Store top canonical correlation
    perm_corr = np.corrcoef(X_c_perm[:, 0], Y_c_perm[:, 0])[0, 1]
    perm_corrs.append(perm_corr)

perm_corrs = np.array(perm_corrs)
```

### 4. Compute P-Value

```python
# P-value: fraction of permutations with correlation >= observed
real_corr = canonical_corrs[0]
p_value = (perm_corrs >= real_corr).mean()

print(f"Real CCA correlation: {real_corr:.4f}")
print(f"Mean permuted correlation: {perm_corrs.mean():.4f}")
print(f"P-value: {p_value:.4f}")

# Effect size
effect_size = (real_corr - perm_corrs.mean()) / perm_corrs.std()
print(f"Effect size (Z-score): {effect_size:.2f}")
```

### 5. Visualization

```python
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

# Histogram of permutation null distribution
axes[0].hist(perm_corrs, bins=50, alpha=0.7, label='Permuted')
axes[0].axvline(real_corr, color='red', linestyle='--', 
                label=f'Observed (p={p_value:.3f})')
axes[0].set_xlabel('Canonical Correlation')
axes[0].set_ylabel('Frequency')
axes[0].set_title('CCA Permutation Test')
axes[0].legend()

# Scree plot of canonical correlations
axes[1].plot(range(1, len(canonical_corrs)+1), canonical_corrs, 
             marker='o', label='Real Data')
axes[1].axhline(perm_corrs.mean(), color='gray', linestyle='--', 
                label='Permuted Mean')
axes[1].set_xlabel('Canonical Component')
axes[1].set_ylabel('Correlation')
axes[1].set_title('Canonical Correlations (Scree Plot)')
axes[1].legend()

plt.tight_layout()
plt.savefig('cca_permutation_results.png', dpi=150)
```

## Interpretation Guidelines

### Statistical Significance
- **p < 0.05**: Significant alignment between modalities
- **p < 0.01**: Strong evidence of cross-modal relationship
- **p < 0.001**: Very strong evidence

### Effect Size
- **Z > 2**: Small to moderate effect
- **Z > 3**: Moderate to large effect  
- **Z > 5**: Large effect

### Multiple Comparisons
When testing multiple canonical components, apply correction:

```python
from statsmodels.stats.multitest import multipletests

# Compute p-value for each component
p_values = [(perm_corrs >= corr).mean() for corr in canonical_corrs]

# Bonferroni correction
_, p_corrected, _, _ = multipletests(p_values, method='bonferroni')
```

## Example Use Cases

### Use Case 1: Multi-Modal Foundation Model Validation

Test if a multi-modal brain model's fMRI and genetics branches learn aligned representations:

```python
# Extract embeddings from each branch
fmri_emb = model.encode_fmri(fmri_data)
gene_emb = model.encode_genetics(genetics_data)

# Run CCA + permutation test
# → If p < 0.05, model successfully learns cross-modal alignment
```

### Use Case 2: Behavioral Prediction Alignment

Test if brain embeddings correlate with behavioral scores:

```python
# Brain embeddings (N x D)
brain_emb = model.encode(fmri_data)

# Behavioral scores (N x K)
behavior_scores = load_behavioral_data()

# Run CCA
# → High correlation suggests embeddings capture behaviorally relevant features
```

## ITU AI4H Alignment

This protocol aligns with:

- **DEL3 Section 5.2**: Validation and verification requirements
- **DEL10.8**: Neurology-specific evaluation protocols
- **DEL0.1**: Statistical significance standards for AI4H benchmarks

## References

1. Hotelling, H. (1936). Relations between two sets of variates. *Biometrika*, 28(3/4), 321-377.
2. Witten, D. M., Tibshirani, R., & Hastie, T. (2009). A penalized matrix decomposition. *Biostatistics*, 10(3), 515-534.
3. Wang, C. et al. (2020). Variational CCA for multimodal representation learning. *ICML*.

## Related Protocols

- [Prediction Baselines](prediction_baselines.md)
- [Partial Correlations](partial_correlations.md)



