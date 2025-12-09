# Partial Correlations

## Overview

Partial correlation analysis controls for confounding variables when evaluating relationships between model representations and outcomes. This is essential for neurogenomic studies where age, sex, site, and other covariates can introduce spurious correlations.

## Background

**Partial Correlation** measures the relationship between two variables while controlling for one or more confounding variables.

Given:
- **X**: Feature of interest (e.g., brain embedding dimension)
- **Y**: Outcome (e.g., cognitive score)
- **Z**: Confounders (e.g., age, sex, scanner site)

The partial correlation between X and Y controlling for Z is the correlation between the residuals of:
1. X regressed on Z
2. Y regressed on Z

## Why Partial Correlations Matter

### Common Confounders in Neurogenomics

#### Neuroimaging
- **Age**: Affects brain structure and function
- **Sex**: Systematic differences in brain anatomy
- **Scanner site**: Multi-site studies have acquisition differences
- **Head motion**: Correlates with many clinical variables
- **Total intracranial volume (TIV)**: Affects structural measurements

#### Genomics
- **Batch effects**: Technical variation between sequencing runs
- **Cell composition**: Proportion of cell types in bulk data
- **Sequencing depth**: Coverage differences across samples
- **Population stratification**: Ancestry-related genetic variation

## Protocol

### 1. Basic Partial Correlation

```python
import numpy as np
from scipy.stats import pearsonr
from sklearn.linear_model import LinearRegression

def partial_correlation(X, Y, Z):
    """
    Compute partial correlation between X and Y controlling for Z.
    
    Args:
        X: (n_samples,) or (n_samples, 1)
        Y: (n_samples,) or (n_samples, 1)  
        Z: (n_samples, n_confounds)
    
    Returns:
        r: Partial correlation coefficient
        p: P-value
    """
    # Reshape if needed
    X = np.asarray(X).reshape(-1, 1) if X.ndim == 1 else X
    Y = np.asarray(Y).reshape(-1, 1) if Y.ndim == 1 else Y
    Z = np.asarray(Z).reshape(-1, 1) if Z.ndim == 1 else Z
    
    # Regress out confounds from X
    lr_x = LinearRegression()
    lr_x.fit(Z, X)
    X_resid = X - lr_x.predict(Z)
    
    # Regress out confounds from Y
    lr_y = LinearRegression()
    lr_y.fit(Z, Y)
    Y_resid = Y - lr_y.predict(Z)
    
    # Correlation of residuals
    r, p = pearsonr(X_resid.ravel(), Y_resid.ravel())
    
    return r, p

# Example usage
from sklearn.datasets import make_regression

# Simulated data
n_samples = 200
X, Y = make_regression(n_samples=n_samples, n_features=1, noise=10, random_state=42)
X = X.ravel()

# Confounders (age, sex)
age = np.random.randn(n_samples)
sex = np.random.randint(0, 2, n_samples)
Z = np.column_stack([age, sex])

# Compute partial correlation
r_partial, p_partial = partial_correlation(X, Y, Z)
print(f"Partial correlation: r={r_partial:.3f}, p={p_partial:.4f}")

# Compare to raw correlation (without controlling)
r_raw, p_raw = pearsonr(X, Y)
print(f"Raw correlation: r={r_raw:.3f}, p={p_raw:.4f}")
```

### 2. Multiple Partial Correlations

When testing many features (e.g., all embedding dimensions):

```python
from statsmodels.stats.multitest import multipletests

def partial_correlation_matrix(X, Y, Z):
    """
    Compute partial correlations for multiple features.
    
    Args:
        X: (n_samples, n_features) - features to test
        Y: (n_samples,) - outcome
        Z: (n_samples, n_confounds) - confounders
    
    Returns:
        correlations: (n_features,) - partial correlation coefficients
        p_values: (n_features,) - uncorrected p-values
        p_corrected: (n_features,) - FDR-corrected p-values
    """
    n_features = X.shape[1]
    correlations = np.zeros(n_features)
    p_values = np.zeros(n_features)
    
    for i in range(n_features):
        r, p = partial_correlation(X[:, i], Y, Z)
        correlations[i] = r
        p_values[i] = p
    
    # FDR correction
    _, p_corrected, _, _ = multipletests(p_values, method='fdr_bh')
    
    return correlations, p_values, p_corrected

# Example: Test all embedding dimensions
embeddings = model.encode(brain_data)  # Shape: (n_samples, embedding_dim)
cognitive_score = load_cognitive_scores()  # Shape: (n_samples,)

confounds = np.column_stack([age, sex, site_indicator])

corrs, p_raw, p_fdr = partial_correlation_matrix(
    embeddings, 
    cognitive_score, 
    confounds
)

# Find significant dimensions (FDR < 0.05)
sig_dims = np.where(p_fdr < 0.05)[0]
print(f"Significant dimensions: {sig_dims}")
print(f"Correlations: {corrs[sig_dims]}")
```

### 3. Partial Correlation with Standardization

For better interpretability, standardize all variables:

```python
from sklearn.preprocessing import StandardScaler

def partial_correlation_standardized(X, Y, Z):
    """Partial correlation with standardized variables."""
    scaler = StandardScaler()
    
    X_std = scaler.fit_transform(X.reshape(-1, 1)).ravel()
    Y_std = scaler.fit_transform(Y.reshape(-1, 1)).ravel()
    Z_std = scaler.fit_transform(Z)
    
    return partial_correlation(X_std, Y_std, Z_std)
```

### 4. Using pingouin Library

For more advanced partial correlation analysis:

```python
import pingouin as pg

# Create DataFrame
import pandas as pd
df = pd.DataFrame({
    'brain_feature': X,
    'cognitive_score': Y,
    'age': age,
    'sex': sex
})

# Partial correlation
result = pg.partial_corr(
    data=df,
    x='brain_feature',
    y='cognitive_score',
    covar=['age', 'sex'],
    method='pearson'
)

print(result)
# Output: n, r, CI95%, p-val
```

## Domain-Specific Applications

### Neuroimaging Example: fMRI Connectivity and Behavior

```python
# Load data
connectivity = load_fmri_connectivity()  # (n_subjects, n_connections)
behavior = load_behavior_score()  # (n_subjects,)
age = load_age()  # (n_subjects,)
sex = load_sex()  # (n_subjects,)
site = load_site()  # (n_subjects,)
motion = load_head_motion()  # (n_subjects,)

# Confound matrix
Z = np.column_stack([age, sex, site, motion])

# Test each connection
n_connections = connectivity.shape[1]
results = []

for i in range(n_connections):
    r, p = partial_correlation(connectivity[:, i], behavior, Z)
    results.append({'connection_id': i, 'r': r, 'p': p})

results_df = pd.DataFrame(results)

# FDR correction
_, results_df['p_fdr'], _, _ = multipletests(
    results_df['p'], 
    method='fdr_bh'
)

# Significant connections
sig_connections = results_df[results_df['p_fdr'] < 0.05]
print(f"Found {len(sig_connections)} significant connections")
```

### Genomics Example: Gene Expression and Disease

```python
# Load single-cell data
gene_expression = load_gene_expression()  # (n_cells, n_genes)
disease_score = load_disease_phenotype()  # (n_cells,)

# Confounders
batch = load_batch_id()  # (n_cells,)
sequencing_depth = load_total_counts()  # (n_cells,)
cell_cycle = load_cell_cycle_score()  # (n_cells,)

Z_genomics = np.column_stack([batch, sequencing_depth, cell_cycle])

# Find disease-associated genes (controlling for technical factors)
corrs, p_raw, p_fdr = partial_correlation_matrix(
    gene_expression,
    disease_score,
    Z_genomics
)

# Top disease-associated genes
top_genes_idx = np.argsort(np.abs(corrs))[-20:]
print(f"Top disease-associated genes: {gene_names[top_genes_idx]}")
```

## Visualization

### 1. Comparison Plot: Raw vs Partial Correlations

```python
import matplotlib.pyplot as plt

# Compute both raw and partial correlations
raw_corrs = []
partial_corrs = []

for i in range(X.shape[1]):
    r_raw, _ = pearsonr(X[:, i], Y)
    r_partial, _ = partial_correlation(X[:, i], Y, Z)
    raw_corrs.append(r_raw)
    partial_corrs.append(r_partial)

raw_corrs = np.array(raw_corrs)
partial_corrs = np.array(partial_corrs)

# Scatter plot
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(raw_corrs, partial_corrs, alpha=0.6)
ax.plot([-1, 1], [-1, 1], 'k--', label='Identity')
ax.set_xlabel('Raw Correlation')
ax.set_ylabel('Partial Correlation\n(controlling for confounds)')
ax.set_title('Effect of Confound Correction')
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('raw_vs_partial_correlation.png', dpi=150)
```

### 2. Manhattan Plot for Genome-Wide Analysis

```python
def manhattan_plot(p_values, chromosome_positions=None):
    """Create Manhattan plot for partial correlation p-values."""
    fig, ax = plt.subplots(figsize=(14, 4))
    
    # -log10(p-value) for y-axis
    neg_log_p = -np.log10(p_values)
    
    # Color by chromosome (if provided)
    if chromosome_positions is not None:
        colors = ['blue', 'orange']
        for chrom in np.unique(chromosome_positions):
            mask = chromosome_positions == chrom
            color = colors[chrom % 2]
            ax.scatter(np.where(mask)[0], neg_log_p[mask], 
                      c=color, s=5, alpha=0.7)
    else:
        ax.scatter(range(len(p_values)), neg_log_p, s=5, alpha=0.7)
    
    # Significance threshold line
    ax.axhline(-np.log10(0.05), color='red', linestyle='--', 
               label='p=0.05')
    ax.axhline(-np.log10(0.05 / len(p_values)), color='green', 
               linestyle='--', label='Bonferroni')
    
    ax.set_xlabel('Feature Index')
    ax.set_ylabel('-log₁₀(p-value)')
    ax.set_title('Partial Correlation Significance')
    ax.legend()
    
    plt.tight_layout()
    return fig
```

## Interpretation Guidelines

### When to Use Partial Correlations

✅ **Use when:**
- Known confounders exist (age, sex, site)
- Multi-site studies
- Evaluating specific hypotheses while controlling for nuisance variables

❌ **Don't use when:**
- No clear confounders
- Confounders are part of the scientific question
- Confounders are colliders (can introduce bias)

### Effect Size Interpretation

| |r| | Interpretation |
|------|----------------|
| < 0.1 | Negligible |
| 0.1 - 0.3 | Small |
| 0.3 - 0.5 | Moderate |
| > 0.5 | Large |

### Statistical Power

Required sample size for 80% power:

| Effect Size (r) | n (α=0.05) |
|-----------------|------------|
| 0.1 | ~780 |
| 0.2 | ~195 |
| 0.3 | ~85 |
| 0.5 | ~30 |

## ITU AI4H Alignment

This protocol aligns with:

- **DEL3 Section 5.3**: Confound control in validation
- **DEL10.8 Section 4.2**: Covariate adjustment in neurology benchmarks
- **DEL0.1**: Statistical terminology standards

## Best Practices

1. **Pre-register confounders**: Decide which confounders to control before analysis
2. **Report both raw and partial**: Show effect of confound correction
3. **Visualize confound effects**: Plot raw vs. partial correlations
4. **Use appropriate corrections**: FDR or Bonferroni for multiple comparisons
5. **Check assumptions**: Linearity, homoscedasticity, normality

## References

1. Fisher, R. A. (1924). The distribution of the partial correlation coefficient. *Metron*, 3, 329-332.
2. Baba, K., et al. (2004). Partial correlation and conditional correlation as measures of conditional independence. *Australian & New Zealand Journal of Statistics*, 46(4), 657-664.
3. Vallat, R. (2018). Pingouin: statistics in Python. *JOSS*, 3(31), 1026.

## Related Protocols

- [CCA & Permutation Testing](cca_permutation.md)
- [Prediction Baselines](prediction_baselines.md)



