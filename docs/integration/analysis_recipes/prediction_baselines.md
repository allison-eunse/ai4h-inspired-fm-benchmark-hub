# Prediction Baselines

## Overview

Establishing strong baselines is critical for fair evaluation of foundation models. This protocol defines standardized baseline approaches for common prediction tasks in neurogenomics, aligned with ITU AI4H standards.

## Baseline Types

### 1. Random Baseline
**When to use**: Always. Provides lower bound for model performance.

```python
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score

# For binary classification
y_true = np.array([0, 1, 0, 1, ...])  # True labels
y_pred_random = np.random.randint(0, 2, size=len(y_true))

acc_random = accuracy_score(y_true, y_pred_random)
print(f"Random baseline accuracy: {acc_random:.3f}")  # ~0.500
```

### 2. Majority Class Baseline
**When to use**: Classification tasks with class imbalance.

```python
from sklearn.dummy import DummyClassifier

# Majority class classifier
majority_clf = DummyClassifier(strategy='most_frequent')
majority_clf.fit(X_train, y_train)
y_pred = majority_clf.predict(X_test)

acc_majority = accuracy_score(y_test, y_pred)
print(f"Majority class baseline: {acc_majority:.3f}")
```

### 3. Logistic Regression (Linear)
**When to use**: Standard baseline for classification tasks.

```python
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train logistic regression
lr = LogisticRegression(max_iter=1000, random_state=42)
lr.fit(X_train_scaled, y_train)

# Evaluate
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]

acc_lr = accuracy_score(y_test, y_pred)
auc_lr = roc_auc_score(y_test, y_prob)

print(f"Logistic Regression - Accuracy: {acc_lr:.3f}, AUC: {auc_lr:.3f}")
```

### 4. Ridge Regression (Linear, Regularized)
**When to use**: High-dimensional data with multicollinearity (e.g., fMRI, genomics).

```python
from sklearn.linear_model import Ridge, RidgeCV

# Cross-validated ridge
alphas = np.logspace(-3, 3, 50)
ridge_cv = RidgeCV(alphas=alphas, cv=5)
ridge_cv.fit(X_train_scaled, y_train)

y_pred = ridge_cv.predict(X_test_scaled)

from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Ridge Regression - MSE: {mse:.3f}, R²: {r2:.3f}")
print(f"Optimal alpha: {ridge_cv.alpha_:.3f}")
```

### 5. Random Forest (Non-linear)
**When to use**: Benchmark for capturing non-linear relationships.

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,
    min_samples_split=10,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]

acc_rf = accuracy_score(y_test, y_pred)
auc_rf = roc_auc_score(y_test, y_prob)

print(f"Random Forest - Accuracy: {acc_rf:.3f}, AUC: {auc_rf:.3f}")
```

### 6. k-Nearest Neighbors (Instance-based)
**When to use**: Baseline for representation quality (in embedding space).

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

y_pred = knn.predict(X_test_scaled)
acc_knn = accuracy_score(y_test, y_pred)

print(f"k-NN baseline: {acc_knn:.3f}")
```

## Domain-Specific Baselines

### Neuroimaging (fMRI/sMRI)

#### Regional Mean Baseline
Use mean signal per brain region as features:

```python
# Assuming parcellated brain data
# X_raw: (n_samples, n_voxels)
# parcellation: (n_voxels,) - region labels

n_regions = len(np.unique(parcellation))
X_regional = np.zeros((len(X_raw), n_regions))

for i, region_id in enumerate(np.unique(parcellation)):
    mask = parcellation == region_id
    X_regional[:, i] = X_raw[:, mask].mean(axis=1)

# Use X_regional for prediction
```

#### Functional Connectivity Baseline
Use pairwise correlations as features:

```python
from nilearn.connectome import ConnectivityMeasure

conn_measure = ConnectivityMeasure(kind='correlation')
connectivity_matrices = conn_measure.fit_transform(timeseries_data)

# Flatten upper triangle as features
n_samples = len(connectivity_matrices)
n_regions = connectivity_matrices.shape[1]
n_features = n_regions * (n_regions - 1) // 2

X_conn = np.zeros((n_samples, n_features))
for i, conn_mat in enumerate(connectivity_matrices):
    X_conn[i] = conn_mat[np.triu_indices(n_regions, k=1)]
```

### Genomics (scRNA-seq)

#### Highly Variable Genes (HVG) Baseline
Use only the most variable genes:

```python
from sklearn.feature_selection import VarianceThreshold

# Select top 2000 most variable genes
gene_vars = X_train.var(axis=0)
top_genes_idx = np.argsort(gene_vars)[-2000:]

X_train_hvg = X_train[:, top_genes_idx]
X_test_hvg = X_test[:, top_genes_idx]
```

#### Cell-Type Marker Genes
Use known marker genes for classification:

```python
# Pre-defined marker gene list
marker_genes = ['CD3D', 'CD79A', 'CST3', 'NKG7', ...]  # Example
marker_idx = [gene_names.index(g) for g in marker_genes if g in gene_names]

X_train_markers = X_train[:, marker_idx]
X_test_markers = X_test[:, marker_idx]
```

## Baseline Evaluation Protocol

### 1. Define Metrics

```python
from sklearn.metrics import (
    accuracy_score, 
    balanced_accuracy_score,
    f1_score, 
    roc_auc_score,
    precision_recall_fscore_support
)

def evaluate_classifier(y_true, y_pred, y_prob=None):
    """Comprehensive classification metrics."""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_prob is not None:
        metrics['auroc'] = roc_auc_score(y_true, y_prob)
    
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average='weighted'
    )
    metrics.update({
        'precision': precision,
        'recall': recall,
    })
    
    return metrics
```

### 2. Cross-Validation

```python
from sklearn.model_selection import cross_validate

# 5-fold cross-validation
cv_results = cross_validate(
    estimator=lr,
    X=X_train_scaled,
    y=y_train,
    cv=5,
    scoring=['accuracy', 'roc_auc', 'f1'],
    return_train_score=True,
    n_jobs=-1
)

print(f"CV Accuracy: {cv_results['test_accuracy'].mean():.3f} "
      f"± {cv_results['test_accuracy'].std():.3f}")
print(f"CV AUC: {cv_results['test_roc_auc'].mean():.3f} "
      f"± {cv_results['test_roc_auc'].std():.3f}")
```

### 3. Baseline Comparison Table

```python
import pandas as pd

baselines = {
    'Random': (acc_random, None),
    'Majority Class': (acc_majority, None),
    'Logistic Regression': (acc_lr, auc_lr),
    'Ridge': (acc_ridge, auc_ridge),
    'Random Forest': (acc_rf, auc_rf),
    'k-NN': (acc_knn, auc_knn),
}

results_df = pd.DataFrame([
    {'Method': name, 'Accuracy': acc, 'AUC': auc}
    for name, (acc, auc) in baselines.items()
])

print(results_df.to_markdown(index=False, floatfmt='.3f'))
```

## Expected Baseline Performance

### Neuroimaging Classification (e.g., AD vs CN)

| Baseline | Typical Accuracy | Typical AUC |
|----------|------------------|-------------|
| Random | 0.50 | 0.50 |
| Majority | 0.50-0.60 | - |
| Logistic Regression | 0.65-0.75 | 0.70-0.80 |
| Ridge | 0.65-0.75 | 0.70-0.80 |
| Random Forest | 0.70-0.80 | 0.75-0.85 |

### Single-Cell Classification (Cell Type)

| Baseline | Typical Accuracy | Typical F1 |
|----------|------------------|------------|
| Random | 0.10-0.20 | 0.10-0.20 |
| Marker Genes + Logistic | 0.75-0.85 | 0.70-0.80 |
| HVG + Random Forest | 0.80-0.90 | 0.75-0.85 |

## ITU AI4H Alignment

This protocol aligns with:

- **DEL3 Section 6.3**: Performance benchmarking requirements
- **DEL3 Section 7.1**: Baseline comparison standards
- **DEL0.1**: Terminology for baseline models ("reference implementation")

## Best Practices

1. **Always report random baseline** to establish floor performance
2. **Use cross-validation** for robust baseline estimates
3. **Match preprocessing** between baselines and foundation models
4. **Report confidence intervals** (e.g., via bootstrapping)
5. **Document hyperparameters** for reproducibility

## Related Protocols

- [CCA & Permutation Testing](cca_permutation.md)
- [Partial Correlations](partial_correlations.md)




