# sMRI Data Specifications

## Overview

Structural MRI (sMRI) captures anatomical brain structure, typically using T1-weighted or T2-weighted sequences. This page defines standardized data formats and preprocessing requirements for sMRI data in this benchmark hub.

## Data Format Requirements

### 1. Raw Structural MRI

**Format**: NIfTI (`.nii` or `.nii.gz`)  
**Typical Shape**: `(256, 256, 256)` at 1mm³ isotropic resolution  
**Data Type**: `float32` or `int16`  
**Contrast**: T1-weighted (most common), T2-weighted, FLAIR, or multi-contrast

```python
import nibabel as nib
import numpy as np

# Load T1-weighted image
img = nib.load('subject_001_T1w.nii.gz')
data = img.get_fdata()  # Shape: (x, y, z)

print(f"Image shape: {data.shape}")
print(f"Voxel size: {img.header.get_zooms()[:3]} mm")
```

### 2. Preprocessed Structural Images

**Minimum preprocessing**:
1. ✅ Skull stripping / brain extraction
2. ✅ Bias field correction (inhomogeneity correction)
3. ✅ Spatial normalization to standard space (MNI152)
4. ✅ Segmentation (GM, WM, CSF)

**Optional**:
- Denoising
- Resolution standardization
- Intensity normalization

```python
# Example: Load preprocessed brain-extracted image
brain_img = nib.load('subject_001_T1w_brain.nii.gz')
brain_data = brain_img.get_fdata()
```

### 3. Derived Features

#### A. Tissue Segmentation Maps

**Format**: NIfTI (probability maps or binary masks)  
**Channels**: Gray matter (GM), White matter (WM), CSF

```python
# Load tissue probability maps from FreeSurfer/FSL/SPM
gm_prob = nib.load('subject_001_GM_prob.nii.gz').get_fdata()
wm_prob = nib.load('subject_001_WM_prob.nii.gz').get_fdata()
csf_prob = nib.load('subject_001_CSF_prob.nii.gz').get_fdata()

# Stack as multi-channel input (for CNNs)
tissue_stack = np.stack([gm_prob, wm_prob, csf_prob], axis=-1)
# Shape: (x, y, z, 3)
```

#### B. FreeSurfer-Derived Metrics

**Format**: CSV or NumPy array  
**Features**: Regional volumes, cortical thickness, surface area

```python
import pandas as pd

# Load FreeSurfer stats
fs_stats = pd.read_csv('subject_001_aparc_stats.csv')
# Columns: region, volume_mm3, thickness_mm, surface_area_mm2

# Extract feature vector
features = fs_stats[['volume_mm3', 'thickness_mm']].values.flatten()
# Shape: (n_regions * 2,)
```

**Common FreeSurfer outputs**:
- `aseg.stats`: Subcortical volumes (hippocampus, amygdala, etc.)
- `aparc.stats`: Cortical parcellation (Desikan-Killiany atlas, 68 regions)
- `aparc.a2009s.stats`: Destrieux atlas (148 regions)

#### C. Voxel-Based Morphometry (VBM)

**Format**: NIfTI (modulated GM/WM maps)  
**Use case**: Voxel-wise comparison of tissue density

```python
# VBM: Modulated, smoothed gray matter
vbm_gm = nib.load('subject_001_VBM_GM.nii.gz').get_fdata()
# Typically smoothed with 8mm FWHM Gaussian kernel
```

## Metadata Requirements

```yaml
dataset_id: ukb_smri
name: UK Biobank Structural MRI
modality: sMRI
sequence: T1-weighted
n_subjects: 40000
resolution: [1.0, 1.0, 1.0]  # mm (x, y, z)
field_strength: 3T
manufacturer: Siemens
preprocessing: UK Biobank Pipeline
standard_space: MNI152NLin2009cAsym
atlas: Desikan-Killiany  # or Destrieux, AAL, etc.
```

## Quality Control Metrics

### 1. Image Quality Assessment

```python
def compute_snr(img_data, brain_mask):
    """
    Signal-to-noise ratio for structural MRI.
    
    Args:
        img_data: (x, y, z) - T1w image
        brain_mask: (x, y, z) - binary brain mask
    
    Returns:
        snr: Signal-to-noise ratio
    """
    brain_signal = img_data[brain_mask > 0]
    
    # Estimate noise from background (air)
    background = img_data[brain_mask == 0]
    noise_std = background.std()
    
    snr = brain_signal.mean() / (noise_std + 1e-8)
    
    return snr

# Typical SNR for 3T: 20-40
snr = compute_snr(data, brain_mask)
print(f"SNR: {snr:.2f}")
```

### 2. Contrast-to-Noise Ratio (CNR)

```python
def compute_cnr(img_data, gm_mask, wm_mask):
    """
    Contrast-to-noise ratio between gray and white matter.
    """
    gm_signal = img_data[gm_mask > 0].mean()
    wm_signal = img_data[wm_mask > 0].mean()
    
    noise_std = img_data[gm_mask > 0].std()
    
    cnr = abs(gm_signal - wm_signal) / (noise_std + 1e-8)
    
    return cnr

cnr = compute_cnr(data, gm_mask, wm_mask)
print(f"CNR (GM-WM): {cnr:.2f}")
```

### 3. FreeSurfer QC Metrics

```python
# Check FreeSurfer reconstruction quality
def check_freesurfer_qc(subjects_dir, subject_id):
    """
    Check FreeSurfer quality control metrics.
    """
    # Euler number (lower is better, typically < -50)
    euler_path = f"{subjects_dir}/{subject_id}/stats/lh.aparc.stats"
    
    # Mean cortical thickness (typical: 2.0-3.0 mm)
    # Total brain volume (typical: 1000-1500 cm³)
    
    # Flag for manual inspection if outliers
    pass
```

## Data Normalization

### 1. Intensity Normalization

```python
from sklearn.preprocessing import StandardScaler

def normalize_intensity(img_data, brain_mask):
    """
    Z-score normalization within brain mask.
    """
    brain_voxels = img_data[brain_mask > 0]
    
    mean = brain_voxels.mean()
    std = brain_voxels.std()
    
    img_normalized = (img_data - mean) / (std + 1e-8)
    img_normalized[brain_mask == 0] = 0  # Keep background at 0
    
    return img_normalized

normalized = normalize_intensity(data, brain_mask)
```

### 2. Total Intracranial Volume (TIV) Correction

```python
def tiv_correction(regional_volumes, tiv):
    """
    Correct regional volumes for head size.
    
    Args:
        regional_volumes: (n_regions,) - volumes in mm³
        tiv: Total intracranial volume in mm³
    
    Returns:
        corrected_volumes: TIV-corrected volumes
    """
    # Method 1: Proportional scaling
    corrected = (regional_volumes / tiv) * 1500000  # Scale to 1.5L
    
    # Method 2: Residuals after regression (preferred)
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression()
    lr.fit(tiv.reshape(-1, 1), regional_volumes.reshape(-1, 1))
    residuals = regional_volumes - lr.predict(tiv.reshape(-1, 1)).ravel()
    
    return residuals

# Load TIV from FreeSurfer
# Typically in aseg.stats: "Estimated Total Intracranial Volume"
tiv = 1450000  # mm³

volumes_corrected = tiv_correction(regional_volumes, tiv)
```

## Data Augmentation

### For Deep Learning Models

```python
import torchio as tio

# Define augmentation pipeline
transforms = tio.Compose([
    tio.RandomAffine(
        scales=(0.9, 1.1),
        degrees=10,
        translation=5,
        p=0.75
    ),
    tio.RandomFlip(axes=('LR',), p=0.5),  # Left-Right flip
    tio.RandomNoise(std=(0, 0.1), p=0.25),
    tio.RandomBiasField(coefficients=0.5, p=0.25),
])

# Apply to image
subject = tio.Subject(
    t1=tio.ScalarImage('subject_001_T1w.nii.gz'),
)
augmented = transforms(subject)
augmented_data = augmented.t1.data  # Shape: (1, x, y, z)
```

## Benchmark Tasks

### 1. Classification

**Typical tasks**:
- Sex classification
- Age group classification
- Site/scanner classification (for robustness testing)

**Input**: T1w images `(x, y, z)` or derived features  
**Output**: Class labels

```python
# Example: Binary classification
X_train = load_t1w_images(train_ids)  # (n_train, x, y, z)
y_train = load_labels(train_ids)  # (n_train,) - {0, 1}
```

### 2. Regression

**Typical tasks**:
- Age prediction (brain age)
- Phenotype prediction
- Continuous score prediction

**Input**: T1w images  
**Output**: Continuous values

```python
# Example: Brain age prediction
X = load_t1w_images(subject_ids)  # (n, x, y, z)
y_age = load_ages(subject_ids)  # (n,) - chronological age in years
```

### 3. Segmentation

**Typical tasks**:
- Tissue segmentation (GM, WM, CSF)
- Lesion segmentation (MS, stroke, tumor)
- Hippocampal subfield segmentation

**Input**: T1w images  
**Output**: Segmentation masks `(x, y, z)` or `(x, y, z, n_classes)`

### 4. Reconstruction/Denoising

**Input**: Low-quality or partial images  
**Output**: High-quality reconstructed images

## Example Data Loading

### Using Nilearn

```python
from nilearn import datasets, plotting
from nilearn.image import resample_to_img

# Load example T1w image
t1_img = nib.load('subject_001_T1w.nii.gz')

# Resample to standard resolution (e.g., 2mm isotropic)
from nilearn.image import resample_img
t1_resampled = resample_img(
    t1_img,
    target_affine=np.diag([2, 2, 2]),
    interpolation='continuous'
)

# Visualize
plotting.plot_anat(t1_resampled)
```

### Loading FreeSurfer-Derived Features

```python
def load_freesurfer_features(subjects_dir, subject_id):
    """
    Load FreeSurfer morphometric features.
    
    Returns:
        features: Dictionary with volumes, thickness, etc.
    """
    import re
    
    # Parse aparc.stats
    stats_file = f"{subjects_dir}/{subject_id}/stats/lh.aparc.stats"
    
    features = {}
    with open(stats_file, 'r') as f:
        for line in f:
            if line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 5:
                region = parts[0]
                features[f"lh_{region}_volume"] = float(parts[3])
                features[f"lh_{region}_thickness"] = float(parts[4])
    
    # Repeat for right hemisphere
    # ... (similar code for rh.aparc.stats)
    
    return features
```

## ITU AI4H Alignment

This specification aligns with:

- **DEL10.8 Section 3.1**: Input data specifications for neurology
- **DEL3 Section 4.2**: Data format and quality requirements
- **DEL0.1**: Standardized neuroimaging terminology

## Tools & Libraries

### Preprocessing
- **FreeSurfer**: Comprehensive cortical reconstruction
- **FSL**: FMRIB Software Library (BET, FAST, FLIRT/FNIRT)
- **SPM**: Statistical Parametric Mapping
- **ANTs**: Advanced Normalization Tools
- **CAT12**: Computational Anatomy Toolbox

### Python Libraries
- **NiBabel**: Read/write neuroimaging formats
- **Nilearn**: Machine learning for neuroimaging
- **TorchIO**: Medical image augmentation
- **MONAI**: Medical imaging deep learning

## References

1. Fischl, B. (2012). FreeSurfer. *NeuroImage*, 62(2), 774-781.
2. Ashburner, J., & Friston, K. J. (2000). Voxel-based morphometry. *NeuroImage*, 11(6), 805-821.
3. Klein, A., et al. (2009). Evaluation of volume-based and surface-based brain image registration methods. *NeuroImage*, 51(1), 214-220.

## Related Documentation

- [fMRI Specifications](fmri.md)
- [Genomics Specifications](genomics.md)
- [Prediction Baselines](../analysis_recipes/prediction_baselines.md)








