# fMRI Data Specifications

## Overview

Functional MRI (fMRI) measures brain activity by detecting changes in blood oxygenation (BOLD signal). This page defines the standardized data formats and preprocessing requirements for fMRI data in this benchmark hub.

## Data Format Requirements

### 1. Raw fMRI Time Series

**Format**: NIfTI (`.nii` or `.nii.gz`) or NumPy array  
**Shape**: `(n_samples, n_timepoints, n_voxels)` or `(n_samples, n_voxels, n_timepoints)`  
**Data Type**: `float32` or `float64`

```python
import numpy as np
import nibabel as nib

# Load NIfTI file
img = nib.load('subject_001_bold.nii.gz')
data = img.get_fdata()  # Shape: (x, y, z, time)

# Reshape to 2D: (time, voxels)
n_timepoints = data.shape[-1]
timeseries = data.reshape(-1, n_timepoints).T  # (time, voxels)
```

### 2. Preprocessed Time Series (Recommended)

**Minimum preprocessing steps**:
1. ✅ Motion correction
2. ✅ Slice timing correction
3. ✅ Spatial normalization to standard space (MNI152)
4. ✅ Nuisance regression (motion parameters, CSF, white matter)
5. ✅ Bandpass filtering (0.01 - 0.1 Hz typical for resting-state)

**Optional**:
- Spatial smoothing (6-8mm FWHM)
- Global signal regression (controversial)
- Denoising (ICA-AROMA, CompCor)

### 3. Parcellated/ROI Time Series

**Format**: NumPy array or CSV  
**Shape**: `(n_samples, n_timepoints, n_regions)`

```python
# Example: 400 parcels (Schaefer atlas), 200 timepoints
roi_timeseries = np.load('subject_timeseries.npy')  
# Shape: (1, 200, 400)
```

**Recommended atlases**:
- **Schaefer 2018** (100-1000 parcels, 7 or 17 networks)
- **AAL3** (170 regions)
- **Gordon** (333 parcels)
- **Harvard-Oxford** (cortical + subcortical)
- **Glasser MMP** (360 parcels)

### 4. Connectivity Matrices

**Format**: NumPy array  
**Shape**: `(n_samples, n_regions, n_regions)` or `(n_samples, n_features)` (vectorized)

```python
from nilearn.connectome import ConnectivityMeasure

# Compute functional connectivity
conn_measure = ConnectivityMeasure(kind='correlation')
connectivity = conn_measure.fit_transform([roi_timeseries])
# Shape: (1, n_regions, n_regions)

# Vectorize upper triangle (for ML)
from sklearn.utils import check_array
import numpy as np

def vectorize_connectivity(conn_mat):
    """Extract upper triangle as feature vector."""
    n_regions = conn_mat.shape[0]
    triu_idx = np.triu_indices(n_regions, k=1)
    return conn_mat[triu_idx]

features = vectorize_connectivity(connectivity[0])
# Shape: (n_regions * (n_regions-1) / 2,)
```

## Metadata Requirements

Each fMRI dataset should include metadata:

```yaml
dataset_id: ukb_fmri_tensor
name: UK Biobank fMRI Tensors
modality: fmri
task: resting_state  # or 'task_based'
n_subjects: 40000
n_timepoints: 490
n_voxels: 91282  # or n_regions if parcellated
tr: 0.735  # Repetition time in seconds
preprocessing: fmriprep_20.2.0
atlas: Schaefer2018_400  # if parcellated
bandpass: [0.01, 0.1]  # Hz
smoothing: 6mm  # FWHM, or null
standard_space: MNI152NLin6Asym
```

## Quality Control Metrics

### 1. Motion Parameters

**Framewise Displacement (FD)**: Measure of head motion

```python
def framewise_displacement(motion_params):
    """
    Calculate framewise displacement (Power et al. 2012).
    
    Args:
        motion_params: (n_timepoints, 6) - 3 translations + 3 rotations
    
    Returns:
        fd: (n_timepoints-1,) - framewise displacement
    """
    # Translations in mm
    trans = motion_params[:, :3]
    
    # Rotations converted to mm (50mm sphere radius)
    rot = motion_params[:, 3:] * 50  # radians to mm
    
    # Absolute derivatives
    dtrans = np.abs(np.diff(trans, axis=0))
    drot = np.abs(np.diff(rot, axis=0))
    
    # Sum
    fd = np.sum(dtrans, axis=1) + np.sum(drot, axis=1)
    
    return fd

# Recommended threshold: FD < 0.5mm for resting-state
mean_fd = fd.mean()
print(f"Mean FD: {mean_fd:.3f} mm")

# Exclude high-motion timepoints (scrubbing)
low_motion_frames = fd < 0.5
clean_timeseries = timeseries[low_motion_frames]
```

### 2. Temporal SNR (tSNR)

```python
def temporal_snr(timeseries):
    """
    Temporal signal-to-noise ratio.
    
    Args:
        timeseries: (n_timepoints, n_voxels)
    
    Returns:
        tsnr: (n_voxels,) - temporal SNR per voxel
    """
    mean_signal = timeseries.mean(axis=0)
    std_signal = timeseries.std(axis=0)
    
    tsnr = mean_signal / (std_signal + 1e-8)
    
    return tsnr

tsnr = temporal_snr(timeseries)
print(f"Mean tSNR: {tsnr.mean():.2f}")

# Typical values: 50-100 for 3T, higher for 7T
```

### 3. Data Completeness

```python
# Check for missing data
n_nan = np.isnan(timeseries).sum()
completeness = 1 - (n_nan / timeseries.size)
print(f"Data completeness: {completeness*100:.1f}%")

# Minimum recommended: 95% completeness
assert completeness > 0.95, "Too much missing data"
```

## Data Augmentation for Robustness Testing

See our [robustness testing documentation](../../index.md#robustness-testing) for details on perturbations:

```python
from fmbench.robustness import (
    ChannelDropout,
    GaussianNoise,
    LineNoise,
    TemporalShift
)

# Example: Add Gaussian noise
noise_probe = GaussianNoise(snr_db=10)
noisy_data = noise_probe.apply(timeseries)
```

## Example Data Loading

### From NIfTI

```python
import nibabel as nib
from nilearn.maskers import NiftiMasker

# Load functional image
func_img = nib.load('subject_001_bold.nii.gz')

# Apply brain mask and extract timeseries
masker = NiftiMasker(
    standardize=True,
    detrend=True,
    low_pass=0.1,
    high_pass=0.01,
    t_r=2.0
)
timeseries = masker.fit_transform(func_img)
# Shape: (n_timepoints, n_voxels)
```

### From Parcellated CSV

```python
import pandas as pd

# Load ROI timeseries
df = pd.read_csv('subject_001_schaefer400.csv')
# Columns: timepoint, region_1, region_2, ..., region_400

timeseries = df.iloc[:, 1:].values  # Exclude timepoint column
# Shape: (n_timepoints, 400)
```

### From HCP-style Data

```python
# HCP stores timeseries as CIFTI (cortical surface + subcortical)
from nibabel import cifti2

cifti_img = cifti2.load('subject_REST_LR.dtseries.nii')
timeseries = cifti_img.get_fdata()
# Shape: (n_timepoints, 91282) - 91k grayordinates
```

## Benchmark Tasks

### 1. Classification

**Typical tasks**:
- Disease vs. Control (AD, ADHD, ASD, schizophrenia)
- Cognitive state classification
- Task decoding

**Input**: `(n_samples, n_timepoints, n_regions)` or connectivity matrices  
**Output**: Class labels `(n_samples,)`

### 2. Reconstruction

**Typical tasks**:
- Masked autoencoding (predict masked timepoints/regions)
- Denoising
- Super-resolution (spatial or temporal)

**Input**: Masked/noisy timeseries  
**Output**: Clean timeseries

### 3. Regression

**Typical tasks**:
- Cognitive score prediction
- Age prediction
- Symptom severity prediction

**Input**: Timeseries  
**Output**: Continuous values `(n_samples,)`

## ITU AI4H Alignment

This specification aligns with:

- **DEL10.8 Section 3.1**: Input data specifications for neurology
- **DEL3 Section 4.2**: Data format requirements
- **DEL0.1**: Standardized terminology (BOLD, fMRI, parcellation)

## Tools & Libraries

### Preprocessing
- **fMRIPrep**: Robust preprocessing pipeline
- **CONN Toolbox**: Connectivity preprocessing
- **DPARSF**: Data Processing Assistant for Resting-State fMRI

### Analysis
- **Nilearn**: Machine learning for neuroimaging
- **NiBabel**: Read/write neuroimaging formats
- **BrainIAK**: Brain Imaging Analysis Kit

### Parcellation
- **Schaefer2018**: `nilearn.datasets.fetch_atlas_schaefer_2018()`
- **AAL3**: `nilearn.datasets.fetch_atlas_aal()`

## References

1. Esteban, O., et al. (2019). fMRIPrep: a robust preprocessing pipeline for fMRI data. *Nature Methods*, 16(1), 111-116.
2. Power, J. D., et al. (2012). Spurious but systematic correlations in functional connectivity MRI. *NeuroImage*, 59(3), 2142-2154.
3. Schaefer, A., et al. (2018). Local-Global Parcellation of the Human Cerebral Cortex. *Cerebral Cortex*, 28(9), 3095-3114.

## Related Documentation

- [sMRI Specifications](smri.md)
- [Genomics Specifications](genomics.md)
- [Robustness Testing](../../index.md#robustness-testing)



