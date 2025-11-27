"""
Robustness evaluation module for foundation model benchmarking.

This module provides robustness testing for foundation models across multiple
modalities (EEG, fMRI, MRI, Genomics), measuring how model outputs degrade
under realistic perturbations.

Supported Modalities:
- EEG/Time-series: channel dropout, line noise, temporal shift
- fMRI: ROI dropout, temporal noise, volume shift
- Structural MRI: voxel noise, spatial perturbations
- Genomics: feature dropout, sequence noise, masking

Universal Probes:
- Gaussian noise (SNR degradation)
- Feature/dimension dropout
- Permutation equivariance testing

Inspired by brainaug-lab methodology. Self-contained - no external dependencies
beyond numpy.

Reference: ITU FG-AI4H DEL3 - Generalizability requirements
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, List, Optional, Sequence

import numpy as np


# ============================================================================
# Core metrics
# ============================================================================

def cosine_similarity(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    Compute per-sample cosine similarity between arrays.
    
    Args:
        a, b: Arrays of shape (N, D)
        eps: Small constant for numerical stability
    
    Returns:
        Array of shape (N,) with cosine similarities
    """
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch {a.shape} vs {b.shape}")
    
    # Flatten to 2D if needed
    if a.ndim == 1:
        a = a.reshape(1, -1)
        b = b.reshape(1, -1)
    elif a.ndim > 2:
        a = a.reshape(a.shape[0], -1)
        b = b.reshape(b.shape[0], -1)
    
    dot = np.sum(a * b, axis=1)
    na = np.linalg.norm(a, axis=1) + eps
    nb = np.linalg.norm(b, axis=1) + eps
    return dot / (na * nb)


def r_auc(xs: Sequence[float], ys: Sequence[float], normalize: bool = True) -> float:
    """
    Compute Reverse Area Under Curve for robustness evaluation.
    
    Measures the area under the similarity vs perturbation curve.
    Higher rAUC indicates better robustness (output stability).
    
    Args:
        xs: Perturbation strength values (grid points)
        ys: Corresponding similarity values
        normalize: If True, normalize to [0, 1] range
    
    Returns:
        rAUC score (higher = more robust)
    """
    x = np.asarray(list(xs), dtype=float)
    y = np.asarray(list(ys), dtype=float)
    
    if x.size < 2:
        return 0.0
    
    # Sort by x for proper integration
    idx = np.argsort(x)
    x, y = x[idx], y[idx]
    
    auc = float(np.trapz(y, x))
    
    if not normalize:
        return auc
    
    xspan = float(np.max(x) - np.min(x))
    if xspan <= 0:
        return 0.0
    
    return auc / xspan


# ============================================================================
# Modality Detection
# ============================================================================

def detect_modality(x: np.ndarray) -> str:
    """
    Detect data modality from array shape.
    
    Returns:
        'genomics': 2D (samples, features) - gene expression, sequences
        'timeseries': 3D (samples, channels, time) - EEG, fMRI
        'volumetric': 4D (samples, x, y, z) - structural MRI
    """
    if x.ndim == 2:
        return "genomics"
    elif x.ndim == 3:
        return "timeseries"
    elif x.ndim == 4:
        return "volumetric"
    else:
        return "unknown"


# ============================================================================
# Universal Perturbation Functions (work across modalities)
# ============================================================================

def _snr_to_noise_std(signal: np.ndarray, target_snr_db: float) -> float:
    """Convert SNR in dB to noise standard deviation."""
    sig_power = float(np.mean(signal ** 2))
    if sig_power <= 1e-12:
        sig_power = 1e-12
    ratio = 10.0 ** (target_snr_db / 10.0)
    noise_power = sig_power / ratio
    return float(np.sqrt(max(noise_power, 1e-12)))


def feature_dropout(x: np.ndarray, p: float, rng: np.random.Generator, axis: int = 1) -> np.ndarray:
    """
    Drop random features/channels/ROIs with probability p.
    
    Works across modalities:
    - Genomics (2D): drops genes/features along axis 1
    - EEG/fMRI (3D): drops channels/ROIs along axis 1  
    - Volumetric (4D): drops slices along specified axis
    
    Args:
        x: Input array of any shape
        p: Dropout probability per feature
        rng: Random number generator
        axis: Axis along which to drop (default: 1 for features/channels)
    
    Returns:
        Array with some features zeroed out
    """
    out = x.copy()
    p = float(np.clip(p, 0.0, 1.0))
    
    if x.ndim == 2:
        # Genomics: (N, features)
        N, F = x.shape
        for n in range(N):
            k = rng.binomial(F, p)
            if k > 0:
                idx = rng.choice(F, size=k, replace=False)
                out[n, idx] = 0.0
    elif x.ndim == 3:
        # Time-series: (N, C, T)
        N, C, T = x.shape
        for n in range(N):
            k = rng.binomial(C, p)
            if k > 0:
                idx = rng.choice(C, size=k, replace=False)
                out[n, idx, :] = 0.0
    elif x.ndim == 4:
        # Volumetric: (N, X, Y, Z) - drop along axis
        N = x.shape[0]
        dim_size = x.shape[axis]
        for n in range(N):
            k = rng.binomial(dim_size, p)
            if k > 0:
                idx = rng.choice(dim_size, size=k, replace=False)
                slices = [slice(None)] * x.ndim
                slices[0] = n
                for i in idx:
                    slices[axis] = i
                    out[tuple(slices)] = 0.0
    
    return out


# Alias for backward compatibility
channel_dropout = feature_dropout


def add_gaussian_noise(x: np.ndarray, snr_db: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add Gaussian white noise at specified SNR.
    
    Works universally across all modalities:
    - Genomics: simulates measurement noise in gene expression
    - EEG/fMRI: simulates physiological/scanner noise
    - MRI: simulates thermal/acquisition noise
    
    Args:
        x: Input array of any shape
        snr_db: Target signal-to-noise ratio in decibels
        rng: Random number generator
    
    Returns:
        Noisy signal
    """
    std = _snr_to_noise_std(x, snr_db)
    noise = rng.normal(0.0, std, size=x.shape).astype(np.float32)
    return (x + noise).astype(np.float32)


def add_line_noise(
    x: np.ndarray,
    fs: int,
    mains: Sequence[int],
    snr_db: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Add powerline interference at specified frequencies."""
    if x.ndim == 2:
        x = x[np.newaxis, ...]
    N, C, T = x.shape
    t = np.arange(T, dtype=np.float32) / float(fs)
    line = np.zeros_like(x)
    for ch in range(C):
        sig = np.zeros(T, dtype=np.float32)
        for f in mains:
            phase = rng.uniform(0, 2 * np.pi)
            sig += np.sin(2 * np.pi * float(f) * t + phase).astype(np.float32)
        sig_rms = np.sqrt(np.mean(sig ** 2)) + 1e-8
        sig = sig / sig_rms
        line[:, ch, :] = sig
    std = _snr_to_noise_std(x, snr_db)
    line = line * std
    return (x + line).astype(np.float32)


def feature_permute(x: np.ndarray, rng: np.random.Generator, axis: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """
    Randomly permute features/channels/genes.
    
    Tests whether model representations are invariant to feature ordering.
    
    Works across modalities:
    - Genomics (2D): permutes gene/feature order
    - EEG/fMRI (3D): permutes channel/ROI order
    - Volumetric (4D): permutes along specified axis
    
    Args:
        x: Input array of any shape
        rng: Random number generator
        axis: Axis to permute (default: 1)
    
    Returns:
        Tuple of (permuted array, permutation indices)
    """
    dim_size = x.shape[axis]
    perm = rng.permutation(dim_size)
    
    # Use advanced indexing to permute along axis
    out = np.take(x, perm, axis=axis)
    return out, perm


# Alias for backward compatibility
channel_permute = feature_permute


def temporal_shift(x: np.ndarray, shift: int, mode: str = "circular") -> np.ndarray:
    """
    Shift signal along time/sequence dimension.
    
    Works across modalities:
    - Time-series (3D): shifts along time axis
    - Genomics (2D): shifts along feature axis (positional)
    
    Args:
        x: Input array
        shift: Number of positions to shift
        mode: 'circular' or 'zero' padding
    
    Returns:
        Shifted array
    """
    shift_axis = -1  # Always shift along last axis
    
    if mode == "circular":
        return np.roll(x, shift, axis=shift_axis)
    else:
        out = np.zeros_like(x)
        dim_size = x.shape[shift_axis]
        if shift >= 0 and shift < dim_size:
            src_slice = [slice(None)] * x.ndim
            dst_slice = [slice(None)] * x.ndim
            src_slice[shift_axis] = slice(None, dim_size - shift)
            dst_slice[shift_axis] = slice(shift, None)
            out[tuple(dst_slice)] = x[tuple(src_slice)]
        elif shift < 0 and -shift < dim_size:
            s = -shift
            src_slice = [slice(None)] * x.ndim
            dst_slice = [slice(None)] * x.ndim
            src_slice[shift_axis] = slice(s, None)
            dst_slice[shift_axis] = slice(None, dim_size - s)
            out[tuple(dst_slice)] = x[tuple(src_slice)]
        return out


# ============================================================================
# Genomics-Specific Perturbations
# ============================================================================

def sequence_mask(x: np.ndarray, mask_prob: float, rng: np.random.Generator,
                  mask_value: float = 0.0) -> np.ndarray:
    """
    Mask random positions (BERT-style masking for genomics).
    
    Simulates missing data or tests model's ability to handle incomplete sequences.
    
    Args:
        x: Input array (N, seq_len) or (N, features)
        mask_prob: Probability of masking each position
        rng: Random number generator
        mask_value: Value to use for masked positions
    
    Returns:
        Masked array
    """
    mask = rng.random(x.shape) < mask_prob
    out = x.copy()
    out[mask] = mask_value
    return out.astype(np.float32)


def expression_noise(x: np.ndarray, scale: float, rng: np.random.Generator) -> np.ndarray:
    """
    Add multiplicative noise (gene expression variability).
    
    Models biological and technical variability in expression data.
    
    Args:
        x: Gene expression array (N, genes)
        scale: Scale of log-normal noise (0.1 = 10% variability)
        rng: Random number generator
    
    Returns:
        Noisy expression data
    """
    noise = np.exp(rng.normal(0, scale, size=x.shape))
    return (x * noise).astype(np.float32)


# ============================================================================
# MRI-Specific Perturbations
# ============================================================================

def intensity_shift(x: np.ndarray, shift_range: float, rng: np.random.Generator) -> np.ndarray:
    """
    Shift intensity values (scanner calibration differences).
    
    Args:
        x: MRI volume or image array
        shift_range: Range of shift as fraction of std
        rng: Random number generator
    
    Returns:
        Intensity-shifted array
    """
    shift = rng.uniform(-shift_range, shift_range) * np.std(x)
    return (x + shift).astype(np.float32)


def contrast_variation(x: np.ndarray, scale_range: tuple, rng: np.random.Generator) -> np.ndarray:
    """
    Vary contrast (scanner/protocol differences).
    
    Args:
        x: MRI volume or image array
        scale_range: (min_scale, max_scale) for contrast
        rng: Random number generator
    
    Returns:
        Contrast-adjusted array
    """
    mean_val = np.mean(x)
    scale = rng.uniform(scale_range[0], scale_range[1])
    return ((x - mean_val) * scale + mean_val).astype(np.float32)


# ============================================================================
# Model Adapter Interface
# ============================================================================

class RobustnessAdapter:
    """
    Adapter that wraps an fmbench model for robustness testing.
    
    The model must implement:
    - predict(X) or predict_proba(X) for classification
    - get_embeddings(X) for embedding extraction (optional)
    """
    
    def __init__(self, model: Any, flatten_input: bool = True):
        """
        Args:
            model: An fmbench model instance
            flatten_input: Whether to flatten (N,C,T) to (N, C*T) for sklearn-style models
        """
        self.model = model
        self.flatten_input = flatten_input
        self._has_embeddings = hasattr(model, "get_embeddings")
        self._has_predict_proba = hasattr(model, "predict_proba")
    
    def _prepare_input(self, x: np.ndarray) -> np.ndarray:
        """Prepare input for the model."""
        if self.flatten_input and x.ndim == 3:
            # Flatten (N, C, T) -> (N, C*T)
            return x.reshape(x.shape[0], -1)
        return x
    
    def get_embeddings(self, x: np.ndarray) -> np.ndarray:
        """Get embeddings from the model."""
        x_prep = self._prepare_input(x)
        if self._has_embeddings:
            return self.model.get_embeddings(x_prep)
        # Fallback: use logits as embeddings
        return self.get_logits(x)
    
    def get_logits(self, x: np.ndarray) -> np.ndarray:
        """Get logits/predictions from the model."""
        x_prep = self._prepare_input(x)
        if self._has_predict_proba:
            return self.model.predict_proba(x_prep)
        else:
            preds = self.model.predict(x_prep)
            # Convert to one-hot style if needed
            if preds.ndim == 1:
                n_classes = int(np.max(preds)) + 1
                logits = np.zeros((len(preds), max(2, n_classes)))
                logits[np.arange(len(preds)), preds.astype(int)] = 1.0
                return logits
            return preds


# ============================================================================
# Robustness Probes
# ============================================================================

def _summarize_cosine_series(base: np.ndarray, aug: np.ndarray) -> tuple[float, float]:
    """Compute mean and std of per-sample cosine similarities."""
    cs = cosine_similarity(base, aug)
    return float(np.mean(cs)), float(np.std(cs))


def dropout_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    grid_p: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe model robustness to feature/channel dropout.
    
    Works across modalities:
    - Genomics: gene/feature dropout
    - EEG/fMRI: channel/ROI dropout
    - MRI: slice/voxel dropout
    
    Args:
        adapter: Model adapter
        x: Input data of any shape
        grid_p: Dropout probabilities to test
        y: Optional labels for accuracy tracking
        rng: Random number generator
    
    Returns:
        Dict with grid, logit/embedding similarity curves, and rAUC
    """
    rng = np.random.default_rng() if rng is None else rng
    modality = detect_modality(x)
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    delta_acc = []
    
    for p in grid_p:
        x_aug = feature_dropout(x, float(p), rng)
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
        
        # Track accuracy delta if labels provided
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            aug_pred = np.argmax(logits_aug, axis=1)
            base_acc = float(np.mean(base_pred == y))
            aug_acc = float(np.mean(aug_pred == y))
            delta_acc.append(aug_acc - base_acc)
    
    return {
        "probe_type": "dropout",
        "modality": modality,
        "grid": list(map(float, grid_p)),
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "rAUC": r_auc(grid_p, logit_means, normalize=True),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "rAUC": r_auc(grid_p, emb_means, normalize=True),
        },
        "delta_accuracy": delta_acc if delta_acc else None,
    }


def noise_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    snr_grid: Sequence[float] = (40, 30, 20, 10, 5, 0),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe model robustness to Gaussian noise at various SNR levels.
    
    Args:
        adapter: Model adapter
        x: Input data (N, C, T)
        snr_grid: SNR values in dB to test (descending = increasing noise)
        y: Optional labels for accuracy tracking
        rng: Random number generator
    
    Returns:
        Dict with grid, similarity curves, and rAUC
    """
    rng = np.random.default_rng() if rng is None else rng
    
    if x.ndim == 2:
        x = x[:, np.newaxis, :]
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    delta_acc = []
    
    for snr in snr_grid:
        x_aug = add_gaussian_noise(x, float(snr), rng)
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            aug_pred = np.argmax(logits_aug, axis=1)
            delta_acc.append(float(np.mean(aug_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "gaussian_noise",
        "grid": list(map(float, snr_grid)),
        "grid_unit": "dB (SNR)",
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "rAUC": r_auc(snr_grid, logit_means, normalize=True),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "rAUC": r_auc(snr_grid, emb_means, normalize=True),
        },
        "delta_accuracy": delta_acc if delta_acc else None,
    }


def line_noise_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    fs: int = 200,
    mains: Sequence[int] = (50, 60),
    snr_grid: Sequence[float] = (40, 30, 20, 10, 5),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe model robustness to powerline interference.
    
    Args:
        adapter: Model adapter
        x: Input data (N, C, T)
        fs: Sampling frequency in Hz
        mains: Powerline frequencies to inject (50Hz EU, 60Hz US)
        snr_grid: SNR values in dB
        y: Optional labels
        rng: Random number generator
    
    Returns:
        Dict with similarity curves and rAUC
    """
    rng = np.random.default_rng() if rng is None else rng
    
    if x.ndim == 2:
        x = x[:, np.newaxis, :]
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    delta_acc = []
    
    for snr in snr_grid:
        x_aug = add_line_noise(x, fs=fs, mains=list(mains), snr_db=float(snr), rng=rng)
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            aug_pred = np.argmax(logits_aug, axis=1)
            delta_acc.append(float(np.mean(aug_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "line_noise",
        "grid": list(map(float, snr_grid)),
        "grid_unit": "dB (SNR)",
        "mains_hz": list(mains),
        "sampling_freq": fs,
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "rAUC": r_auc(snr_grid, logit_means, normalize=True),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "rAUC": r_auc(snr_grid, emb_means, normalize=True),
        },
        "delta_accuracy": delta_acc if delta_acc else None,
    }


def permutation_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    num_perms: int = 10,
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe model equivariance to feature/channel permutation.
    
    Works across modalities:
    - Genomics: gene ordering invariance
    - EEG/fMRI: channel/ROI ordering invariance
    
    High similarity indicates the model's representations are invariant
    to feature ordering (often desirable).
    
    Args:
        adapter: Model adapter
        x: Input data of any shape
        num_perms: Number of random permutations to test
        y: Optional labels
        rng: Random number generator
    
    Returns:
        Dict with similarity statistics
    """
    rng = np.random.default_rng() if rng is None else rng
    modality = detect_modality(x)
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    emb_sims, logit_sims = [], []
    acc_deltas = []
    
    for _ in range(num_perms):
        x_perm, perm = feature_permute(x, rng)
        
        emb_perm = adapter.get_embeddings(x_perm)
        emb_sim = float(np.mean(cosine_similarity(base_emb, emb_perm)))
        emb_sims.append(emb_sim)
        
        logits_perm = adapter.get_logits(x_perm)
        logit_sim = float(np.mean(cosine_similarity(base_logits, logits_perm)))
        logit_sims.append(logit_sim)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            perm_pred = np.argmax(logits_perm, axis=1)
            acc_deltas.append(float(np.mean(perm_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "permutation",
        "num_permutations": num_perms,
        "embedding": {
            "sim_hist": emb_sims,
            "sim_mean": float(np.mean(emb_sims)),
            "sim_std": float(np.std(emb_sims)),
        },
        "logit": {
            "sim_hist": logit_sims,
            "sim_mean": float(np.mean(logit_sims)),
            "sim_std": float(np.std(logit_sims)),
        },
        "acc_delta_mean": float(np.mean(acc_deltas)) if acc_deltas else None,
    }


def shift_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    shifts: Sequence[int] = (0, 10, 20, 40, 80),
    y: Optional[np.ndarray] = None,
) -> Dict[str, Any]:
    """
    Probe model sensitivity to temporal shifts.
    
    Args:
        adapter: Model adapter
        x: Input data (N, C, T)
        shifts: Shift amounts in samples to test
        y: Optional labels
    
    Returns:
        Dict with similarity vs shift curves
    """
    if x.ndim == 2:
        x = x[:, np.newaxis, :]
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    emb_sims, logit_sims = [], []
    acc_deltas = []
    
    for s in shifts:
        x_shift = temporal_shift(x, int(s))
        
        emb_shift = adapter.get_embeddings(x_shift)
        emb_sim = float(np.mean(cosine_similarity(base_emb, emb_shift)))
        emb_sims.append(emb_sim)
        
        logits_shift = adapter.get_logits(x_shift)
        logit_sim = float(np.mean(cosine_similarity(base_logits, logits_shift)))
        logit_sims.append(logit_sim)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            shift_pred = np.argmax(logits_shift, axis=1)
            acc_deltas.append(float(np.mean(shift_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "temporal_shift",
        "shifts": list(map(int, shifts)),
        "embedding": {
            "sim_hist": emb_sims,
            "sim_mean": float(np.mean(emb_sims)),
            "rAUC": r_auc(shifts, emb_sims, normalize=True),
        },
        "logit": {
            "sim_hist": logit_sims,
            "sim_mean": float(np.mean(logit_sims)),
            "rAUC": r_auc(shifts, logit_sims, normalize=True),
        },
        "acc_delta": acc_deltas if acc_deltas else None,
    }


# ============================================================================
# Genomics-Specific Probes
# ============================================================================

def masking_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    mask_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.15, 0.2, 0.3),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe model robustness to random masking (BERT-style).
    
    Particularly relevant for genomics/sequence models.
    Tests how well models handle missing or masked data.
    
    Args:
        adapter: Model adapter
        x: Input data (N, seq_len) or (N, features)
        mask_grid: Masking probabilities to test
        y: Optional labels
        rng: Random number generator
    
    Returns:
        Dict with similarity curves and rAUC
    """
    rng = np.random.default_rng() if rng is None else rng
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    delta_acc = []
    
    for p in mask_grid:
        x_aug = sequence_mask(x, float(p), rng)
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            aug_pred = np.argmax(logits_aug, axis=1)
            delta_acc.append(float(np.mean(aug_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "masking",
        "modality": "genomics",
        "grid": list(map(float, mask_grid)),
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "rAUC": r_auc(mask_grid, logit_means, normalize=True),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "rAUC": r_auc(mask_grid, emb_means, normalize=True),
        },
        "delta_accuracy": delta_acc if delta_acc else None,
    }


def expression_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    scale_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3, 0.5),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe robustness to expression variability (multiplicative noise).
    
    Relevant for gene expression and single-cell data.
    Tests how models handle biological/technical variability.
    
    Args:
        adapter: Model adapter
        x: Expression data (N, genes)
        scale_grid: Log-normal noise scales to test
        y: Optional labels
        rng: Random number generator
    
    Returns:
        Dict with similarity curves and rAUC
    """
    rng = np.random.default_rng() if rng is None else rng
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    delta_acc = []
    
    for scale in scale_grid:
        if scale == 0:
            x_aug = x.copy()
        else:
            x_aug = expression_noise(x, float(scale), rng)
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            aug_pred = np.argmax(logits_aug, axis=1)
            delta_acc.append(float(np.mean(aug_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "expression_variability",
        "modality": "genomics",
        "grid": list(map(float, scale_grid)),
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "rAUC": r_auc(scale_grid, logit_means, normalize=True),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "rAUC": r_auc(scale_grid, emb_means, normalize=True),
        },
        "delta_accuracy": delta_acc if delta_acc else None,
    }


# ============================================================================
# MRI-Specific Probes
# ============================================================================

def intensity_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    shift_grid: Sequence[float] = (0.0, 0.05, 0.1, 0.2, 0.3),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe robustness to intensity shifts (scanner calibration).
    
    Relevant for MRI data where different scanners have different baselines.
    
    Args:
        adapter: Model adapter
        x: MRI data
        shift_grid: Intensity shift ranges (fraction of std)
        y: Optional labels
        rng: Random number generator
    
    Returns:
        Dict with similarity curves and rAUC
    """
    rng = np.random.default_rng() if rng is None else rng
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    delta_acc = []
    
    for shift_range in shift_grid:
        if shift_range == 0:
            x_aug = x.copy()
        else:
            x_aug = intensity_shift(x, float(shift_range), rng)
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
        
        if y is not None:
            base_pred = np.argmax(base_logits, axis=1)
            aug_pred = np.argmax(logits_aug, axis=1)
            delta_acc.append(float(np.mean(aug_pred == y) - np.mean(base_pred == y)))
    
    return {
        "probe_type": "intensity_shift",
        "modality": "mri",
        "grid": list(map(float, shift_grid)),
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "rAUC": r_auc(shift_grid, logit_means, normalize=True),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "rAUC": r_auc(shift_grid, emb_means, normalize=True),
        },
        "delta_accuracy": delta_acc if delta_acc else None,
    }


def contrast_probe(
    adapter: RobustnessAdapter,
    x: np.ndarray,
    scale_grid: Sequence[tuple] = ((1.0, 1.0), (0.9, 1.1), (0.8, 1.2), (0.7, 1.3), (0.5, 1.5)),
    y: Optional[np.ndarray] = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, Any]:
    """
    Probe robustness to contrast variations (scanner/protocol differences).
    
    Args:
        adapter: Model adapter
        x: MRI data
        scale_grid: Contrast scale ranges (min, max)
        y: Optional labels
        rng: Random number generator
    
    Returns:
        Dict with similarity curves
    """
    rng = np.random.default_rng() if rng is None else rng
    
    base_logits = adapter.get_logits(x)
    base_emb = adapter.get_embeddings(x)
    
    logit_means, logit_stds = [], []
    emb_means, emb_stds = [], []
    grid_labels = []
    
    for scale_range in scale_grid:
        x_aug = contrast_variation(x, scale_range, rng)
        grid_labels.append(f"{scale_range[0]}-{scale_range[1]}")
        
        logits_aug = adapter.get_logits(x_aug)
        m, s = _summarize_cosine_series(base_logits, logits_aug)
        logit_means.append(m)
        logit_stds.append(s)
        
        emb_aug = adapter.get_embeddings(x_aug)
        em, es = _summarize_cosine_series(base_emb, emb_aug)
        emb_means.append(em)
        emb_stds.append(es)
    
    return {
        "probe_type": "contrast_variation",
        "modality": "mri",
        "grid": grid_labels,
        "logit": {
            "mean": logit_means,
            "std": logit_stds,
            "sim_mean": float(np.mean(logit_means)),
        },
        "embedding": {
            "mean": emb_means,
            "std": emb_stds,
            "sim_mean": float(np.mean(emb_means)),
        },
    }


# ============================================================================
# Robustness Runner
# ============================================================================

class RobustnessRunner:
    """
    Runner for robustness evaluation of foundation models.
    
    Runs a suite of robustness probes and aggregates results
    into a standardized metrics dict for leaderboard integration.
    
    Supports multiple modalities:
    - Genomics (2D): gene expression, sequences
    - Time-series (3D): EEG, fMRI
    - Volumetric (4D): structural MRI
    """
    
    # Probes available for each modality
    MODALITY_PROBES = {
        "genomics": ["dropout", "noise", "permutation", "masking", "expression"],
        "timeseries": ["dropout", "noise", "line_noise", "permutation", "shift"],
        "volumetric": ["dropout", "noise", "permutation", "intensity", "contrast"],
        "unknown": ["dropout", "noise", "permutation"],
    }
    
    # Default probes (universal)
    DEFAULT_PROBES = ["dropout", "noise", "permutation"]
    
    def __init__(
        self,
        model: Any,
        data_dir: str,
        probes: Optional[List[str]] = None,
        modality: Optional[str] = None,
        fs: int = 200,
        seed: int = 42,
    ):
        """
        Args:
            model: Model instance (must have predict/predict_proba)
            data_dir: Directory containing X.npy and y.npy
            probes: List of probe names to run (default: auto-detect from modality)
            modality: Force modality ('genomics', 'timeseries', 'volumetric') or auto-detect
            fs: Sampling frequency for time-series data
            seed: Random seed for reproducibility
        """
        self.model = model
        self.data_dir = data_dir
        self._user_probes = probes
        self.modality = modality
        self.fs = fs
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        
        self.X = None
        self.y = None
        self.adapter = None
        self.probes = None  # Set after loading data
    
    def load_data(self):
        """Load data from the data directory."""
        self.X = np.load(os.path.join(self.data_dir, "X.npy"))
        self.y = np.load(os.path.join(self.data_dir, "y.npy"))
        
        # Auto-detect modality if not specified
        if self.modality is None:
            self.modality = detect_modality(self.X)
        
        # Set probes based on modality or user preference
        if self._user_probes:
            self.probes = self._user_probes
        else:
            self.probes = self.MODALITY_PROBES.get(self.modality, self.DEFAULT_PROBES)
        
        # Determine if we need to flatten for sklearn-style models
        flatten = not hasattr(self.model, "forward")
        self.adapter = RobustnessAdapter(self.model, flatten_input=flatten)
    
    def run(self) -> Dict[str, Any]:
        """
        Run all configured probes and return aggregated metrics.
        
        Returns:
            Dict with per-probe results and aggregate robustness scores
        """
        if self.X is None:
            self.load_data()
        
        results: Dict[str, Any] = {
            "modality": self.modality,
            "data_shape": list(self.X.shape),
            "sampling_freq": self.fs,
            "seed": self.seed,
            "probes": {},
        }
        
        # Run each probe - includes universal and modality-specific probes
        probe_funcs = {
            # Universal probes (all modalities)
            "dropout": lambda: dropout_probe(self.adapter, self.X, y=self.y, rng=self.rng),
            "noise": lambda: noise_probe(self.adapter, self.X, y=self.y, rng=self.rng),
            "permutation": lambda: permutation_probe(
                self.adapter, self.X, y=self.y, rng=self.rng
            ),
            
            # Time-series probes (EEG, fMRI)
            "line_noise": lambda: line_noise_probe(
                self.adapter, self.X, fs=self.fs, y=self.y, rng=self.rng
            ),
            "shift": lambda: shift_probe(self.adapter, self.X, y=self.y),
            
            # Genomics probes
            "masking": lambda: masking_probe(self.adapter, self.X, y=self.y, rng=self.rng),
            "expression": lambda: expression_probe(self.adapter, self.X, y=self.y, rng=self.rng),
            
            # MRI probes
            "intensity": lambda: intensity_probe(self.adapter, self.X, y=self.y, rng=self.rng),
            "contrast": lambda: contrast_probe(self.adapter, self.X, y=self.y, rng=self.rng),
        }
        
        print(f"Running {len(self.probes)} probes for {self.modality} data: {self.probes}")
        
        for probe_name in self.probes:
            if probe_name in probe_funcs:
                try:
                    results["probes"][probe_name] = probe_funcs[probe_name]()
                except Exception as e:
                    results["probes"][probe_name] = {"error": str(e)}
                    print(f"Warning: Probe {probe_name} failed: {e}", file=sys.stderr)
            else:
                print(f"Warning: Unknown probe '{probe_name}', skipping.", file=sys.stderr)
        
        # Compute aggregate robustness scores
        results["aggregate"] = self._compute_aggregate_scores(results["probes"])
        
        return results
    
    def _compute_aggregate_scores(self, probes: Dict[str, Any]) -> Dict[str, float]:
        """Compute summary robustness scores from probe results."""
        agg: Dict[str, float] = {}
        
        # Collect rAUC scores where available
        raucs = []
        
        for name, probe_result in probes.items():
            if "error" in probe_result:
                continue
            
            # Get rAUC from logit results
            if "logit" in probe_result and "rAUC" in probe_result["logit"]:
                rauc = probe_result["logit"]["rAUC"]
                agg[f"{name}_rAUC"] = round(rauc, 4)
                raucs.append(rauc)
            
            # Get mean similarity for equivariance probes
            if name == "permutation" and "logit" in probe_result:
                agg["perm_equivariance"] = round(probe_result["logit"].get("sim_mean", 0), 4)
            if name == "shift" and "logit" in probe_result:
                agg["shift_sensitivity"] = round(probe_result["logit"].get("sim_mean", 0), 4)
        
        # Overall robustness score (mean of rAUCs)
        if raucs:
            agg["robustness_score"] = round(float(np.mean(raucs)), 4)
        
        return agg


# ============================================================================
# Integration with fmbench runners
# ============================================================================

def get_robustness_runner(
    model: Any,
    data_dir: str,
    **kwargs,
) -> RobustnessRunner:
    """Factory function for robustness runner."""
    return RobustnessRunner(model, data_dir, **kwargs)


__all__ = [
    # Runner
    "RobustnessRunner",
    "RobustnessAdapter",
    "get_robustness_runner",
    
    # Universal probes
    "dropout_probe",
    "noise_probe",
    "permutation_probe",
    
    # Time-series probes
    "line_noise_probe",
    "shift_probe",
    
    # Genomics probes
    "masking_probe",
    "expression_probe",
    
    # MRI probes
    "intensity_probe",
    "contrast_probe",
    
    # Perturbation functions
    "feature_dropout",
    "channel_dropout",  # alias
    "add_gaussian_noise",
    "add_line_noise",
    "feature_permute",
    "channel_permute",  # alias
    "temporal_shift",
    "sequence_mask",
    "expression_noise",
    "intensity_shift",
    "contrast_variation",
    
    # Metrics
    "cosine_similarity",
    "r_auc",
    "detect_modality",
]

