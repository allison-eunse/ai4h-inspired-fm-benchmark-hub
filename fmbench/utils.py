import os
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification, make_regression

def generate_classification_data(
    output_dir: str,
    n_samples: int = 200,
    n_features: int = 20,
    n_classes: int = 2,
    n_informative: int = 10,
    seed: int = 42,
):
    """Generate synthetic classification data (X.npy, y.npy, metadata.csv)."""
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_classes=n_classes,
        random_state=seed
    )
    
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    
    # Generate fake metadata with realistic stratification columns
    rng = np.random.default_rng(seed)
    ids = [f"SUBJ_{i:04d}" for i in range(n_samples)]
    ages = rng.integers(20, 85, size=n_samples)
    sexes = rng.choice(["M", "F"], size=n_samples)
    sites = rng.choice(["SiteA", "SiteB", "SiteC"], size=n_samples)
    scanners = rng.choice(["Siemens", "GE", "Philips"], size=n_samples)
    disease_stages = rng.choice(["CN", "MCI", "AD"], size=n_samples)
    ethnicities = rng.choice(
        ["White", "Black", "Asian", "Hispanic", "Other"], 
        size=n_samples,
        p=[0.4, 0.2, 0.15, 0.2, 0.05]  # Weighted distribution
    )
    
    df = pd.DataFrame({
        "subject_id": ids,
        "age": ages,
        "sex": sexes,
        "site": sites,
        "scanner": scanners,
        "disease_stage": disease_stages,
        "ethnicity": ethnicities,
        "diagnosis": y  # For reference/stratification
    })
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Generated classification data in {output_dir}: {X.shape}")

def generate_regression_data(
    output_dir: str,
    n_samples: int = 200,
    n_features: int = 50,
    n_targets: int = 1,
    seed: int = 42
):
    """Generate synthetic regression data (X.npy, y.npy, metadata.csv)."""
    os.makedirs(output_dir, exist_ok=True)
    
    X, y = make_regression(
        n_samples=n_samples,
        n_features=n_features,
        n_targets=n_targets,
        noise=0.1,
        random_state=seed
    )
    
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    
    # Metadata
    ids = [f"SUBJ_{i:04d}" for i in range(n_samples)]
    ages = np.random.randint(18, 70, size=n_samples)
    df = pd.DataFrame({
        "subject_id": ids,
        "age": ages,
    })
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"Generated regression data in {output_dir}: {X.shape}")

def generate_timeseries_data(
    output_dir: str,
    n_samples: int = 100,
    n_channels: int = 16,
    n_timepoints: int = 500,
    n_classes: int = 2,
    fs: int = 200,
    seed: int = 42,
):
    """
    Generate synthetic time-series data for robustness testing.
    
    Creates 3D tensors (N, C, T) suitable for brainaug-lab probes.
    Simulates EEG-like signals with class-dependent frequency content.
    """
    os.makedirs(output_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    
    # Generate class labels
    y = rng.integers(0, n_classes, size=n_samples)
    
    # Create time axis
    t = np.arange(n_timepoints) / fs
    
    # Generate synthetic signals with class-dependent characteristics
    X = np.zeros((n_samples, n_channels, n_timepoints), dtype=np.float32)
    
    for i in range(n_samples):
        label = y[i]
        for c in range(n_channels):
            # Base signal: mixture of frequencies that vary by class
            base_freq = 8 + label * 4  # Class 0: alpha (8-12Hz), Class 1: beta (12-16Hz)
            signal = np.sin(2 * np.pi * base_freq * t + rng.uniform(0, 2*np.pi))
            
            # Add harmonics
            signal += 0.5 * np.sin(2 * np.pi * (base_freq * 2) * t + rng.uniform(0, 2*np.pi))
            
            # Add channel-specific variation
            signal += 0.3 * np.sin(2 * np.pi * (c + 1) * t + rng.uniform(0, 2*np.pi))
            
            # Add noise
            signal += rng.normal(0, 0.2, size=n_timepoints)
            
            X[i, c, :] = signal.astype(np.float32)
    
    # Normalize
    X = (X - X.mean()) / (X.std() + 1e-8)
    
    np.save(os.path.join(output_dir, "X.npy"), X)
    np.save(os.path.join(output_dir, "y.npy"), y)
    
    # Metadata
    ids = [f"SUBJ_{i:04d}" for i in range(n_samples)]
    ages = rng.integers(20, 75, size=n_samples)
    sexes = rng.choice(["M", "F"], size=n_samples)
    sites = rng.choice(["SiteA", "SiteB"], size=n_samples)
    
    df = pd.DataFrame({
        "subject_id": ids,
        "age": ages,
        "sex": sexes,
        "site": sites,
        "label": y,
    })
    df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    
    # Write sampling info
    info = {
        "n_samples": n_samples,
        "n_channels": n_channels,
        "n_timepoints": n_timepoints,
        "n_classes": n_classes,
        "sampling_freq": fs,
        "seed": seed,
    }
    import json
    with open(os.path.join(output_dir, "info.json"), "w") as f:
        json.dump(info, f, indent=2)
    
    print(f"Generated time-series data in {output_dir}: {X.shape} @ {fs}Hz")


def generate_all_toy_data(root_dir: str = "toy_data"):
    """Generate all standard toy datasets."""
    
    # 1. Neuro / fMRI Classification
    generate_classification_data(
        os.path.join(root_dir, "neuro/fmri_classification"),
        n_features=100,  # Simulating ROIs
        n_classes=2
    )
    
    # 2. Genomics / Gene Expression Classification
    generate_classification_data(
        os.path.join(root_dir, "genomics/gene_expression_classification"),
        n_features=50,   # Simulating Genes
        n_classes=3,     # Multi-class
        n_informative=20
    )
    
    # 3. Neuro / Regression (e.g. age prediction or reconstruction)
    generate_regression_data(
        os.path.join(root_dir, "neuro/regression"),
        n_features=100,
        n_targets=100  # Multi-output (e.g. reconstruction)
    )
    
    # 4. Neuro / Robustness Testing (3D time-series)
    generate_timeseries_data(
        os.path.join(root_dir, "neuro/robustness"),
        n_samples=100,
        n_channels=16,
        n_timepoints=500,
        n_classes=2,
        fs=200
    )
    
    # 5. Neuro / EEG-like time-series classification
    generate_timeseries_data(
        os.path.join(root_dir, "neuro/eeg_classification"),
        n_samples=150,
        n_channels=32,
        n_timepoints=1000,
        n_classes=3,
        fs=256
    )

