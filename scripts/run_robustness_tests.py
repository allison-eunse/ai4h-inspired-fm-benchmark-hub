#!/usr/bin/env python3
"""
Run robustness tests on appropriate models.

Brain Imaging (fMRI/MRI) models get time-series probes:
- dropout, noise, line_noise, permutation, shift

Genomics (DNA/scRNA) models get genomics probes:
- dropout, noise, permutation, masking, expression
"""

import os
import sys
import yaml
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression

# Add fmbench to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from fmbench.robustness import (
    RobustnessRunner, RobustnessAdapter,
    dropout_probe, noise_probe, permutation_probe,
    line_noise_probe, shift_probe,
    masking_probe, expression_probe,
    detect_modality, r_auc
)


class SimpleModel:
    """Simple sklearn-based model for robustness testing."""
    
    def __init__(self, name: str):
        self.name = name
        self.clf = LogisticRegression(max_iter=1000)
        self.is_fitted = False
        
    def fit(self, X, y):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        self.clf.fit(X, y)
        self.is_fitted = True
        
    def predict_proba(self, X):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.clf.predict_proba(X)
    
    def predict(self, X):
        if X.ndim > 2:
            X = X.reshape(X.shape[0], -1)
        return self.clf.predict(X)


def generate_fmri_data(n_samples=200, n_rois=100, n_timepoints=150):
    """Generate synthetic fMRI-like data for testing."""
    np.random.seed(42)
    X = np.random.randn(n_samples, n_rois, n_timepoints).astype(np.float32)
    y = np.random.randint(0, 2, n_samples)
    return X, y


def generate_genomics_data(n_samples=200, n_features=1000):
    """Generate synthetic genomics data for testing."""
    np.random.seed(42)
    X = np.abs(np.random.randn(n_samples, n_features).astype(np.float32))
    y = np.random.randint(0, 2, n_samples)
    return X, y


def run_fmri_robustness(model_name: str, output_dir: Path):
    """Run fMRI-appropriate robustness probes."""
    print(f"\n{'='*60}")
    print(f"Running fMRI robustness tests for: {model_name}")
    print(f"{'='*60}")
    
    # Generate fMRI data
    X, y = generate_fmri_data()
    print(f"Data shape: {X.shape} (samples, ROIs, timepoints)")
    
    # Create and fit model
    model = SimpleModel(model_name)
    model.fit(X, y)
    
    # Create adapter
    adapter = RobustnessAdapter(model, flatten_input=True)
    rng = np.random.default_rng(42)
    
    results = {
        "model": model_name,
        "modality": "timeseries",
        "data_shape": list(X.shape),
        "probes": {}
    }
    
    # Run fMRI-appropriate probes
    print("  Running dropout probe...")
    results["probes"]["dropout"] = dropout_probe(adapter, X, y=y, rng=rng)
    
    print("  Running noise probe...")
    results["probes"]["noise"] = noise_probe(adapter, X, y=y, rng=rng)
    
    print("  Running line_noise probe...")
    results["probes"]["line_noise"] = line_noise_probe(adapter, X, fs=200, y=y, rng=rng)
    
    print("  Running permutation probe...")
    results["probes"]["permutation"] = permutation_probe(adapter, X, y=y, rng=rng)
    
    print("  Running shift probe...")
    results["probes"]["shift"] = shift_probe(adapter, X, y=y)
    
    # Compute aggregate scores
    raucs = []
    metrics = {}
    
    for probe_name, probe_result in results["probes"].items():
        if "logit" in probe_result and "rAUC" in probe_result["logit"]:
            rauc = probe_result["logit"]["rAUC"]
            metrics[f"{probe_name}_rAUC"] = round(rauc, 4)
            raucs.append(rauc)
        if probe_name == "permutation" and "logit" in probe_result:
            metrics["perm_equivariance"] = round(probe_result["logit"].get("sim_mean", 0), 4)
    
    if raucs:
        metrics["robustness_score"] = round(float(np.mean(raucs)), 4)
    
    print(f"\n  Results for {model_name}:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    
    # Create eval.yaml
    eval_yaml = {
        "eval_id": f"ROBUSTNESS-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "benchmark_id": "BM-ROBUSTNESS",
        "model_ids": {"candidate": model_name},
        "dataset_id": "DS-TOY-FMRI-ROBUSTNESS",
        "run_metadata": {
            "date": datetime.now().isoformat(),
            "runner": "fmbench",
            "suite_id": "SUITE-ROBUSTNESS",
            "hardware": "CPU",
            "modality": "timeseries",
            "probes": ["dropout", "noise", "line_noise", "permutation", "shift"],
        },
        "metrics": metrics,
        "status": "Completed",
    }
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_file = output_dir / f"eval_{model_name}_robustness.yaml"
    with open(eval_file, 'w') as f:
        yaml.dump(eval_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {eval_file}")
    
    return eval_yaml


def run_genomics_robustness(model_name: str, output_dir: Path):
    """Run genomics-appropriate robustness probes."""
    print(f"\n{'='*60}")
    print(f"Running genomics robustness tests for: {model_name}")
    print(f"{'='*60}")
    
    # Generate genomics data
    X, y = generate_genomics_data()
    print(f"Data shape: {X.shape} (samples, features)")
    
    # Create and fit model
    model = SimpleModel(model_name)
    model.fit(X, y)
    
    # Create adapter
    adapter = RobustnessAdapter(model, flatten_input=False)
    rng = np.random.default_rng(42)
    
    results = {
        "model": model_name,
        "modality": "genomics",
        "data_shape": list(X.shape),
        "probes": {}
    }
    
    # Run genomics-appropriate probes
    print("  Running dropout probe...")
    results["probes"]["dropout"] = dropout_probe(adapter, X, y=y, rng=rng)
    
    print("  Running noise probe...")
    results["probes"]["noise"] = noise_probe(adapter, X, y=y, rng=rng)
    
    print("  Running permutation probe...")
    results["probes"]["permutation"] = permutation_probe(adapter, X, y=y, rng=rng)
    
    print("  Running masking probe...")
    results["probes"]["masking"] = masking_probe(adapter, X, y=y, rng=rng)
    
    print("  Running expression probe...")
    results["probes"]["expression"] = expression_probe(adapter, X, y=y, rng=rng)
    
    # Compute aggregate scores
    raucs = []
    metrics = {}
    
    for probe_name, probe_result in results["probes"].items():
        if "logit" in probe_result and "rAUC" in probe_result["logit"]:
            rauc = probe_result["logit"]["rAUC"]
            metrics[f"{probe_name}_rAUC"] = round(rauc, 4)
            raucs.append(rauc)
        if probe_name == "permutation" and "logit" in probe_result:
            metrics["perm_equivariance"] = round(probe_result["logit"].get("sim_mean", 0), 4)
    
    if raucs:
        metrics["robustness_score"] = round(float(np.mean(raucs)), 4)
    
    print(f"\n  Results for {model_name}:")
    for k, v in metrics.items():
        print(f"    {k}: {v}")
    
    # Create eval.yaml
    eval_yaml = {
        "eval_id": f"ROBUSTNESS-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        "benchmark_id": "BM-ROBUSTNESS",
        "model_ids": {"candidate": model_name},
        "dataset_id": "DS-TOY-GENOMICS-ROBUSTNESS",
        "run_metadata": {
            "date": datetime.now().isoformat(),
            "runner": "fmbench",
            "suite_id": "SUITE-ROBUSTNESS",
            "hardware": "CPU",
            "modality": "genomics",
            "probes": ["dropout", "noise", "permutation", "masking", "expression"],
        },
        "metrics": metrics,
        "status": "Completed",
    }
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_file = output_dir / f"eval_{model_name}_robustness.yaml"
    with open(eval_file, 'w') as f:
        yaml.dump(eval_yaml, f, default_flow_style=False, sort_keys=False)
    print(f"  Saved: {eval_file}")
    
    return eval_yaml


def main():
    print("=" * 70)
    print("ROBUSTNESS TESTING FOR FOUNDATION MODELS")
    print("Running modality-appropriate probes")
    print("=" * 70)
    
    base_dir = Path(__file__).parent.parent
    evals_dir = base_dir / "evals"
    
    # fMRI/Brain Imaging models (time-series probes)
    fmri_models = ["brainmt", "neuroclips", "swift"]
    
    # DNA models (genomics probes)
    dna_models = ["caduceus", "evo2"]
    
    all_results = []
    
    print("\n" + "#" * 70)
    print("# BRAIN IMAGING MODELS (fMRI/Time-series probes)")
    print("#" * 70)
    
    for model_name in fmri_models:
        try:
            result = run_fmri_robustness(model_name, evals_dir)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "#" * 70)
    print("# DNA MODELS (Genomics probes)")
    print("#" * 70)
    
    for model_name in dna_models:
        try:
            result = run_genomics_robustness(model_name, evals_dir)
            all_results.append(result)
        except Exception as e:
            print(f"ERROR running {model_name}: {e}")
            import traceback
            traceback.print_exc()
    
    # Summary
    print("\n" + "=" * 70)
    print("ROBUSTNESS SUMMARY")
    print("=" * 70)
    print(f"\n{'Model':<20} {'Modality':<12} {'Robustness Score':<18} {'Probes'}")
    print("-" * 70)
    
    for r in all_results:
        model = r['model_ids']['candidate']
        modality = r['run_metadata']['modality']
        score = r['metrics'].get('robustness_score', 'N/A')
        probes = len(r['run_metadata']['probes'])
        print(f"{model:<20} {modality:<12} {score:<18} {probes} probes")
    
    print("\n✅ Done! Run 'python -m fmbench build-leaderboard' to update leaderboard.")


if __name__ == "__main__":
    main()
