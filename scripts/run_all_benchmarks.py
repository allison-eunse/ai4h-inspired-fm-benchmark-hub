#!/usr/bin/env python3
"""
Run All Benchmarks and Update Leaderboard

This script:
1. Runs each model adapter against benchmarks
2. Records results to evals/
3. Rebuilds the leaderboard with updated rankings and explanations
"""

import os
import sys
import yaml
import json
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from fmbench.model_adapters import ADAPTER_REGISTRY, get_adapter
from fmbench.leaderboard import build_leaderboard


# =============================================================================
# BENCHMARK CONFIGURATIONS
# =============================================================================

# Map models to their appropriate benchmarks
MODEL_BENCHMARK_MAP = {
    # Neurology models → Neuro benchmarks
    "brainlm": ["BM-001", "BM-BRAIN-MAE", "BM-FMRI-GRANULAR", "robustness_testing"],
    "brainjepa": ["BM-001", "BM-FMRI-GRANULAR", "robustness_testing"],
    "brainharmony": ["robustness_testing"],
    "brainmt": ["BM-FMRI-GRANULAR"],
    "neuroclip": ["BM-FMRI-GRANULAR"],
    
    # Genomics models → Genomics benchmarks  
    "geneformer": ["BM-002", "robustness_testing"],
    "hyenadna": ["BM-002", "robustness_testing"],
    "caduceus": ["BM-002", "robustness_testing"],
    "dnabert2": ["BM-002"],
    "evo2": ["BM-002"],
    "swift": ["BM-002"],
    
    # Vision-Language models → Report generation
    "openflamingo": ["BM-REPORT-GEN"],
    "medflamingo": ["BM-REPORT-GEN"],
    "titan": ["BM-REPORT-GEN"],
    "uni": ["BM-001"],  # Pathology classification
    "radbert": ["BM-REPORT-GEN"],
    "m3fm": ["BM-REPORT-GEN"],
    "me_llama": ["BM-REPORT-GEN"],
}

# Benchmark metadata for explanations
BENCHMARK_INFO = {
    "BM-001": {
        "name": "AD Classification",
        "primary_metric": "AUROC",
        "description": "Alzheimer's Disease classification from brain MRI"
    },
    "BM-002": {
        "name": "Cell Type Annotation", 
        "primary_metric": "Accuracy",
        "description": "Single-cell RNA-seq cell type classification"
    },
    "BM-BRAIN-MAE": {
        "name": "Brain Time-Series",
        "primary_metric": "Correlation",
        "description": "fMRI reconstruction and masked autoencoding"
    },
    "BM-FMRI-GRANULAR": {
        "name": "fMRI Granular",
        "primary_metric": "AUROC",
        "description": "Detailed fMRI analysis across scanners/sites"
    },
    "BM-REPORT-GEN": {
        "name": "Report Generation",
        "primary_metric": "report_quality_score",
        "description": "Clinical report generation quality"
    },
    "robustness_testing": {
        "name": "Robustness",
        "primary_metric": "robustness_score",
        "description": "Model stability under perturbations"
    },
}


def run_model_benchmark(
    model_name: str,
    benchmark_id: str,
    data_dir: str,
    output_dir: str,
) -> Dict[str, Any]:
    """
    Run a single model on a benchmark and return metrics.
    """
    print(f"\n{'='*60}")
    print(f"Running {model_name} on {benchmark_id}")
    print(f"{'='*60}")
    
    try:
        # Get adapter
        adapter = get_adapter(model_name)
        adapter.load()
        
        # Load test data
        import numpy as np
        X_path = os.path.join(data_dir, "X.npy")
        y_path = os.path.join(data_dir, "y.npy")
        
        if not os.path.exists(X_path):
            print(f"  ⚠️  No data found at {data_dir}")
            return None
            
        X = np.load(X_path)
        y = np.load(y_path)
        
        print(f"  📊 Data shape: X={X.shape}, y={y.shape}")
        
        # Run predictions
        if hasattr(adapter, 'predict_proba'):
            y_prob = adapter.predict_proba(X)
            y_pred = np.argmax(y_prob, axis=1) if y_prob.ndim == 2 else (y_prob > 0.5).astype(int)
        else:
            y_pred = adapter.predict(X)
            y_prob = None
            
        # Calculate metrics
        from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
        
        metrics = {
            "Accuracy": round(accuracy_score(y, y_pred), 4),
            "F1-Score": round(f1_score(y, y_pred, average="macro", zero_division=0), 4),
        }
        
        if y_prob is not None:
            try:
                if y_prob.ndim == 2:
                    auroc = roc_auc_score(y, y_prob[:, 1] if y_prob.shape[1] == 2 else y_prob, multi_class="ovr")
                else:
                    auroc = roc_auc_score(y, y_prob)
                metrics["AUROC"] = round(auroc, 4)
            except:
                pass
                
        print(f"  ✅ Metrics: {metrics}")
        return metrics
        
    except Exception as e:
        print(f"  ❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


def generate_eval_yaml(
    model_name: str,
    benchmark_id: str,
    metrics: Dict[str, Any],
    dataset_id: str = "toy_data",
) -> str:
    """Generate evaluation YAML content."""
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    eval_id = f"{benchmark_id}-{model_name}-{timestamp}"
    
    eval_record = {
        "eval_id": eval_id,
        "benchmark_id": benchmark_id,
        "model_ids": {
            "candidate": model_name
        },
        "dataset_id": dataset_id,
        "run_metadata": {
            "date": datetime.now().strftime("%Y-%m-%d"),
            "runner": "fmbench_auto",
            "hardware": "auto-detected",
        },
        "metrics": metrics,
        "status": "Completed"
    }
    
    return yaml.dump(eval_record, default_flow_style=False, sort_keys=False)


def generate_ranking_explanation(
    benchmark_id: str,
    results: List[Dict],
) -> str:
    """Generate human-readable explanation for rankings."""
    if not results:
        return ""
        
    info = BENCHMARK_INFO.get(benchmark_id, {})
    primary = info.get("primary_metric", "score")
    
    # Sort by primary metric
    sorted_results = sorted(
        results, 
        key=lambda x: x.get("metrics", {}).get(primary, 0),
        reverse=True
    )
    
    explanation = f"\n## Ranking Analysis for {info.get('name', benchmark_id)}\n\n"
    explanation += f"**Task:** {info.get('description', 'N/A')}\n"
    explanation += f"**Primary Metric:** `{primary}`\n\n"
    
    if len(sorted_results) >= 1:
        top = sorted_results[0]
        explanation += f"### 🥇 Top Performer: {top['model']}\n"
        explanation += f"- Score: {top['metrics'].get(primary, 'N/A')}\n"
        
        if len(sorted_results) >= 2:
            second = sorted_results[1]
            gap = top['metrics'].get(primary, 0) - second['metrics'].get(primary, 0)
            explanation += f"- Lead over #{2}: +{gap:.4f}\n"
            
    explanation += "\n### Full Rankings\n\n"
    explanation += "| Rank | Model | Score | Notes |\n"
    explanation += "|:---:|:---|:---:|:---|\n"
    
    for i, r in enumerate(sorted_results, 1):
        score = r['metrics'].get(primary, 'N/A')
        tier = "⭐" if score >= 0.9 else "✅" if score >= 0.8 else "🔶" if score >= 0.7 else "📈"
        explanation += f"| {i} | {r['model']} | {score} | {tier} |\n"
        
    return explanation


def main():
    """Main entry point."""
    print("🚀 FM Benchmark Hub - Auto Evaluation Runner")
    print("=" * 60)
    
    # Paths
    evals_dir = PROJECT_ROOT / "evals"
    toy_data_dir = PROJECT_ROOT / "toy_data"
    results_dir = PROJECT_ROOT / "results" / "auto_run"
    
    os.makedirs(evals_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Track all results by benchmark
    all_results: Dict[str, List[Dict]] = {}
    
    # Run each model
    for model_name, benchmarks in MODEL_BENCHMARK_MAP.items():
        print(f"\n📦 Model: {model_name}")
        
        # Check if adapter exists
        if model_name not in ADAPTER_REGISTRY:
            print(f"  ⚠️  No adapter for {model_name}, skipping...")
            continue
            
        for benchmark_id in benchmarks:
            # Determine data directory
            if benchmark_id == "robustness_testing":
                data_dir = toy_data_dir / "neuro" / "robustness"
            elif benchmark_id in ["BM-002"]:
                data_dir = toy_data_dir / "genomics" / "cell_type"
            else:
                data_dir = toy_data_dir / "fmri" / "classification"
                
            # Create toy data if missing
            if not data_dir.exists():
                data_dir = toy_data_dir / "fmri" / "classification"
                
            if not data_dir.exists():
                print(f"  ⚠️  No data for {benchmark_id}, generating toy data...")
                subprocess.run([
                    sys.executable, "-m", "fmbench", "generate-toy-data",
                    "--output-dir", str(toy_data_dir)
                ], cwd=PROJECT_ROOT)
                
            # Run benchmark
            metrics = run_model_benchmark(
                model_name=model_name,
                benchmark_id=benchmark_id,
                data_dir=str(data_dir),
                output_dir=str(results_dir / model_name / benchmark_id),
            )
            
            if metrics:
                # Save eval YAML
                eval_yaml = generate_eval_yaml(model_name, benchmark_id, metrics)
                timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
                eval_path = evals_dir / f"auto-{benchmark_id}-{model_name}-{timestamp}.yaml"
                
                with open(eval_path, "w") as f:
                    f.write(eval_yaml)
                print(f"  💾 Saved: {eval_path.name}")
                
                # Track for ranking
                if benchmark_id not in all_results:
                    all_results[benchmark_id] = []
                all_results[benchmark_id].append({
                    "model": model_name,
                    "metrics": metrics
                })
    
    # Generate ranking explanations
    print("\n" + "=" * 60)
    print("📊 Generating Ranking Explanations")
    print("=" * 60)
    
    explanations_path = results_dir / "ranking_explanations.md"
    with open(explanations_path, "w") as f:
        f.write("# Auto-Generated Ranking Explanations\n\n")
        f.write(f"*Generated: {datetime.now().isoformat()}*\n\n")
        
        for benchmark_id, results in all_results.items():
            explanation = generate_ranking_explanation(benchmark_id, results)
            f.write(explanation)
            f.write("\n---\n")
            
    print(f"  📝 Saved explanations: {explanations_path}")
    
    # Rebuild leaderboard
    print("\n" + "=" * 60)
    print("🏆 Rebuilding Leaderboard")
    print("=" * 60)
    
    build_leaderboard()
    
    print("\n✅ All benchmarks complete!")
    print(f"   Results: {results_dir}")
    print(f"   Evals: {evals_dir}")
    print("   Leaderboard: docs/leaderboards/index.md")


if __name__ == "__main__":
    main()

