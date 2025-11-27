import argparse
import glob
import os
import shutil
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional

import yaml

from .leaderboard import build_leaderboard
from .utils import generate_all_toy_data
from .models import load_model_from_config
from .runners import get_runner


def _load_yaml(path: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    # Attach filename for easier debugging
    data.setdefault("_filename", os.path.basename(path))
    return data


def _find_suite_by_id(tests_dir: str, suite_id: str) -> Optional[str]:
    for path in glob.glob(os.path.join(tests_dir, "*.yaml")):
        data = _load_yaml(path)
        if data.get("suite_id") == suite_id:
            return path
    return None


def _find_benchmark_by_id(benchmarks_dir: str, benchmark_id: str) -> Optional[str]:
    for path in glob.glob(os.path.join(benchmarks_dir, "*.yaml")):
        data = _load_yaml(path)
        if data.get("benchmark_id") == benchmark_id:
            return path
    return None


def _find_dataset_path(datasets_dir: str, dataset_id: str) -> Optional[str]:
    """
    Find the physical path for a dataset.
    Returns the path specified in the dataset YAML if found.
    """
    for path in glob.glob(os.path.join(datasets_dir, "*.yaml")):
        data = _load_yaml(path)
        if data.get("dataset_id") == dataset_id:
            return data.get("path")
    return None


def cmd_list_suites(args: argparse.Namespace) -> None:
    tests_dir = args.tests_dir
    paths = sorted(glob.glob(os.path.join(tests_dir, "*.yaml")))

    suites: List[Dict[str, Any]] = []
    for path in paths:
        try:
            data = _load_yaml(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            continue
        if "suite_id" in data:
            suites.append(
                {
                    "suite_id": data.get("suite_id"),
                    "name": data.get("name", ""),
                    "benchmark_id": data.get("benchmark_id", ""),
                    "file": os.path.basename(path),
                }
            )

    if not suites:
        print(f"No suites found in {tests_dir!r}.")
        return

    print(f"Suites in {tests_dir}:")
    print(f"{'SUITE_ID':24} {'BENCHMARK_ID':16} NAME")
    print("-" * 72)
    for s in suites:
        print(
            f"{s['suite_id']:<24} {s['benchmark_id']:<16} {s['name']} "
            f"({s['file']})"
        )


def cmd_list_benchmarks(args: argparse.Namespace) -> None:
    benchmarks_dir = args.benchmarks_dir
    paths = sorted(glob.glob(os.path.join(benchmarks_dir, "*.yaml")))

    benchmarks: List[Dict[str, Any]] = []
    for path in paths:
        try:
            data = _load_yaml(path)
        except Exception as exc:  # pragma: no cover - defensive
            print(f"Error reading {path}: {exc}", file=sys.stderr)
            continue
        if "benchmark_id" in data:
            benchmarks.append(
                {
                    "benchmark_id": data.get("benchmark_id"),
                    "name": data.get("name", ""),
                    "health_domain": data.get("health_domain", ""),
                    "ai_task": data.get("ai_task", ""),
                    "file": os.path.basename(path),
                }
            )

    if not benchmarks:
        print(f"No benchmarks found in {benchmarks_dir!r}.")
        return

    print(f"Benchmarks in {benchmarks_dir}:")
    print(f"{'BENCHMARK_ID':20} {'HEALTH_DOMAIN':16} {'AI_TASK':14} NAME")
    print("-" * 96)
    for b in benchmarks:
        print(
            f"{b['benchmark_id']:<20} {b['health_domain']:<16} "
            f"{b['ai_task']:<14} {b['name']} ({b['file']})"
        )


def cmd_run(args: argparse.Namespace) -> None:
    """
    Run a benchmark suite for a specific model configuration.
    
    Steps:
    1. Load suite and model config.
    2. Determine benchmark and runner type.
    3. Determine dataset path.
    4. Instantiate model and runner.
    5. Execute run and compute metrics.
    6. Write eval record and report.
    """
    tests_dir = args.tests_dir
    benchmarks_dir = args.benchmarks_dir
    datasets_dir = args.datasets_dir
    suite_id = args.suite
    model_config_path = args.model
    out_dir = args.out
    evals_dir = args.evals_dir
    reports_dir = args.reports_dir

    # 1. Validate Suite
    suite_path = _find_suite_by_id(tests_dir, suite_id)
    if not suite_path:
        print(f"Error: suite_id {suite_id!r} not found in {tests_dir!r}.", file=sys.stderr)
        sys.exit(1)
    suite = _load_yaml(suite_path)
    benchmark_id = suite.get("benchmark_id")
    if not benchmark_id:
        print(f"Error: suite {suite_id!r} is missing 'benchmark_id'.", file=sys.stderr)
        sys.exit(1)

    # 2. Validate Model Config
    if not os.path.exists(model_config_path):
        print(f"Error: model config {model_config_path!r} does not exist.", file=sys.stderr)
        sys.exit(1)
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f) or {}
    model_id = model_cfg.get("model_id")
    if not model_id:
        print(f"Error: model config must contain 'model_id'.", file=sys.stderr)
        sys.exit(1)

    # 3. Determine Runner Type via Benchmark AI Task
    benchmark_path = _find_benchmark_by_id(benchmarks_dir, benchmark_id)
    if not benchmark_path:
        print(f"Error: benchmark_id {benchmark_id!r} not found in {benchmarks_dir!r}.", file=sys.stderr)
        sys.exit(1)
    benchmark = _load_yaml(benchmark_path)
    ai_task = benchmark.get("ai_task", "").lower()
    
    # Simple mapping for MVP
    if "classification" in ai_task:
        runner_type = "classification"
    elif "reconstruction" in ai_task or "regression" in ai_task:
        runner_type = "regression"
    else:
        print(f"Warning: Unknown AI task '{ai_task}', defaulting to 'classification'.")
        runner_type = "classification"

    # 4. Determine Dataset Path
    dataset_id = args.dataset
    if not dataset_id:
        dataset_id = suite.get("inputs", {}).get("dataset", {}).get("default")
    if not dataset_id:
        print("Error: No dataset_id provided and no default in suite.", file=sys.stderr)
        sys.exit(1)
        
    dataset_path_val = _find_dataset_path(datasets_dir, dataset_id)
    if not dataset_path_val or not os.path.exists(dataset_path_val):
        print(f"Error: Dataset path for {dataset_id!r} not found (checked {dataset_path_val}).", file=sys.stderr)
        print("Tip: Run `fmbench generate-toy-data` and ensure datasets/*.yaml points to it.", file=sys.stderr)
        sys.exit(1)

    # 5. Instantiate Model and Runner
    print(f"Loading model from {model_config_path}...")
    try:
        model = load_model_from_config(model_cfg)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Starting {runner_type} run on {dataset_id}...")
    try:
        runner = get_runner(runner_type, model, dataset_path_val)
        metrics = runner.run()
    except Exception as e:
        print(f"Error during execution: {e}", file=sys.stderr)
        sys.exit(1)
        
    print(f"Run complete. Metrics: {metrics}")

    # 6. Write Outputs
    os.makedirs(evals_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    eval_id = f"{suite_id}-{model_id}-{timestamp}"

    eval_record: Dict[str, Any] = {
        "eval_id": eval_id,
        "benchmark_id": benchmark_id,
        "model_ids": {"candidate": model_id},
        "dataset_id": dataset_id,
        "run_metadata": {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "runner": "fmbench",
            "suite_id": suite_id,
            "model_config": os.path.relpath(model_config_path),
            "output_dir": os.path.relpath(out_dir),
        },
        "metrics": metrics,
        "status": "Completed",
    }

    eval_filename = f"{eval_id}.yaml"
    eval_path = os.path.join(evals_dir, eval_filename)
    with open(eval_path, "w") as f:
        yaml.safe_dump(eval_record, f, sort_keys=False)

    # Report
    def format_metrics_report(m: Dict[str, Any]) -> str:
        """Format metrics dict into readable Markdown."""
        lines = []
        
        # Overall metrics (non-dict values)
        overall = [(k, v) for k, v in m.items() if not isinstance(v, dict)]
        if overall:
            lines.append("### Overall")
            for k, v in overall:
                lines.append(f"- **{k}**: {v}")
        
        # Stratified metrics
        stratified = m.get("stratified", {})
        if stratified:
            lines.append("\n### Stratified Performance")
            for strat_key, groups in stratified.items():
                lines.append(f"\n#### By {strat_key.replace('_', ' ').title()}")
                lines.append("")
                # Table header
                sample_metrics = list(next(iter(groups.values())).keys())
                header = "| Group | " + " | ".join(sample_metrics) + " |"
                sep = "|" + "|".join(["---"] * (len(sample_metrics) + 1)) + "|"
                lines.append(header)
                lines.append(sep)
                # Table rows
                for group_name, group_metrics in sorted(groups.items()):
                    vals = [str(group_metrics.get(m, "-")) for m in sample_metrics]
                    lines.append(f"| {group_name} | " + " | ".join(vals) + " |")
        
        return "\n".join(lines)
    
    metrics_report = format_metrics_report(metrics)
    
    report_content = f"""# Evaluation Report: {eval_id}

- **Suite**: {suite_id}
- **Benchmark**: {benchmark_id}
- **Model**: {model_id}
- **Dataset**: {dataset_id}
- **Task**: {ai_task}

## Performance Metrics

{metrics_report}

## Execution Details
- **Date**: {eval_record['run_metadata']['date']}
- **Runner**: fmbench ({runner_type})
"""

    report_filename = f"{eval_id}.md"
    report_path = os.path.join(reports_dir, report_filename)
    with open(report_path, "w") as f:
        f.write(report_content)

    # Mirror outputs
    shutil.copy2(eval_path, os.path.join(out_dir, "eval.yaml"))
    shutil.copy2(report_path, os.path.join(out_dir, "report.md"))

    print("Results written:")
    print(f"- Eval YAML: {eval_path}")
    print(f"- Report:    {report_path}")
    print(f"- Out dir:   {out_dir}")



def cmd_build_leaderboard(args: argparse.Namespace) -> None:
    output_path = build_leaderboard(
        benchmarks_dir=args.benchmarks_dir,
        models_dir=args.models_dir,
        datasets_dir=args.datasets_dir,
        evals_dir=args.evals_dir,
        output_path=args.output,
    )
    print(f"Leaderboard built at {output_path}")


def cmd_generate_toy_data(args: argparse.Namespace) -> None:
    generate_all_toy_data(root_dir=args.output_dir)


def cmd_run_robustness(args: argparse.Namespace) -> None:
    """
    Run robustness evaluation suite for a model.
    
    This tests how well model outputs remain stable under realistic
    perturbations (noise, dropout, artifacts).
    """
    from .robustness import RobustnessRunner
    
    model_config_path = args.model
    data_dir = args.data
    out_dir = args.out
    evals_dir = args.evals_dir
    reports_dir = args.reports_dir
    
    # Load model config
    if not os.path.exists(model_config_path):
        print(f"Error: model config {model_config_path!r} does not exist.", file=sys.stderr)
        sys.exit(1)
    
    with open(model_config_path, "r") as f:
        model_cfg = yaml.safe_load(f) or {}
    model_id = model_cfg.get("model_id")
    if not model_id:
        print("Error: model config must contain 'model_id'.", file=sys.stderr)
        sys.exit(1)
    
    # Check data directory
    if not os.path.exists(data_dir):
        print(f"Error: data directory {data_dir!r} does not exist.", file=sys.stderr)
        print("Tip: Run `fmbench generate-toy-data` first.", file=sys.stderr)
        sys.exit(1)
    
    # Load model
    print(f"Loading model from {model_config_path}...")
    try:
        model = load_model_from_config(model_cfg)
    except Exception as e:
        print(f"Error loading model: {e}", file=sys.stderr)
        sys.exit(1)
    
    # Run robustness evaluation
    print("Running robustness evaluation...")
    probes = args.probes.split(",") if args.probes else None
    
    try:
        runner = RobustnessRunner(
            model=model,
            data_dir=data_dir,
            probes=probes,
            fs=args.sampling_freq,
            seed=args.seed,
        )
        metrics = runner.run()
    except Exception as e:
        print(f"Error during robustness evaluation: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print(f"Robustness evaluation complete.")
    print(f"Aggregate scores: {metrics.get('aggregate', {})}")
    
    # Write outputs
    os.makedirs(evals_dir, exist_ok=True)
    os.makedirs(reports_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    eval_id = f"ROBUSTNESS-{model_id}-{timestamp}"
    
    eval_record: Dict[str, Any] = {
        "eval_id": eval_id,
        "benchmark_id": "robustness_testing",
        "model_ids": {"candidate": model_id},
        "data_dir": os.path.relpath(data_dir),
        "run_metadata": {
            "date": datetime.utcnow().strftime("%Y-%m-%d"),
            "runner": "fmbench-robustness",
            "model_config": os.path.relpath(model_config_path),
            "output_dir": os.path.relpath(out_dir),
            "sampling_freq": args.sampling_freq,
            "seed": args.seed,
        },
        "metrics": metrics.get("aggregate", {}),
        "probes": metrics.get("probes", {}),
        "status": "Completed",
    }
    
    eval_filename = f"{eval_id}.yaml"
    eval_path = os.path.join(evals_dir, eval_filename)
    with open(eval_path, "w") as f:
        yaml.safe_dump(eval_record, f, sort_keys=False, default_flow_style=False)
    
    # Generate robustness report
    report_content = _generate_robustness_report(eval_id, model_id, metrics, args)
    
    report_filename = f"{eval_id}.md"
    report_path = os.path.join(reports_dir, report_filename)
    with open(report_path, "w") as f:
        f.write(report_content)
    
    # Mirror outputs
    shutil.copy2(eval_path, os.path.join(out_dir, "eval.yaml"))
    shutil.copy2(report_path, os.path.join(out_dir, "report.md"))
    
    print("Results written:")
    print(f"- Eval YAML: {eval_path}")
    print(f"- Report:    {report_path}")
    print(f"- Out dir:   {out_dir}")


def _generate_robustness_report(
    eval_id: str,
    model_id: str,
    metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    """Generate Markdown report for robustness evaluation."""
    
    agg = metrics.get("aggregate", {})
    probes = metrics.get("probes", {})
    
    report = f"""# Robustness Evaluation Report: {eval_id}

## Overview

- **Model**: {model_id}
- **Benchmark**: Robustness Testing (ITU AI4H DEL3 Aligned)
- **Date**: {datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")}

## Aggregate Robustness Scores

| Metric | Score |
|--------|-------|
"""
    
    for metric_name, score in sorted(agg.items()):
        report += f"| **{metric_name}** | {score:.4f} |\n"
    
    report += "\n## Probe Results\n\n"
    
    # Detailed probe results
    for probe_name, probe_result in probes.items():
        if "error" in probe_result:
            report += f"### {probe_name.replace('_', ' ').title()}\n\n"
            report += f"⚠️ Error: {probe_result['error']}\n\n"
            continue
        
        report += f"### {probe_name.replace('_', ' ').title()}\n\n"
        
        if "grid" in probe_result:
            report += f"**Grid**: {probe_result['grid']}\n\n"
        
        if "logit" in probe_result:
            logit = probe_result["logit"]
            if "rAUC" in logit:
                report += f"**Logit rAUC**: {logit['rAUC']:.4f}\n\n"
            if "mean" in logit and isinstance(logit["mean"], list):
                report += "| Perturbation | Cosine Similarity |\n"
                report += "|--------------|------------------|\n"
                grid = probe_result.get("grid", range(len(logit["mean"])))
                for g, m in zip(grid, logit["mean"]):
                    report += f"| {g} | {m:.4f} |\n"
                report += "\n"
            elif "sim_mean" in logit:
                report += f"**Mean Similarity**: {logit['sim_mean']:.4f} (±{logit.get('sim_std', 0):.4f})\n\n"
    
    report += """## Interpretation Guide

- **rAUC (Reverse Area Under Curve)**: Measures output stability as perturbation increases. Higher is better (0-1 scale).
- **Cosine Similarity**: How similar outputs remain after perturbation. 1.0 = identical, 0.0 = orthogonal.
- **Robustness Score**: Aggregate mean of all rAUC values across probes.

## References

- ITU FG-AI4H DEL3: AI4H Requirement Specifications
- brainaug-lab: Reproducible EEG Augmentation Lab
"""
    
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="fmbench",
        description="Opinionated CLI for AI4H-inspired FM benchmark suites.",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    # list-suites
    p_ls = subparsers.add_parser("list-suites", help="List available test suites.")
    p_ls.add_argument(
        "--tests-dir",
        default="tests",
        help="Directory containing test suite YAML files (default: tests).",
    )
    p_ls.set_defaults(func=cmd_list_suites)

    # list-benchmarks
    p_lb = subparsers.add_parser(
        "list-benchmarks", help="List available benchmark definitions."
    )
    p_lb.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory containing benchmark YAML files (default: benchmarks).",
    )
    p_lb.set_defaults(func=cmd_list_benchmarks)

    # run
    p_run = subparsers.add_parser(
        "run",
        help=(
            "Run a test suite for a given model configuration on toy data and "
            "record an eval + report stub."
        ),
    )
    p_run.add_argument(
        "--suite",
        required=True,
        help="Suite ID to run (see `fmbench list-suites`).",
    )
    p_run.add_argument(
        "--model",
        required=True,
        help="Path to a model configuration YAML (must contain `model_id`).",
    )
    p_run.add_argument(
        "--out",
        required=True,
        help="Output directory for run-specific artifacts (eval.yaml, report.md, plots).",
    )
    p_run.add_argument(
        "--dataset",
        help="Override dataset_id for this run (default: suite.inputs.dataset.default).",
    )
    p_run.add_argument(
        "--tests-dir",
        default="tests",
        help="Directory containing test suite YAML files (default: tests).",
    )
    p_run.add_argument(
        "--evals-dir",
        default="evals",
        help="Directory where eval YAML files are stored (default: evals).",
    )
    p_run.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory where Markdown reports are stored (default: reports).",
    )
    p_run.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory containing benchmark YAML files (default: benchmarks).",
    )
    p_run.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Directory containing dataset YAML files (default: datasets).",
    )
    p_run.set_defaults(func=cmd_run)

    # build-leaderboard
    p_bl = subparsers.add_parser(
        "build-leaderboard",
        help="Build the Markdown leaderboard from eval/model/dataset/benchmark YAMLs.",
    )
    p_bl.add_argument(
        "--benchmarks-dir",
        default="benchmarks",
        help="Directory containing benchmark YAML files (default: benchmarks).",
    )
    p_bl.add_argument(
        "--models-dir",
        default="models",
        help="Directory containing model YAML files (default: models).",
    )
    p_bl.add_argument(
        "--datasets-dir",
        default="datasets",
        help="Directory containing dataset YAML files (default: datasets).",
    )
    p_bl.add_argument(
        "--evals-dir",
        default="evals",
        help="Directory containing eval YAML files (default: evals).",
    )
    p_bl.add_argument(
        "--output",
        default="docs/leaderboards/index.md",
        help="Output path for the generated leaderboard Markdown.",
    )
    p_bl.set_defaults(func=cmd_build_leaderboard)

    # generate-toy-data
    p_gtd = subparsers.add_parser(
        "generate-toy-data",
        help="Generate synthetic toy datasets (classification, regression, robustness) in toy_data/.",
    )
    p_gtd.add_argument(
        "--output-dir",
        default="toy_data",
        help="Root directory for generated toy data (default: toy_data).",
    )
    p_gtd.set_defaults(func=cmd_generate_toy_data)

    # run-robustness
    p_rob = subparsers.add_parser(
        "run-robustness",
        help=(
            "Run robustness evaluation for a model using brainaug-lab probes. "
            "Tests model resilience to noise, dropout, and artifacts."
        ),
    )
    p_rob.add_argument(
        "--model",
        required=True,
        help="Path to a model configuration YAML (must contain `model_id`).",
    )
    p_rob.add_argument(
        "--data",
        required=True,
        help="Directory containing X.npy and y.npy (3D time-series format preferred).",
    )
    p_rob.add_argument(
        "--out",
        required=True,
        help="Output directory for robustness evaluation artifacts.",
    )
    p_rob.add_argument(
        "--probes",
        default=None,
        help=(
            "Comma-separated list of probes to run "
            "(default: dropout,noise,line_noise,permutation,shift)."
        ),
    )
    p_rob.add_argument(
        "--sampling-freq",
        type=int,
        default=200,
        help="Sampling frequency in Hz for time-series data (default: 200).",
    )
    p_rob.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42).",
    )
    p_rob.add_argument(
        "--evals-dir",
        default="evals",
        help="Directory for evaluation YAML files (default: evals).",
    )
    p_rob.add_argument(
        "--reports-dir",
        default="reports",
        help="Directory for Markdown reports (default: reports).",
    )
    p_rob.set_defaults(func=cmd_run_robustness)

    return parser


def main(argv: Optional[List[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)


__all__ = ["main", "build_parser"]


