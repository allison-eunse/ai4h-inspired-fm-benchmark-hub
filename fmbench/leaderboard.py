import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional

import yaml


def load_yaml_files(directory: str) -> List[Dict]:
    """Load all YAML files from a directory into a list of dicts."""
    data: List[Dict] = []
    files = glob.glob(os.path.join(directory, "*.yaml"))
    for f in files:
        try:
            with open(f, "r") as stream:
                content = yaml.safe_load(stream)
                if content:
                    # Keep track of origin file for debugging
                    content["_filename"] = os.path.basename(f)
                    data.append(content)
        except yaml.YAMLError as exc:
            print(f"Error parsing {f}: {exc}", file=sys.stderr)
    return data


def get_by_id(data_list: List[Dict], id_key: str, target_id: str) -> Optional[Dict]:
    """Find an item in a list of dicts by its ID key."""
    for item in data_list:
        if item.get(id_key) == target_id:
            return item
    return None


def format_metrics(metrics_dict: Optional[Dict], max_display: int = 6) -> str:
    """Format a dictionary of metrics into a Markdown-friendly string."""
    if not metrics_dict:
        return "-"
    
    # Filter out internal/nested keys
    display_metrics = {
        k: v for k, v in metrics_dict.items()
        if not k.startswith("_") and not isinstance(v, dict)
    }
    
    # Prioritize common metrics
    priority_keys = ["AUROC", "Accuracy", "F1-Score", "robustness_score", "MSE", "R2"]
    sorted_keys = sorted(
        display_metrics.keys(),
        key=lambda x: (priority_keys.index(x) if x in priority_keys else 100, x)
    )
    
    # Limit display
    if len(sorted_keys) > max_display:
        sorted_keys = sorted_keys[:max_display]
    
    return "<br>".join([f"**{k}**: {display_metrics[k]}" for k in sorted_keys])


def generate_markdown_table(
    benchmark: Dict, evals: List[Dict], models: List[Dict], datasets: List[Dict]
) -> str:
    """Generate a Markdown table for a specific benchmark."""

    # Table Header with ITU context
    md = f"### {benchmark.get('name', 'Unnamed Benchmark')}\n\n"
    md += (
        f"**Health Topic**: {benchmark.get('health_topic', 'N/A')} | "
        f"**AI Task**: {benchmark.get('ai_task', 'N/A')}\n\n"
    )
    md += f"*{benchmark.get('description', '')}*\n\n"

    # Add ITU Metadata if available
    if "clinical_relevance" in benchmark:
        md += f"**Clinical Relevance**: {benchmark['clinical_relevance']}\n\n"

    md += "| Model | Dataset | Metrics | Status | Date |\n"
    md += "| :--- | :--- | :--- | :--- | :--- |\n"

    if not evals:
        md += "| *No evaluations yet* | | | | |\n"
        return md

    for ev in evals:
        # Resolve Model(s)
        model_names = []
        for role, mid in ev.get("model_ids", {}).items():
            model_data = get_by_id(models, "model_id", mid)
            name = model_data.get("name", mid) if model_data else mid
            model_names.append(f"{name} ({role})")
        model_str = "<br>".join(model_names) if model_names else "-"

        # Resolve Dataset
        did = ev.get("dataset_id")
        dataset_data = get_by_id(datasets, "dataset_id", did)
        dataset_name = dataset_data.get("name", did) if dataset_data else (did or "-")

        # Metrics
        metrics_str = format_metrics(ev.get("metrics", {}))

        # Metadata
        status = ev.get("status", "Unknown")
        date = ev.get("run_metadata", {}).get("date", "-")

        md += f"| {model_str} | {dataset_name} | {metrics_str} | {status} | {date} |\n"

    md += "\n"
    return md


def build_leaderboard(
    benchmarks_dir: str = "benchmarks",
    models_dir: str = "models",
    datasets_dir: str = "datasets",
    evals_dir: str = "evals",
    output_path: str = "docs/leaderboards/index.md",
) -> str:
    """
    Build the Markdown leaderboard file from repository metadata.

    Returns the path of the generated file.
    """
    print("Building Leaderboard with ITU AI4H Standards...")

    # Load all data
    benchmarks = load_yaml_files(benchmarks_dir)
    models = load_yaml_files(models_dir)
    datasets = load_yaml_files(datasets_dir)
    evals = load_yaml_files(evals_dir)

    # Filter out template
    benchmarks = [
        b for b in benchmarks if b.get("benchmark_id") != "unique_benchmark_id_here"
    ]

    # Group evals by benchmark_id
    evals_by_benchmark: Dict[str, List[Dict]] = defaultdict(list)
    for e in evals:
        bid = e.get("benchmark_id")
        if bid:
            evals_by_benchmark[bid].append(e)
        else:
            print(
                f"Warning: Eval {e.get('_filename')} missing benchmark_id",
                file=sys.stderr,
            )

    # Group benchmarks by health_domain
    benchmarks_by_domain: Dict[str, List[Dict]] = defaultdict(list)
    for b in benchmarks:
        domain = b.get("health_domain", "Uncategorized")
        benchmarks_by_domain[domain].append(b)

    # Generate Markdown content
    content = "# Benchmark Leaderboards\n\n"
    content += (
        "Automated leaderboards generated from repository metadata, "
        "aligned with **ITU FG-AI4H** standards.\n\n"
    )

    # Iterate by domain for categorization
    for domain in sorted(benchmarks_by_domain.keys()):
        content += f"## {domain}\n\n"

        domain_benchmarks = benchmarks_by_domain[domain]
        # Sort benchmarks by name for stable ordering
        domain_benchmarks.sort(key=lambda x: x.get("name", ""))

        for bm in domain_benchmarks:
            bid = bm.get("benchmark_id")
            bm_evals = evals_by_benchmark.get(bid, [])
            # Sort evals by date desc
            bm_evals.sort(
                key=lambda x: x.get("run_metadata", {}).get("date", ""), reverse=True
            )

            content += generate_markdown_table(bm, bm_evals, models, datasets)

    # Write to file
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Leaderboard written to {output_path}")
    return output_path


__all__ = [
    "build_leaderboard",
    "load_yaml_files",
    "generate_markdown_table",
]


