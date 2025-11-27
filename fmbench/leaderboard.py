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
    priority_keys = ["AUROC", "Accuracy", "F1-Score", "robustness_score", "MSE", "R2",
                     "clinical_accuracy", "finding_recall", "bertscore", "bleu"]
    sorted_keys = sorted(
        display_metrics.keys(),
        key=lambda x: (priority_keys.index(x) if x in priority_keys else 100, x)
    )
    
    # Limit display
    if len(sorted_keys) > max_display:
        sorted_keys = sorted_keys[:max_display]
    
    return "<br>".join([f"**{k}**: {display_metrics[k]}" for k in sorted_keys])


def generate_stratified_tables(
    evals: List[Dict], 
    models: List[Dict],
    stratify_by: List[str]
) -> str:
    """Generate sub-tables stratified by specific dimensions."""
    md = ""
    
    for eval_data in evals:
        metrics = eval_data.get("metrics", {})
        stratified = metrics.get("stratified", {})
        
        if not stratified:
            continue
        
        model_ids = eval_data.get("model_ids", {})
        model_name = list(model_ids.values())[0] if model_ids else "Unknown"
        model_data = get_by_id(models, "model_id", model_name)
        if model_data:
            model_name = model_data.get("name", model_name)
        
        for strat_key in stratify_by:
            if strat_key not in stratified:
                continue
            
            strat_data = stratified[strat_key]
            if not strat_data:
                continue
            
            # Create collapsible section
            title = strat_key.replace("_", " ").title()
            md += f"\n<details>\n<summary>📊 <strong>{model_name}</strong> by {title}</summary>\n\n"
            
            # Get all metric keys from first group
            sample_metrics = list(next(iter(strat_data.values())).keys())
            # Filter to important metrics
            display_metrics = [m for m in sample_metrics if m not in ["N"]][:4]
            
            # Table header
            md += f"| {title} | " + " | ".join(display_metrics) + " | N |\n"
            md += "|" + "|".join(["---"] * (len(display_metrics) + 2)) + "|\n"
            
            # Sort groups
            for group_name in sorted(strat_data.keys()):
                group_metrics = strat_data[group_name]
                vals = [str(group_metrics.get(m, "-")) for m in display_metrics]
                n = group_metrics.get("N", "-")
                md += f"| {group_name} | " + " | ".join(vals) + f" | {n} |\n"
            
            md += "\n</details>\n"
    
    return md


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

    # Main leaderboard table
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
    
    # Add stratified sub-tables if available
    stratify_dimensions = ["scanner", "site", "acquisition_type", "preprocessing", 
                          "disease_stage", "sex", "age_group", "ethnicity"]
    stratified_content = generate_stratified_tables(evals, models, stratify_dimensions)
    if stratified_content:
        md += "#### Granular Performance Breakdown\n\n"
        md += stratified_content
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


