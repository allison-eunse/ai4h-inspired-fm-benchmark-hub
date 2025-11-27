import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import yaml


# Ranking badges and emojis
RANK_BADGES = {
    1: "🥇",
    2: "🥈", 
    3: "🥉",
}

RANK_TITLES = {
    1: "Champion",
    2: "Runner-up",
    3: "Bronze",
}

# Performance tier emojis
TIER_EMOJIS = {
    "excellent": "⭐",
    "good": "✅",
    "fair": "🔶",
    "needs_improvement": "📈",
}


def get_rank_badge(rank: int) -> str:
    """Get emoji badge for rank position."""
    if rank in RANK_BADGES:
        return RANK_BADGES[rank]
    elif rank <= 5:
        return "🏅"
    elif rank <= 10:
        return "🎖️"
    else:
        return f"#{rank}"


def get_performance_tier(score: float, metric: str = "AUROC") -> Tuple[str, str]:
    """Get performance tier based on score."""
    if metric in ["AUROC", "Accuracy", "F1-Score", "robustness_score"]:
        if score >= 0.9:
            return "excellent", "⭐ Excellent"
        elif score >= 0.8:
            return "good", "✅ Good"
        elif score >= 0.7:
            return "fair", "🔶 Fair"
        else:
            return "needs_improvement", "📈 Developing"
    return "unknown", ""


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
    ranked_evals: List[Tuple], 
    models: List[Dict],
    stratify_by: List[str]
) -> str:
    """Generate sub-tables stratified by specific dimensions."""
    md = ""
    
    # Handle both old format (List[Dict]) and new format (List[Tuple])
    for item in ranked_evals:
        if isinstance(item, tuple):
            _, _, eval_data = item
        else:
            eval_data = item
            
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
            
            # Emoji for different stratification types
            strat_emoji = {
                "scanner": "🔬",
                "site": "🏥",
                "acquisition_type": "📡",
                "preprocessing": "⚙️",
                "disease_stage": "🩺",
                "sex": "👤",
                "age_group": "📅",
                "ethnicity": "🌍",
                "report_type": "📄",
                "complexity": "📊",
                "field_strength": "🧲",
            }
            emoji = strat_emoji.get(strat_key, "📈")
            
            # Create collapsible section
            title = strat_key.replace("_", " ").title()
            md += f"\n<details>\n<summary>{emoji} <strong>{model_name}</strong> by {title}</summary>\n\n"
            
            # Get all metric keys from first group
            first_group = next(iter(strat_data.values()))
            sample_metrics = list(first_group.keys()) if first_group else []
            # Filter to important metrics
            display_metrics = [m for m in sample_metrics if m not in ["N"]][:4]
            
            if not display_metrics:
                md += "*No detailed metrics available*\n"
                md += "\n</details>\n"
                continue
            
            # Table header
            md += f"| {title} | " + " | ".join(display_metrics) + " | N |\n"
            md += "|" + "|".join(["---"] * (len(display_metrics) + 2)) + "|\n"
            
            # Sort groups and add performance indicator
            sorted_groups = sorted(strat_data.items(), 
                                   key=lambda x: x[1].get(display_metrics[0], 0) if display_metrics else 0,
                                   reverse=True)
            
            for idx, (group_name, group_metrics) in enumerate(sorted_groups):
                # Add medal for top performers within category
                prefix = ""
                if idx == 0 and len(sorted_groups) > 1:
                    prefix = "🥇 "
                elif idx == 1 and len(sorted_groups) > 2:
                    prefix = "🥈 "
                elif idx == 2 and len(sorted_groups) > 3:
                    prefix = "🥉 "
                
                vals = []
                for m in display_metrics:
                    v = group_metrics.get(m, "-")
                    if isinstance(v, float):
                        vals.append(f"{v:.4f}")
                    else:
                        vals.append(str(v))
                
                n = group_metrics.get("N", "-")
                md += f"| {prefix}{group_name} | " + " | ".join(vals) + f" | {n} |\n"
            
            md += "\n</details>\n"
    
    return md


def get_primary_score(metrics: Dict, ai_task: str = "") -> Tuple[float, str]:
    """Extract primary score for ranking from metrics dict."""
    # Priority order for different task types
    if "generation" in ai_task.lower():
        priority = ["report_quality_score", "clinical_accuracy", "bertscore", "bleu"]
    elif "robustness" in ai_task.lower():
        priority = ["robustness_score", "perm_equivariance"]
    else:
        priority = ["AUROC", "Accuracy", "F1-Score", "Correlation", "R2"]
    
    for key in priority:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key]), key
    
    # Fallback: first numeric value
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.startswith("_"):
            return float(v), k
    
    return 0.0, "unknown"


def generate_markdown_table(
    benchmark: Dict, evals: List[Dict], models: List[Dict], datasets: List[Dict]
) -> str:
    """Generate a gamified Markdown leaderboard with full rankings."""
    
    ai_task = benchmark.get("ai_task", "")
    
    # Header with emoji based on domain
    domain_emoji = {
        "Neurology": "🧠",
        "Genomics": "🧬",
        "Cross-Domain": "🌐",
    }
    domain = benchmark.get("health_domain", "")
    emoji = domain_emoji.get(domain, "📊")
    
    md = f"### {emoji} {benchmark.get('name', 'Unnamed Benchmark')}\n\n"
    
    # Task badges
    task_badges = {
        "Classification": "🎯",
        "Regression": "📈",
        "Generation": "✍️",
        "Reconstruction": "🔄",
        "Robustness Assessment": "🛡️",
    }
    task_badge = task_badges.get(ai_task, "📋")
    
    md += f"{task_badge} **Task**: {ai_task} | "
    md += f"🏥 **Health Topic**: {benchmark.get('health_topic', 'N/A')}\n\n"
    
    if benchmark.get('description'):
        md += f"*{benchmark.get('description')}*\n\n"

    if "clinical_relevance" in benchmark:
        md += f"!!! info \"Clinical Relevance\"\n    {benchmark['clinical_relevance']}\n\n"

    if not evals:
        md += "!!! warning \"No submissions yet\"\n"
        md += "    Be the first to submit! Check the [Submission Guide](../contributing/submission_guide.md).\n\n"
        return md

    # Sort evals by primary metric for ranking
    ranked_evals = []
    for ev in evals:
        metrics = ev.get("metrics", {})
        score, metric_name = get_primary_score(metrics, ai_task)
        ranked_evals.append((score, metric_name, ev))
    
    # Sort descending by score
    ranked_evals.sort(key=lambda x: x[0], reverse=True)
    
    # Deduplicate by model (keep best score per model)
    seen_models = set()
    unique_ranked = []
    for score, metric_name, ev in ranked_evals:
        model_ids = ev.get("model_ids", {})
        model_key = tuple(sorted(model_ids.items()))
        if model_key not in seen_models:
            seen_models.add(model_key)
            unique_ranked.append((score, metric_name, ev))
    
    # Trophy case for top 3
    if len(unique_ranked) >= 1:
        md += "#### 🏆 Leaderboard\n\n"
    
    # Main ranking table
    md += "| Rank | Model | Score | Dataset | Details |\n"
    md += "| :---: | :--- | :---: | :--- | :--- |\n"
    
    for rank, (score, metric_name, ev) in enumerate(unique_ranked, 1):
        # Rank badge
        rank_display = get_rank_badge(rank)
        
        # Model name
        model_names = []
        for role, mid in ev.get("model_ids", {}).items():
            model_data = get_by_id(models, "model_id", mid)
            name = model_data.get("name", mid) if model_data else mid
            model_names.append(name)
        model_str = ", ".join(model_names) if model_names else "Unknown"
        
        # Add trophy styling for top 3
        if rank == 1:
            model_str = f"**{model_str}** 👑"
        elif rank == 2:
            model_str = f"**{model_str}**"
        elif rank == 3:
            model_str = f"**{model_str}**"
        
        # Score with tier indicator
        tier, tier_label = get_performance_tier(score, metric_name)
        score_display = f"{score:.4f}" if isinstance(score, float) else str(score)
        
        # Dataset
        did = ev.get("dataset_id")
        dataset_data = get_by_id(datasets, "dataset_id", did)
        dataset_name = dataset_data.get("name", did) if dataset_data else (did or "-")
        
        # Compact details
        date = ev.get("run_metadata", {}).get("date", "-")
        details = f"{metric_name} | {date}"
        
        md += f"| {rank_display} | {model_str} | {score_display} | {dataset_name} | {details} |\n"
    
    md += "\n"
    
    # Full metrics expansion for each model
    md += "<details>\n<summary>📋 <strong>Full Metrics for All Models</strong></summary>\n\n"
    
    for rank, (score, metric_name, ev) in enumerate(unique_ranked, 1):
        model_names = []
        for role, mid in ev.get("model_ids", {}).items():
            model_data = get_by_id(models, "model_id", mid)
            name = model_data.get("name", mid) if model_data else mid
            model_names.append(name)
        model_str = ", ".join(model_names) if model_names else "Unknown"
        
        md += f"**{get_rank_badge(rank)} {model_str}**\n\n"
        
        metrics = ev.get("metrics", {})
        # Filter to non-dict metrics
        display_metrics = {k: v for k, v in metrics.items() 
                         if not isinstance(v, dict) and not k.startswith("_")}
        
        if display_metrics:
            md += "| Metric | Value |\n|---|---|\n"
            for k, v in sorted(display_metrics.items()):
                val = f"{v:.4f}" if isinstance(v, float) else str(v)
                md += f"| {k} | {val} |\n"
            md += "\n"
    
    md += "</details>\n\n"
    
    # Add stratified sub-tables if available
    stratify_dimensions = ["scanner", "site", "acquisition_type", "preprocessing", 
                          "disease_stage", "sex", "age_group", "ethnicity",
                          "report_type", "complexity", "field_strength"]
    stratified_content = generate_stratified_tables(unique_ranked, models, stratify_dimensions)
    if stratified_content:
        md += "#### 📊 Granular Performance Breakdown\n\n"
        md += "Expand sections below to see how models perform across different conditions:\n\n"
        md += stratified_content
        md += "\n"
    
    # Methodology note
    md += "---\n"
    md += f"*Ranked by {metric_name}. Higher is better. "
    md += f"Last updated from {len(evals)} evaluation(s).*\n\n"
    
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

    # Count stats for summary
    total_benchmarks = len(benchmarks)
    total_evals = len(evals)
    unique_models = set()
    for e in evals:
        for mid in e.get("model_ids", {}).values():
            unique_models.add(mid)

    # Generate Markdown content with gamified header
    content = "# 🏆 Foundation Model Leaderboards\n\n"
    
    # Stats banner
    content += "!!! success \"Benchmark Hub Stats\"\n"
    content += f"    🎯 **{total_benchmarks}** Benchmarks | "
    content += f"🤖 **{len(unique_models)}** Models Evaluated | "
    content += f"📊 **{total_evals}** Total Evaluations\n\n"
    
    content += (
        "Welcome to the **AI4H-Inspired FM Benchmark Hub**! "
        "Rankings below show **all submitted models** from best to developing, "
        "helping you find the right model for your use case.\n\n"
    )
    
    # Quick navigation
    content += "## 🧭 Quick Navigation\n\n"
    domain_emojis = {"Neurology": "🧠", "Genomics": "🧬", "Cross-Domain": "🌐"}
    for domain in sorted(benchmarks_by_domain.keys()):
        emoji = domain_emojis.get(domain, "📊")
        anchor = domain.lower().replace(" ", "-").replace("_", "-")
        content += f"- [{emoji} {domain}](#{anchor})\n"
    content += "\n---\n\n"

    # Iterate by domain for categorization
    for domain in sorted(benchmarks_by_domain.keys()):
        domain_emoji = domain_emojis.get(domain, "📊")
        content += f"## {domain_emoji} {domain}\n\n"

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

    # Footer with contribution info
    content += "---\n\n"
    content += "## 🚀 Submit Your Model\n\n"
    content += "Want to see your Foundation Model on these leaderboards?\n\n"
    content += "1. 📥 **Download** the benchmark suite: `pip install -e .`\n"
    content += "2. 🧪 **Run** evaluations: `python -m fmbench run-robustness --help`\n"
    content += "3. 📤 **Submit** via Pull Request - see [Submission Guide](../contributing/submission_guide.md)\n\n"
    content += "*Aligned with [ITU/WHO FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards for healthcare AI evaluation.*\n"

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


