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


def get_metric_explanation(metric: str) -> str:
    """Get a human-readable explanation of what a metric measures."""
    explanations = {
        "AUROC": "Area Under ROC Curve - measures discrimination ability (0.5 = random, 1.0 = perfect)",
        "Accuracy": "Proportion of correct predictions (0.0-1.0)",
        "F1-Score": "Harmonic mean of precision and recall, balancing false positives/negatives",
        "Correlation": "Pearson correlation between predicted and actual values (-1 to 1)",
        "MSE": "Mean Squared Error - average squared difference (lower is better)",
        "R2": "Coefficient of determination - variance explained by model (0.0-1.0)",
        "robustness_score": "Average performance retention under data perturbations (0.0-1.0)",
        "report_quality_score": "Composite score of linguistic fluency + clinical accuracy (0.0-1.0)",
        "clinical_accuracy": "Proportion of clinically correct findings in generated reports",
        "bertscore": "Semantic similarity using BERT embeddings (0.0-1.0)",
        "bleu": "N-gram overlap with reference text (0-100)",
        "finding_recall": "Proportion of ground-truth findings captured in output",
        "hallucination_rate": "Proportion of generated content not in source (lower is better)",
        "perm_equivariance": "Model's consistency when input features are permuted",
        "dropout_rAUC": "Relative AUROC under feature dropout perturbations",
        "noise_rAUC": "Relative AUROC under Gaussian noise injection",
    }
    return explanations.get(metric, f"Performance measure for this task")


def generate_scoring_methodology(primary_metric: str, ai_task: str) -> str:
    """Generate a clean, readable explanation of scoring."""
    md = "\n<details>\n<summary>📐 <strong>How are models scored?</strong></summary>\n\n"
    
    # Simple metric explanation
    md += f"**Ranking by:** `{primary_metric}`\n\n"
    
    # Score tiers - simple visual table
    md += "| Score | Rating | Meaning |\n"
    md += "|:---:|:---:|:---|\n"
    md += "| ≥0.90 | ⭐ Excellent | Clinical-ready |\n"
    md += "| 0.80-0.89 | ✅ Good | Needs validation |\n"
    md += "| 0.70-0.79 | 🔶 Fair | Research only |\n"
    md += "| <0.70 | 📈 Developing | Not recommended |\n\n"
    
    # Brief rules
    md += "*Aligned with [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) standards (DEL3).*\n\n"
    
    md += "</details>\n\n"
    
    return md


def generate_ranking_explanation(
    ranked_evals: List[Tuple],
    models: List[Dict],
    primary_metric: str,
    ai_task: str
) -> str:
    """Generate a detailed explanation of why models are ranked the way they are."""
    if len(ranked_evals) < 2:
        return ""
    
    md = "\n#### 📖 Ranking Explanation\n\n"
    md += "!!! abstract \"Why These Rankings?\"\n"
    
    # Top performer analysis
    top_score, top_metric, top_ev = ranked_evals[0]
    top_model_id = list(top_ev.get("model_ids", {}).values())[0] if top_ev.get("model_ids") else "Unknown"
    top_model = get_by_id(models, "model_id", top_model_id)
    top_name = top_model.get("name", top_model_id) if top_model else top_model_id
    
    md += f"    **🥇 {top_name}** leads with {primary_metric}={top_score:.4f}\n\n"
    
    # Comparison with runner-up
    if len(ranked_evals) >= 2:
        second_score, _, second_ev = ranked_evals[1]
        second_model_id = list(second_ev.get("model_ids", {}).values())[0] if second_ev.get("model_ids") else "Unknown"
        second_model = get_by_id(models, "model_id", second_model_id)
        second_name = second_model.get("name", second_model_id) if second_model else second_model_id
        
        diff = top_score - second_score
        diff_pct = (diff / second_score * 100) if second_score != 0 else 0
        
        md += f"    - Gap to 🥈 **{second_name}**: +{diff:.4f} ({diff_pct:.1f}% better)\n"
    
    # Range analysis
    if len(ranked_evals) >= 3:
        last_score, _, last_ev = ranked_evals[-1]
        range_val = top_score - last_score
        md += f"    - Score range across all models: {range_val:.4f}\n"
    
    # Performance tier breakdown
    tiers = {"excellent": 0, "good": 0, "fair": 0, "needs_improvement": 0}
    for score, _, _ in ranked_evals:
        tier, _ = get_performance_tier(score, primary_metric)
        if tier in tiers:
            tiers[tier] += 1
    
    tier_summary = []
    if tiers["excellent"]: tier_summary.append(f"⭐ {tiers['excellent']} excellent")
    if tiers["good"]: tier_summary.append(f"✅ {tiers['good']} good")
    if tiers["fair"]: tier_summary.append(f"🔶 {tiers['fair']} fair")
    if tiers["needs_improvement"]: tier_summary.append(f"📈 {tiers['needs_improvement']} developing")
    
    if tier_summary:
        md += f"    - Performance distribution: {', '.join(tier_summary)}\n"
    
    md += "\n"
    return md


def generate_full_metrics_table(
    ranked_evals: List[Tuple],
    models: List[Dict],
    primary_metric: str
) -> str:
    """Generate an expanded full metrics comparison table."""
    if not ranked_evals:
        return ""
    
    md = "\n#### 📋 Complete Metrics Comparison\n\n"
    
    # Collect all metric keys across all evals
    all_metric_keys = set()
    for _, _, ev in ranked_evals:
        metrics = ev.get("metrics", {})
        for k, v in metrics.items():
            if not isinstance(v, dict) and not k.startswith("_"):
                all_metric_keys.add(k)
    
    # Priority ordering
    priority = ["AUROC", "Accuracy", "F1-Score", "robustness_score", "report_quality_score",
                "clinical_accuracy", "Correlation", "MSE", "bertscore", "bleu",
                "finding_recall", "hallucination_rate"]
    sorted_metrics = sorted(all_metric_keys, 
                           key=lambda x: (priority.index(x) if x in priority else 100, x))
    
    # Build comparison table
    md += "| Rank | Model | " + " | ".join(sorted_metrics[:8]) + " |\n"
    md += "|:---:|:---|" + "|".join([":---:"] * min(len(sorted_metrics), 8)) + "|\n"
    
    for rank, (score, metric_name, ev) in enumerate(ranked_evals, 1):
        # Model name
        model_id = list(ev.get("model_ids", {}).values())[0] if ev.get("model_ids") else "Unknown"
        model_data = get_by_id(models, "model_id", model_id)
        model_name = model_data.get("name", model_id) if model_data else model_id
        
        # Rank display
        rank_badge = get_rank_badge(rank)
        
        # Performance tier indicator
        tier, tier_label = get_performance_tier(score, primary_metric)
        tier_emoji = {"excellent": "⭐", "good": "✅", "fair": "🔶", "needs_improvement": "📈"}.get(tier, "")
        
        # Metric values
        metrics = ev.get("metrics", {})
        vals = []
        for m in sorted_metrics[:8]:
            v = metrics.get(m, "-")
            if isinstance(v, float):
                # Highlight if this is primary metric
                if m == primary_metric:
                    vals.append(f"**{v:.4f}**")
                else:
                    vals.append(f"{v:.4f}")
            else:
                vals.append(str(v))
        
        md += f"| {rank_badge} {tier_emoji} | {model_name} | " + " | ".join(vals) + " |\n"
    
    # Legend
    md += "\n"
    md += "!!! tip \"Legend\"\n"
    md += f"    📊 **Primary metric**: {primary_metric} (bold) | "
    md += "⭐ Excellent (≥0.9) | ✅ Good (≥0.8) | 🔶 Fair (≥0.7) | 📈 Developing (<0.7)\n\n"
    
    return md


def generate_markdown_table(
    benchmark: Dict, evals: List[Dict], models: List[Dict], datasets: List[Dict]
) -> str:
    """Generate a gamified Markdown leaderboard with full rankings and explanations."""
    
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
    
    # Get primary metric name from top eval
    primary_metric = unique_ranked[0][1] if unique_ranked else "score"
    
    # Trophy header
    md += "#### 🏆 Leaderboard\n\n"
    
    # Podium for top 3 if we have multiple models
    if len(unique_ranked) >= 3:
        md += "```\n"
        md += "         🥇          \n"
        top_name = list(unique_ranked[0][2].get("model_ids", {}).values())[0] if unique_ranked[0][2].get("model_ids") else "?"
        top_model = get_by_id(models, "model_id", top_name)
        top_display = (top_model.get("name", top_name) if top_model else top_name)[:12]
        md += f"     [{top_display}]    \n"
        md += "    ┌─────────┐     \n"
        
        second_name = list(unique_ranked[1][2].get("model_ids", {}).values())[0] if unique_ranked[1][2].get("model_ids") else "?"
        second_model = get_by_id(models, "model_id", second_name)
        second_display = (second_model.get("name", second_name) if second_model else second_name)[:10]
        
        third_name = list(unique_ranked[2][2].get("model_ids", {}).values())[0] if unique_ranked[2][2].get("model_ids") else "?"
        third_model = get_by_id(models, "model_id", third_name)
        third_display = (third_model.get("name", third_name) if third_model else third_name)[:10]
        
        md += f" 🥈 │         │ 🥉  \n"
        md += f"[{second_display}]│         │[{third_display}]\n"
        md += "────┴─────────┴────\n"
        md += "```\n\n"
    
    # Main ranking table - ALL models
    md += f"**All {len(unique_ranked)} models ranked by {primary_metric}:**\n\n"
    md += "| Rank | Model | Score | Performance | Dataset | Date |\n"
    md += "| :---: | :--- | :---: | :---: | :--- | :---: |\n"
    
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
        
        # Add styling for top 3
        if rank == 1:
            model_str = f"**{model_str}** 👑"
        elif rank == 2:
            model_str = f"**{model_str}** 🌟"
        elif rank == 3:
            model_str = f"**{model_str}** ✨"
        
        # Score display
        score_display = f"{score:.4f}" if isinstance(score, float) else str(score)
        
        # Performance tier
        tier, tier_label = get_performance_tier(score, primary_metric)
        
        # Dataset
        did = ev.get("dataset_id")
        dataset_data = get_by_id(datasets, "dataset_id", did)
        dataset_name = dataset_data.get("name", did) if dataset_data else (did or "-")
        
        # Date
        date = ev.get("run_metadata", {}).get("date", "-")
        
        md += f"| {rank_display} | {model_str} | {score_display} | {tier_label} | {dataset_name} | {date} |\n"
    
    md += "\n"
    
    # Add ranking explanation
    md += generate_ranking_explanation(unique_ranked, models, primary_metric, ai_task)
    
    # Add scoring methodology explanation
    md += generate_scoring_methodology(primary_metric, ai_task)
    
    # Full metrics comparison table (visible by default)
    md += generate_full_metrics_table(unique_ranked, models, primary_metric)
    
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
    md += f"*Ranked by **{primary_metric}** (higher is better). "
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

    # Footer with submission info
    content += "---\n\n"
    content += "## 🚀 Get Your Model on the Leaderboard\n\n"
    content += "Want to see your Foundation Model ranked here?\n\n"
    content += "1. 📥 **Download** the benchmark suite and run locally\n"
    content += "2. 🧪 **Evaluate** your model: `python -m fmbench run --help`\n"
    content += "3. 📤 **Submit** your results via [GitHub Issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=benchmark_submission.md)\n\n"
    content += "💡 **Propose new evaluation protocols** via [Issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=protocol_proposal.md)\n\n"
    content += "!!! note \"Curated Benchmark Hub\"\n"
    content += "    All submissions are reviewed before being added. See [Submission Guide](../contributing/submission_guide.md) for details.\n\n"
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


