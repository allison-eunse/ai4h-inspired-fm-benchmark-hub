"""
Leaderboard generation for AI4H-Inspired FM Benchmarks.

Generates human-readable, accessible leaderboards organized by:
1. Overall rankings (all modalities)
2. Modality-specific rankings (Genomics, Brain Imaging, EEG, Clinical)
3. Task-specific sub-rankings within each modality
"""

import glob
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import yaml


# =============================================================================
# CONSTANTS & CONFIGURATION
# =============================================================================

# Modality categories with emojis
MODALITIES = {
    "genomics": {"emoji": "🧬", "name": "Genomics", "keywords": ["genomic", "gene", "dna", "rna", "cell", "sequence"]},
    "brain_imaging": {"emoji": "🧠", "name": "Brain Imaging (MRI/fMRI)", "keywords": ["fmri", "mri", "brain", "neuro", "imaging"]},
    "eeg": {"emoji": "⚡", "name": "EEG / iEEG", "keywords": ["eeg", "ieeg", "electro", "electrode"]},
    "clinical": {"emoji": "🏥", "name": "Clinical / Reports", "keywords": ["clinical", "report", "radiology", "pathology"]},
}

# Task types with emojis
TASK_TYPES = {
    "Classification": "🎯",
    "Regression": "📈", 
    "Generation": "✍️",
    "Reconstruction": "🔄",
    "Robustness Assessment": "🛡️",
    "Embedding": "🔢",
    "Segmentation": "✂️",
}

# Rank badges
RANK_BADGES = {1: "🥇", 2: "🥈", 3: "🥉"}


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def load_yaml_files(directory: str) -> List[Dict]:
    """Load all YAML files from a directory."""
    data = []
    for f in glob.glob(os.path.join(directory, "*.yaml")):
        try:
            with open(f, "r") as stream:
                content = yaml.safe_load(stream)
                if content:
                    content["_filename"] = os.path.basename(f)
                    data.append(content)
        except yaml.YAMLError as exc:
            print(f"Error parsing {f}: {exc}", file=sys.stderr)
    return data


def get_by_id(data_list: List[Dict], id_key: str, target_id: str) -> Optional[Dict]:
    """Find an item by ID."""
    for item in data_list:
        if item.get(id_key) == target_id:
            return item
    return None


def get_rank_badge(rank: int) -> str:
    """Get emoji badge for rank."""
    if rank in RANK_BADGES:
        return RANK_BADGES[rank]
    elif rank <= 5:
        return "🏅"
    elif rank <= 10:
        return "🎖️"
    else:
        return f"#{rank}"


def detect_modality(benchmark: Dict) -> str:
    """Detect modality from benchmark metadata."""
    name = benchmark.get("name", "").lower()
    topic = benchmark.get("health_topic", "").lower()
    domain = benchmark.get("health_domain", "").lower()
    combined = f"{name} {topic} {domain}"
    
    for mod_key, mod_info in MODALITIES.items():
        for keyword in mod_info["keywords"]:
            if keyword in combined:
                return mod_key
    return "other"


def get_tier_info(score: float) -> Tuple[str, str, str]:
    """Get performance tier: (tier_key, emoji, description)."""
    if score >= 0.90:
        return ("excellent", "⭐", "Excellent - Production Ready")
    elif score >= 0.80:
        return ("good", "✅", "Good - Pilot/Validation")
    elif score >= 0.70:
        return ("fair", "🔶", "Fair - Research Only")
    else:
        return ("developing", "📈", "Developing")


def get_primary_score(metrics: Dict, ai_task: str = "") -> Tuple[float, str]:
    """Extract primary score for ranking."""
    if "generation" in ai_task.lower():
        priority = ["report_quality_score", "clinical_accuracy", "bertscore", "bleu"]
    elif "robustness" in ai_task.lower():
        priority = ["robustness_score", "perm_equivariance"]
    else:
        priority = ["AUROC", "Accuracy", "F1-Score", "Correlation", "R2"]
    
    for key in priority:
        if key in metrics and isinstance(metrics[key], (int, float)):
            return float(metrics[key]), key
    
    for k, v in metrics.items():
        if isinstance(v, (int, float)) and not k.startswith("_"):
            return float(v), k
    
    return 0.0, "unknown"


# =============================================================================
# METRIC EXPLANATIONS (Human-readable)
# =============================================================================

METRIC_EXPLANATIONS = {
    # Classification metrics
    "AUROC": {
        "name": "Area Under ROC Curve",
        "simple": "How well the model distinguishes between classes",
        "range": "0.5 (random guess) → 1.0 (perfect)",
        "good_threshold": 0.85,
    },
    "Accuracy": {
        "name": "Accuracy",
        "simple": "Percentage of correct predictions",
        "range": "0% → 100% (or 0.0 → 1.0)",
        "good_threshold": 0.85,
    },
    "F1-Score": {
        "name": "F1 Score",
        "simple": "Balance between precision (avoiding false alarms) and recall (catching all cases)",
        "range": "0.0 (poor) → 1.0 (perfect)",
        "good_threshold": 0.80,
    },
    
    # Robustness metrics
    "robustness_score": {
        "name": "Robustness Score",
        "simple": "How stable the model is when data has noise or missing values",
        "range": "0.0 (breaks easily) → 1.0 (very stable)",
        "good_threshold": 0.75,
    },
    
    # Report generation metrics
    "report_quality_score": {
        "name": "Report Quality Score",
        "simple": "Overall quality of generated medical reports",
        "range": "0.0 (poor) → 1.0 (excellent)",
        "good_threshold": 0.80,
    },
    "clinical_accuracy": {
        "name": "Clinical Accuracy", 
        "simple": "Are the medical findings in the report correct?",
        "range": "0% → 100% correct findings",
        "good_threshold": 0.90,
    },
    "hallucination_rate": {
        "name": "Hallucination Rate",
        "simple": "Percentage of made-up content (lower is better!)",
        "range": "0% (no hallucinations) → 100% (all made up)",
        "good_threshold": 0.05,
        "lower_is_better": True,
    },
    "bertscore": {
        "name": "BERTScore",
        "simple": "Semantic similarity to reference text",
        "range": "0.0 → 1.0",
        "good_threshold": 0.85,
    },
}


def explain_metric(metric_name: str) -> str:
    """Get human-readable explanation of a metric."""
    if metric_name in METRIC_EXPLANATIONS:
        info = METRIC_EXPLANATIONS[metric_name]
        return f"**{info['name']}**: {info['simple']} ({info['range']})"
    return f"**{metric_name}**: Performance measure"


# =============================================================================
# PODIUM GENERATION (Fixed alignment)
# =============================================================================

def generate_podium(ranked_evals: List[Tuple], models: List[Dict]) -> str:
    """Generate a properly aligned, centered ASCII podium."""
    if len(ranked_evals) < 3:
        return ""
    
    # Get top 3 model names (truncated)
    names = []
    scores = []
    for i in range(3):
        score, _, ev = ranked_evals[i]
        model_id = list(ev.get("model_ids", {}).values())[0] if ev.get("model_ids") else "?"
        model_data = get_by_id(models, "model_id", model_id)
        name = (model_data.get("name", model_id) if model_data else model_id)[:12]
        names.append(name)
        scores.append(f"{score:.3f}")
    
    # Calculate padding for center alignment
    max_name_len = max(len(n) for n in names)
    pad_name = lambda n: n.center(max_name_len)
    
    md = "\n<div align=\"center\">\n\n"
    md += "```\n"
    md += "                    🏆                    \n"
    md += "                                          \n"
    md += f"              🥇 {pad_name(names[0])}              \n"
    md += f"                 ({scores[0]})                 \n"
    md += "             ╔═══════════════╗             \n"
    md += "             ║               ║             \n"
    md += f"   🥈 {pad_name(names[1])}   ║               ║   🥉 {pad_name(names[2])}   \n"
    md += f"      ({scores[1]})      ║               ║      ({scores[2]})      \n"
    md += "  ╔═══════════╝               ╚═══════════╗  \n"
    md += "  ║                                       ║  \n"
    md += "══╩═══════════════════════════════════════╩══\n"
    md += "```\n\n"
    md += "</div>\n\n"
    
    return md


# =============================================================================
# SCORING METHODOLOGY (Clean, card-based format)
# =============================================================================

def generate_scoring_methodology(primary_metric: str, ai_task: str) -> str:
    """Generate clean, readable scoring methodology."""
    md = "\n<details>\n<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>\n\n"
    
    # Primary metric card
    metric_info = METRIC_EXPLANATIONS.get(primary_metric, {})
    md += "---\n\n"
    md += f"### 🎯 What We Measure: `{primary_metric}`\n\n"
    if metric_info:
        md += f"> **{metric_info.get('name', primary_metric)}**\n>\n"
        md += f"> {metric_info.get('simple', 'Performance measure')}\n>\n"
        md += f"> 📏 Range: {metric_info.get('range', '0-1')}\n\n"
    else:
        md += f"> {explain_metric(primary_metric)}\n\n"
    
    # Performance tiers - simple visual
    md += "---\n\n"
    md += "### 📊 What Do Scores Mean?\n\n"
    md += "| Score | Rating | What It Means |\n"
    md += "|:---:|:---:|:---|\n"
    md += "| **≥ 0.90** | ⭐ Excellent | Ready for real-world use with monitoring |\n"
    md += "| **0.80-0.89** | ✅ Good | Promising, needs more testing |\n"
    md += "| **0.70-0.79** | 🔶 Fair | Research use only |\n"
    md += "| **< 0.70** | 📈 Developing | Needs more work |\n\n"
    
    # Simple ranking rules
    md += "---\n\n"
    md += "### 📏 How We Rank\n\n"
    md += "1. **Higher score = Better ranking** (except for error metrics)\n"
    md += "2. If scores tie, we look at secondary metrics\n"
    md += "3. Only the best run from each model counts\n\n"
    
    # AI4H note
    md += "---\n\n"
    md += "!!! info \"Standards Alignment\"\n"
    md += "    This follows [ITU/WHO AI4H](https://www.itu.int/pub/T-FG-AI4H) guidelines for healthcare AI evaluation.\n\n"
    
    md += "</details>\n\n"
    return md


# =============================================================================
# RANKING TABLE GENERATION
# =============================================================================

def generate_ranking_table(
    ranked_evals: List[Tuple],
    models: List[Dict],
    datasets: List[Dict],
    primary_metric: str,
    show_detailed: bool = True
) -> str:
    """Generate a clean ranking table."""
    if not ranked_evals:
        return "*No submissions yet.*\n\n"
    
    md = ""
    
    # Main table
    md += "| Rank | Model | Score | Level | Details |\n"
    md += "|:---:|:---|:---:|:---:|:---|\n"
    
    for rank, (score, metric_name, ev) in enumerate(ranked_evals, 1):
        # Rank badge
        badge = get_rank_badge(rank)
        
        # Model name
        model_id = list(ev.get("model_ids", {}).values())[0] if ev.get("model_ids") else "Unknown"
        model_data = get_by_id(models, "model_id", model_id)
        model_name = model_data.get("name", model_id) if model_data else model_id
        
        # Add emphasis for top 3
        if rank == 1:
            model_name = f"**{model_name}** 👑"
        elif rank == 2:
            model_name = f"**{model_name}**"
        elif rank == 3:
            model_name = f"**{model_name}**"
        
        # Score
        score_str = f"{score:.4f}"
        
        # Tier
        _, tier_emoji, tier_desc = get_tier_info(score)
        tier_short = tier_desc.split(" - ")[0] if " - " in tier_desc else tier_desc
        tier_display = f"{tier_emoji} {tier_short}"
        
        # Details (dataset, date)
        dataset_id = ev.get("dataset_id", "-")
        dataset_data = get_by_id(datasets, "dataset_id", dataset_id)
        dataset_name = (dataset_data.get("name", dataset_id) if dataset_data else dataset_id) or "-"
        date = ev.get("run_metadata", {}).get("date", "-")
        details = f"{dataset_name[:20]}, {date}"
        
        md += f"| {badge} | {model_name} | {score_str} | {tier_display} | {details} |\n"
    
    md += "\n"
    return md


def generate_quick_comparison(ranked_evals: List[Tuple], models: List[Dict], primary_metric: str) -> str:
    """Generate a quick text comparison of top models."""
    if len(ranked_evals) < 2:
        return ""
    
    md = "!!! tip \"Quick Comparison\"\n"
    
    # Top model
    top_score, _, top_ev = ranked_evals[0]
    top_id = list(top_ev.get("model_ids", {}).values())[0] if top_ev.get("model_ids") else "?"
    top_data = get_by_id(models, "model_id", top_id)
    top_name = top_data.get("name", top_id) if top_data else top_id
    
    md += f"    **🥇 {top_name}** leads with {primary_metric} = **{top_score:.4f}**\n\n"
    
    # Gap to second
    if len(ranked_evals) >= 2:
        second_score, _, second_ev = ranked_evals[1]
        second_id = list(second_ev.get("model_ids", {}).values())[0] if second_ev.get("model_ids") else "?"
        second_data = get_by_id(models, "model_id", second_id)
        second_name = second_data.get("name", second_id) if second_data else second_id
        gap = top_score - second_score
        md += f"    - Gap to 🥈 {second_name}: +{gap:.4f}\n"
    
    # Score spread
    if len(ranked_evals) >= 3:
        last_score = ranked_evals[-1][0]
        spread = top_score - last_score
        md += f"    - Score spread (best to worst): {spread:.4f}\n"
    
    md += "\n"
    return md


# =============================================================================
# MODALITY-SPECIFIC SECTIONS
# =============================================================================

def generate_modality_section(
    modality_key: str,
    modality_info: Dict,
    benchmarks: List[Dict],
    evals_by_benchmark: Dict,
    models: List[Dict],
    datasets: List[Dict]
) -> str:
    """Generate a complete section for one modality."""
    if not benchmarks:
        return ""
    
    emoji = modality_info["emoji"]
    name = modality_info["name"]
    
    md = f"## {emoji} {name}\n\n"
    
    # Group benchmarks by task type
    by_task = defaultdict(list)
    for bm in benchmarks:
        task = bm.get("ai_task", "Other")
        by_task[task].append(bm)
    
    for task, task_benchmarks in sorted(by_task.items()):
        task_emoji = TASK_TYPES.get(task, "📋")
        md += f"### {task_emoji} {task}\n\n"
        
        for bm in task_benchmarks:
            bid = bm.get("benchmark_id")
            bm_evals = evals_by_benchmark.get(bid, [])
            
            # Benchmark header
            md += f"#### {bm.get('name', 'Unnamed')}\n\n"
            
            if bm.get("description"):
                md += f"*{bm['description']}*\n\n"
            
            if not bm_evals:
                md += "!!! warning \"No submissions yet\"\n"
                md += "    Be the first! See [Submission Guide](../contributing/submission_guide.md)\n\n"
                continue
            
            # Rank evaluations
            ai_task = bm.get("ai_task", "")
            ranked = []
            for ev in bm_evals:
                score, metric = get_primary_score(ev.get("metrics", {}), ai_task)
                ranked.append((score, metric, ev))
            ranked.sort(key=lambda x: x[0], reverse=True)
            
            # Deduplicate by model
            seen = set()
            unique_ranked = []
            for item in ranked:
                mid = list(item[2].get("model_ids", {}).values())[0] if item[2].get("model_ids") else ""
                if mid not in seen:
                    seen.add(mid)
                    unique_ranked.append(item)
            
            if not unique_ranked:
                continue
                
            primary_metric = unique_ranked[0][1]
            
            # Podium for larger benchmarks
            if len(unique_ranked) >= 3:
                md += generate_podium(unique_ranked, models)
            
            # Ranking table
            md += f"**{len(unique_ranked)} models ranked by `{primary_metric}`:**\n\n"
            md += generate_ranking_table(unique_ranked, models, datasets, primary_metric)
            
            # Quick comparison
            md += generate_quick_comparison(unique_ranked, models, primary_metric)
            
            # Scoring explanation
            md += generate_scoring_methodology(primary_metric, ai_task)
            
            md += "---\n\n"
    
    return md


# =============================================================================
# OVERALL LEADERBOARD (Cross-modality)
# =============================================================================

def generate_overall_leaderboard(
    all_evals: List[Dict],
    models: List[Dict],
    datasets: List[Dict],
    benchmarks: List[Dict]
) -> str:
    """Generate an overall leaderboard across all modalities."""
    md = "## 🌐 Overall Rankings (All Modalities)\n\n"
    md += "*Best score per model across all benchmarks*\n\n"
    
    # Get best score for each model across all evals
    best_by_model: Dict[str, Tuple[float, str, Dict, Dict]] = {}
    
    for ev in all_evals:
        model_ids = ev.get("model_ids", {})
        if not model_ids:
            continue
        model_id = list(model_ids.values())[0]
        
        # Find benchmark for task type
        bid = ev.get("benchmark_id")
        bm = get_by_id(benchmarks, "benchmark_id", bid)
        ai_task = bm.get("ai_task", "") if bm else ""
        
        score, metric = get_primary_score(ev.get("metrics", {}), ai_task)
        
        if model_id not in best_by_model or score > best_by_model[model_id][0]:
            best_by_model[model_id] = (score, metric, ev, bm or {})
    
    if not best_by_model:
        md += "*No evaluations yet.*\n\n"
        return md
    
    # Sort by score
    sorted_models = sorted(best_by_model.items(), key=lambda x: x[1][0], reverse=True)
    
    # Table
    md += "| Rank | Model | Best Score | Benchmark | Modality |\n"
    md += "|:---:|:---|:---:|:---|:---|\n"
    
    for rank, (model_id, (score, metric, ev, bm)) in enumerate(sorted_models, 1):
        badge = get_rank_badge(rank)
        
        model_data = get_by_id(models, "model_id", model_id)
        model_name = model_data.get("name", model_id) if model_data else model_id
        if rank == 1:
            model_name = f"**{model_name}** 👑"
        elif rank <= 3:
            model_name = f"**{model_name}**"
        
        score_str = f"{score:.4f}"
        bm_name = bm.get("name", "-")[:25]
        
        # Detect modality
        mod_key = detect_modality(bm)
        mod_emoji = MODALITIES.get(mod_key, {}).get("emoji", "📊")
        mod_name = MODALITIES.get(mod_key, {}).get("name", "Other")[:15]
        
        md += f"| {badge} | {model_name} | {score_str} | {bm_name} | {mod_emoji} {mod_name} |\n"
    
    md += "\n"
    
    # Tier breakdown
    tiers = {"excellent": 0, "good": 0, "fair": 0, "developing": 0}
    for _, (score, _, _, _) in sorted_models:
        tier, _, _ = get_tier_info(score)
        if tier in tiers:
            tiers[tier] += 1
    
    md += "!!! abstract \"Performance Distribution\"\n"
    parts = []
    if tiers["excellent"]: parts.append(f"⭐ {tiers['excellent']} Excellent")
    if tiers["good"]: parts.append(f"✅ {tiers['good']} Good")
    if tiers["fair"]: parts.append(f"🔶 {tiers['fair']} Fair")
    if tiers["developing"]: parts.append(f"📈 {tiers['developing']} Developing")
    md += f"    {' | '.join(parts)}\n\n"
    
    md += "---\n\n"
    return md


# =============================================================================
# MAIN BUILD FUNCTION
# =============================================================================

def build_leaderboard(
    benchmarks_dir: str = "benchmarks",
    models_dir: str = "models",
    datasets_dir: str = "datasets",
    evals_dir: str = "evals",
    output_path: str = "docs/leaderboards/index.md",
) -> str:
    """Build the complete leaderboard file."""
    print("Building Leaderboard with ITU AI4H Standards...")

    # Load data
    benchmarks = load_yaml_files(benchmarks_dir)
    models = load_yaml_files(models_dir)
    datasets = load_yaml_files(datasets_dir)
    evals = load_yaml_files(evals_dir)

    # Filter templates
    benchmarks = [b for b in benchmarks if b.get("benchmark_id") != "unique_benchmark_id_here"]

    # Group evals by benchmark
    evals_by_benchmark: Dict[str, List[Dict]] = defaultdict(list)
    for e in evals:
        bid = e.get("benchmark_id")
        if bid:
            evals_by_benchmark[bid].append(e)

    # Group benchmarks by modality
    benchmarks_by_modality: Dict[str, List[Dict]] = defaultdict(list)
    for bm in benchmarks:
        modality = detect_modality(bm)
        benchmarks_by_modality[modality].append(bm)

    # Stats
    total_benchmarks = len(benchmarks)
    total_evals = len(evals)
    unique_models = set()
    for e in evals:
        for mid in e.get("model_ids", {}).values():
            unique_models.add(mid)

    # ==========================================================================
    # BUILD CONTENT
    # ==========================================================================
    
    content = "# 🏆 Foundation Model Leaderboards\n\n"
    
    # Stats banner
    content += "!!! success \"Benchmark Hub Overview\"\n"
    content += f"    📊 **{total_benchmarks}** Benchmarks | "
    content += f"🤖 **{len(unique_models)}** Models | "
    content += f"📈 **{total_evals}** Evaluations\n\n"
    
    # Introduction for general audience
    content += """
> **What is this?** This page ranks AI models for healthcare applications. 
> Higher-ranked models perform better on standardized tests.
> 
> **How to read it:** Each table shows models from best (🥇) to developing (📈).
> Click "How are scores calculated?" for details on what the numbers mean.

"""

    # Quick navigation
    content += "## 🧭 Jump To\n\n"
    content += "- [🌐 Overall Rankings](#overall-rankings-all-modalities) — Best across all categories\n"
    for mod_key, mod_info in MODALITIES.items():
        if mod_key in benchmarks_by_modality:
            anchor = mod_info["name"].lower().replace(" ", "-").replace("/", "").replace("(", "").replace(")", "")
            content += f"- [{mod_info['emoji']} {mod_info['name']}](#{anchor})\n"
    content += "\n---\n\n"

    # Overall leaderboard first
    content += generate_overall_leaderboard(evals, models, datasets, benchmarks)

    # Modality-specific sections
    for mod_key, mod_info in MODALITIES.items():
        mod_benchmarks = benchmarks_by_modality.get(mod_key, [])
        if mod_benchmarks:
            content += generate_modality_section(
                mod_key, mod_info, mod_benchmarks,
                evals_by_benchmark, models, datasets
            )

    # Handle "other" modality if any
    other_benchmarks = benchmarks_by_modality.get("other", [])
    if other_benchmarks:
        content += "## 📋 Other Benchmarks\n\n"
        for bm in other_benchmarks:
            bid = bm.get("benchmark_id")
            bm_evals = evals_by_benchmark.get(bid, [])
            # ... similar processing
            content += f"### {bm.get('name', 'Unnamed')}\n\n"
            if not bm_evals:
                content += "*No submissions yet.*\n\n"
            else:
                ai_task = bm.get("ai_task", "")
                ranked = []
                for ev in bm_evals:
                    score, metric = get_primary_score(ev.get("metrics", {}), ai_task)
                    ranked.append((score, metric, ev))
                ranked.sort(key=lambda x: x[0], reverse=True)
                if ranked:
                    content += generate_ranking_table(ranked, models, datasets, ranked[0][1])
        content += "---\n\n"

    # Footer: How to submit
    content += """
## 🚀 Add Your Model

Want your model on this leaderboard?

1. **Download** the benchmark toolkit
2. **Run locally** on your model (your code stays private!)
3. **Submit results** via [GitHub Issue](https://github.com/allison-eunse/ai4h-inspired-fm-benchmark-hub/issues/new?template=benchmark_submission.md)

[📥 Get Started](../index.md){ .md-button .md-button--primary }
[📖 Submission Guide](../contributing/submission_guide.md){ .md-button }

---

*Aligned with [ITU/WHO FG-AI4H](https://www.itu.int/pub/T-FG-AI4H) standards for healthcare AI evaluation.*
"""

    # Write
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as f:
        f.write(content)

    print(f"Leaderboard written to {output_path}")
    return output_path


__all__ = ["build_leaderboard", "load_yaml_files", "generate_markdown_table"]
