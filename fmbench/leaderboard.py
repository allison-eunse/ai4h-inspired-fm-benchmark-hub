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
# METRIC EXPLANATIONS (Human-readable, beginner-friendly)
# =============================================================================

METRIC_EXPLANATIONS = {
    # Classification metrics
    "AUROC": {
        "name": "Area Under ROC Curve (AUROC)",
        "simple": "Measures how well the model can tell apart different categories (e.g., healthy vs. diseased)",
        "detailed": (
            "Think of it like this: if you randomly pick one positive case and one negative case, "
            "AUROC tells you the probability that the model correctly identifies which is which. "
            "A score of 0.5 means the model is just guessing randomly (like flipping a coin), "
            "while 1.0 means it perfectly separates all cases."
        ),
        "range": "0.5 (random guessing) → 1.0 (perfect separation)",
        "example": "An AUROC of 0.85 means the model correctly ranks a positive case higher than a negative case 85% of the time.",
        "good_threshold": 0.85,
    },
    "Accuracy": {
        "name": "Accuracy",
        "simple": "The percentage of predictions the model got right",
        "detailed": (
            "This is the most intuitive metric: out of all the predictions the model made, "
            "how many were correct? For example, if a model makes 100 predictions and 90 are correct, "
            "the accuracy is 90% (or 0.90). While easy to understand, accuracy can be misleading "
            "when classes are imbalanced (e.g., if 95% of cases are healthy, a model that always "
            "predicts 'healthy' would have 95% accuracy but miss all diseases)."
        ),
        "range": "0.0 (all wrong) → 1.0 (all correct)",
        "example": "An accuracy of 0.92 means the model correctly classified 92 out of every 100 samples.",
        "good_threshold": 0.85,
    },
    "F1-Score": {
        "name": "F1 Score",
        "simple": "A balanced measure that considers both false alarms and missed cases",
        "detailed": (
            "F1 Score balances two important aspects: (1) Precision — when the model says 'positive', "
            "how often is it right? (2) Recall — out of all actual positives, how many did the model find? "
            "F1 is the harmonic mean of these two, so it's high only when both are good. "
            "This is especially useful in healthcare where both false alarms (unnecessary worry/treatment) "
            "and missed cases (delayed diagnosis) have real consequences."
        ),
        "range": "0.0 (poor) → 1.0 (perfect balance of precision and recall)",
        "example": "An F1 of 0.85 indicates the model has a good balance between catching real cases and avoiding false alarms.",
        "good_threshold": 0.80,
    },
    "Correlation": {
        "name": "Correlation",
        "simple": "How closely the model's predictions match the actual values",
        "detailed": (
            "Correlation measures the strength and direction of the relationship between predicted "
            "and actual values. A correlation of 1.0 means perfect positive agreement (when actual "
            "goes up, prediction goes up proportionally), while 0 means no relationship at all. "
            "This is commonly used for reconstruction tasks where we want to see how well the model "
            "can recreate the original signal."
        ),
        "range": "-1.0 (perfect inverse) → 0 (no relationship) → 1.0 (perfect match)",
        "example": "A correlation of 0.78 means the model's outputs track reasonably well with the true values.",
        "good_threshold": 0.70,
    },
    
    # Robustness metrics
    "robustness_score": {
        "name": "Robustness Score",
        "simple": "How stable and reliable the model is when data quality isn't perfect",
        "detailed": (
            "Real-world data is messy: sensors fail, signals have noise, and recordings have artifacts. "
            "The robustness score measures how well a model maintains its performance when we "
            "deliberately add these imperfections. A highly robust model gives consistent results "
            "even with noisy or incomplete data, which is critical for clinical deployment where "
            "data quality varies between hospitals and equipment."
        ),
        "range": "0.0 (performance collapses with any noise) → 1.0 (completely stable)",
        "example": "A robustness score of 0.82 means the model maintains most of its accuracy even when data has noise or missing values.",
        "good_threshold": 0.75,
    },
    
    # Report generation metrics
    "report_quality_score": {
        "name": "Report Quality Score",
        "simple": "An overall measure of how good the AI-generated medical reports are",
        "detailed": (
            "This composite score combines multiple aspects of report quality: clinical accuracy "
            "(are the findings correct?), completeness (are important findings mentioned?), "
            "language quality (is it well-written?), and safety (no harmful content). "
            "It provides a single number to compare models, though looking at individual components "
            "gives more insight into specific strengths and weaknesses."
        ),
        "range": "0.0 (poor quality) → 1.0 (excellent quality)",
        "example": "A score of 0.85 indicates the model generates reports that are mostly accurate, complete, and well-structured.",
        "good_threshold": 0.80,
    },
    "clinical_accuracy": {
        "name": "Clinical Accuracy", 
        "simple": "Are the medical findings in the generated report actually correct?",
        "detailed": (
            "This measures the factual correctness of medical statements in AI-generated reports. "
            "Expert clinicians review the reports and check whether each finding matches the ground truth. "
            "In healthcare, this is perhaps the most critical metric — an inaccurate finding could lead "
            "to wrong diagnoses or treatment decisions."
        ),
        "range": "0.0 (all findings wrong) → 1.0 (all findings correct)",
        "example": "A clinical accuracy of 0.92 means 92% of the medical findings in the report are verified as correct.",
        "good_threshold": 0.90,
    },
    "hallucination_rate": {
        "name": "Hallucination Rate",
        "simple": "How often the AI makes up information that isn't supported by the input data",
        "detailed": (
            "AI models can sometimes generate plausible-sounding but completely fabricated content — "
            "called 'hallucinations'. In medical reports, this is dangerous: the AI might mention "
            "a finding that doesn't exist in the image. This metric measures how often this happens. "
            "Unlike most metrics, LOWER is better here — we want as few hallucinations as possible."
        ),
        "range": "0.0 (no hallucinations — ideal) → 1.0 (everything is made up)",
        "example": "A hallucination rate of 0.05 means only 5% of generated content is unsupported by the input — quite good!",
        "good_threshold": 0.05,
        "lower_is_better": True,
    },
    "bertscore": {
        "name": "BERTScore",
        "simple": "How similar the generated text is to the reference text in meaning (not just exact words)",
        "detailed": (
            "BERTScore uses AI to compare the meaning of generated reports against reference reports "
            "written by experts. Unlike simple word-matching, it understands that 'cardiac enlargement' "
            "and 'enlarged heart' mean the same thing. This makes it better at evaluating whether "
            "the AI captured the right medical concepts, even if it used different phrasing."
        ),
        "range": "0.0 (completely different meaning) → 1.0 (semantically identical)",
        "example": "A BERTScore of 0.87 indicates the generated report conveys very similar clinical meaning to the expert reference.",
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
    """Generate clean, readable scoring methodology with beginner-friendly explanations."""
    md = "\n<details>\n<summary>📐 <strong>How are scores calculated?</strong> (click to expand)</summary>\n\n"
    
    # Introduction for beginners
    md += "---\n\n"
    md += "### 📖 Understanding This Leaderboard\n\n"
    md += "This section explains how we measure and compare AI models. "
    md += "Don't worry if you're new to AI metrics — we'll break it down step by step.\n\n"
    
    # Primary metric card with detailed explanation
    metric_info = METRIC_EXPLANATIONS.get(primary_metric, {})
    md += "---\n\n"
    md += f"### 🎯 The Main Metric: `{primary_metric}`\n\n"
    
    if metric_info:
        md += f"**{metric_info.get('name', primary_metric)}**\n\n"
        md += f"**In simple terms:** {metric_info.get('simple', 'A performance measure')}\n\n"
        
        # Add detailed explanation if available
        if metric_info.get('detailed'):
            md += f"**How it works:** {metric_info.get('detailed')}\n\n"
        
        md += f"**Score range:** {metric_info.get('range', '0 to 1')}\n\n"
        
        # Add example if available
        if metric_info.get('example'):
            md += f"💡 **Example:** {metric_info.get('example')}\n\n"
    else:
        md += f"This metric measures model performance. Higher values generally indicate better performance.\n\n"
    
    # Task-specific context
    md += "---\n\n"
    md += "### 🧠 How This Metric Fits This Task\n\n"
    md += "Different tasks emphasize different aspects of performance. Here's how this metric should be interpreted for this benchmark:\n\n"
    lower_ai_task = (ai_task or "").lower()
    if "generation" in lower_ai_task:
        md += (
            "- For **report generation**, we care not only about language quality but also clinical safety.\n"
            "  This metric is usually combined with others (e.g., clinical accuracy, hallucination rate,\n"
            "  and completeness of findings) to judge whether the generated report is both readable **and**\n"
            "  medically reliable.\n\n"
        )
    elif "robustness" in lower_ai_task:
        md += (
            "- For **robustness assessment**, this metric summarizes how much the model's outputs change\n"
            "  when we add realistic perturbations (e.g., noise, signal dropout, channel permutations,\n"
            "  or temporal shifts). A higher score means the model is more stable under these stress tests.\n"
            "  In this benchmark, we probe robustness using:\n"
            "  - **Dropout**: randomly masking channels or features to mimic sensor failure or missing data\n"
            "  - **Gaussian noise**: adding noise at different signal-to-noise ratios (SNR) to simulate\n"
            "    electrical or physiological noise\n"
            "  - **Line noise**: injecting 50/60 Hz interference similar to mains power artifacts\n"
            "  - **Channel permutation**: shuffling channels to test invariance to channel ordering\n"
            "  - **Temporal shifts**: misaligning signals in time to simulate timing jitter\n"
            "  This reflects real-world variability between scanners, hospitals, and acquisition protocols.\n\n"
        )
    elif "reconstruction" in lower_ai_task and "classification" not in lower_ai_task or "regression" in lower_ai_task:
        md += (
            "- For **regression / continuous prediction** tasks, this metric captures how closely the model's\n"
            "  predicted values track the true values over a range (e.g., symptom severity, signal amplitude).\n"
            "  We are usually interested in both overall fit (correlation) and error magnitude.\n\n"
        )
    else:
        md += (
            "- For **classification** tasks (e.g., disease vs. no disease), this metric helps you understand\n"
            "  how reliably the model separates different outcome groups. In addition to raw accuracy,\n"
            "  we recommend also looking at metrics like AUROC and F1 Score, especially when classes are\n"
            "  imbalanced (for example, when positive cases are rare).\n\n"
        )
    
    # Performance tiers - with more context
    md += "---\n\n"
    md += "### 📊 Performance Tiers: What Do the Scores Mean?\n\n"
    md += "We group models into performance tiers to help you quickly understand how ready they are for different uses:\n\n"
    md += "| Score Range | Rating | Interpretation | Suitable For |\n"
    md += "|:---:|:---:|:---|:---|\n"
    md += "| **≥ 0.90** | ⭐ Excellent | Top-tier performance, consistently reliable | Clinical pilots with physician oversight |\n"
    md += "| **0.80 – 0.89** | ✅ Good | Strong performance, shows real promise | Validation studies, controlled testing |\n"
    md += "| **0.70 – 0.79** | 🔶 Fair | Moderate performance, has limitations | Research and development only |\n"
    md += "| **< 0.70** | 📈 Developing | Below typical benchmarks, needs improvement | Early research, not for clinical use |\n\n"
    
    md += "!!! tip \"Important Context\"\n"
    md += "    These thresholds are general guidelines. The acceptable score depends on the specific "
    md += "clinical application, risk level, and whether the AI assists or replaces human judgment. "
    md += "Always consult domain experts when evaluating fitness for a particular use case.\n\n"
    
    # Ranking rules - more detailed
    md += "---\n\n"
    md += "### 📏 How We Determine Rankings\n\n"
    md += "Models are ranked following these principles:\n\n"
    md += "1. **Primary metric determines rank** — The model with the highest score in the main metric ranks first. "
    md += "For metrics where lower is better (like error rates), the lowest score wins.\n\n"
    md += "2. **Ties are broken by secondary metrics** — If two models have identical primary scores, "
    md += "we look at other relevant metrics to determine which performs better overall.\n\n"
    md += "3. **Best run per model** — If a model was evaluated multiple times (e.g., with different settings), "
    md += "only its best result appears on the leaderboard. This ensures fair comparison.\n\n"
    md += "4. **Reproducibility required** — All results must be reproducible. We record the evaluation date, "
    md += "dataset used, and configuration to ensure transparency.\n\n"
    
    # Why this matters
    md += "---\n\n"
    md += "### 🏥 Why This Matters for Healthcare AI\n\n"
    md += "Healthcare AI has higher stakes than many other AI applications. A model that works "
    md += "95% of the time might sound good, but that 5% could mean missed diagnoses or incorrect treatments. "
    md += "That's why we:\n\n"
    md += "- Use **multiple metrics** to capture different aspects of performance\n"
    md += "- Test **robustness** to real-world data quality issues\n"
    md += "- Require **transparency** about evaluation conditions\n"
    md += "- Follow **international standards** for healthcare AI assessment\n\n"
    
    # AI4H note
    md += "---\n\n"
    md += "### 🌍 Standards Alignment\n\n"
    md += "This benchmark follows the [ITU/WHO Focus Group on AI for Health (FG-AI4H)](https://www.itu.int/pub/T-FG-AI4H) "
    md += "framework, which provides internationally recognized guidelines for evaluating healthcare AI systems. "
    md += "This ensures our evaluations are:\n\n"
    md += "- **Rigorous** — Following established scientific methodology\n"
    md += "- **Comparable** — Using standardized metrics across different models\n"
    md += "- **Trustworthy** — Aligned with WHO/ITU recommendations for health AI\n\n"
    
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
        dataset_id = ev.get("dataset_id")
        if dataset_id:
            dataset_data = get_by_id(datasets, "dataset_id", dataset_id)
            dataset_name = (dataset_data.get("name", dataset_id) if dataset_data else dataset_id)
        else:
            # Fallback: derive a short label from data_dir if no dataset_id
            data_dir = ev.get("data_dir", "")
            if data_dir:
                # Extract a meaningful short name from the path
                # e.g., "toy_data/neuro/robustness" -> "neuro/robustness"
                parts = data_dir.replace("\\", "/").split("/")
                # Skip common prefixes like "toy_data"
                parts = [p for p in parts if p and p not in ("toy_data", "data")]
                dataset_name = "/".join(parts[-2:]) if len(parts) >= 2 else (parts[-1] if parts else "N/A")
            else:
                dataset_name = "N/A"
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
                
                # Deduplicate by model (keep best score per model)
                seen = set()
                unique_ranked = []
                for item in ranked:
                    mid = list(item[2].get("model_ids", {}).values())[0] if item[2].get("model_ids") else ""
                    if mid not in seen:
                        seen.add(mid)
                        unique_ranked.append(item)
                
                if unique_ranked:
                    content += generate_ranking_table(unique_ranked, models, datasets, unique_ranked[0][1])
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
