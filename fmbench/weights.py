"""
Weights management utilities (non-git, local cache).

Goal:
- Allow downloading model weights from HuggingFace / URLs into a user cache directory.
- Keep weights OUT of the git repository.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Dict, Optional


def get_fmbench_cache_dir() -> Path:
    """
    Cache directory used by fmbench for weights and large artifacts.

    Priority:
    1) env FMBENCH_CACHE_DIR
    2) ~/.cache/fmbench
    """
    env = os.environ.get("FMBENCH_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (Path.home() / ".cache" / "fmbench").resolve()


def get_weights_dir() -> Path:
    d = get_fmbench_cache_dir() / "weights"
    d.mkdir(parents=True, exist_ok=True)
    return d


def ensure_hf_snapshot(
    repo_id: str,
    *,
    revision: Optional[str] = None,
    local_dir_name: Optional[str] = None,
    allow_patterns: Optional[list[str]] = None,
) -> Path:
    """
    Download a HuggingFace snapshot into fmbench weights cache.

    Returns the local directory path.
    """
    try:
        from huggingface_hub import snapshot_download
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "huggingface_hub is required to download weights. "
            "Install with: pip install -e \".[models]\""
        ) from e

    weights_dir = get_weights_dir()
    safe_name = local_dir_name or repo_id.replace("/", "__")
    local_dir = weights_dir / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)

    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=str(local_dir),
        local_dir_use_symlinks=False,
        allow_patterns=allow_patterns,
    )
    return local_dir


DEFAULT_WEIGHT_SOURCES: Dict[str, Dict] = {
    # DNA models (HuggingFace)
    "dnabert2": {"type": "hf", "repo_id": "zhihan1996/DNABERT-2-117M"},
    "hyenadna": {"type": "hf", "repo_id": "LongSafari/hyenadna-small-32k-seqlen-hf"},
    "evo2": {"type": "hf", "repo_id": "arcinstitute/evo2_1b_base"},
    "caduceus": {"type": "hf", "repo_id": "kuleshov-group/caduceus-ps_seqlen-131k_d_model-256_n_layer-16"},

    # scRNA-seq
    "geneformer": {"type": "hf", "repo_id": "ctheodoris/Geneformer"},
}


def resolve_weights_source(adapter_name: str, model_cfg: Optional[Dict] = None) -> Dict:
    """
    Resolve a weight source dict for an adapter.

    Users can override defaults by setting in config:
      hf_repo_id: "org/repo"
      hf_revision: "main" (optional)
    """
    model_cfg = model_cfg or {}

    # Config override
    if "hf_repo_id" in model_cfg:
        return {
            "type": "hf",
            "repo_id": model_cfg["hf_repo_id"],
            "revision": model_cfg.get("hf_revision"),
        }

    src = DEFAULT_WEIGHT_SOURCES.get(adapter_name)
    if not src:
        raise ValueError(
            f"No default weights source registered for adapter {adapter_name!r}. "
            "Provide weights manually via checkpoint_path, or add `hf_repo_id` to your model config."
        )
    return src

