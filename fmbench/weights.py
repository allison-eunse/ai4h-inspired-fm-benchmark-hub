"""
Weights management utilities (non-git, local cache).

Goal:
- Allow downloading model weights from HuggingFace / URLs into a user cache directory.
- Keep weights OUT of the git repository.
"""

from __future__ import annotations

import os
import hashlib
import shutil
import tarfile
import zipfile
from urllib.parse import urlparse
from urllib.request import Request, urlopen
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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _guess_filename_from_url(url: str) -> str:
    parsed = urlparse(url)
    name = Path(parsed.path).name
    return name or "weights.bin"


def ensure_url_download(
    url: str,
    *,
    sha256: Optional[str] = None,
    local_dir_name: Optional[str] = None,
    filename: Optional[str] = None,
    extract: bool = False,
) -> Path:
    """
    Download a checkpoint from a direct URL into fmbench weights cache.

    - Stores under: ~/.cache/fmbench/weights/<local_dir_name>/...
    - Optionally verifies sha256
    - Optionally extracts .zip / .tar(.gz) archives

    Returns:
      - extracted directory path if extract=True
      - downloaded file path otherwise
    """
    weights_dir = get_weights_dir()
    safe_name = local_dir_name or urlparse(url).netloc.replace(":", "_")
    local_dir = weights_dir / safe_name
    local_dir.mkdir(parents=True, exist_ok=True)

    fname = filename or _guess_filename_from_url(url)
    target = local_dir / fname

    # If already exists and hash matches, reuse
    if target.exists() and sha256:
        existing = _sha256_file(target)
        if existing.lower() == sha256.lower():
            return local_dir if extract else target

    req = Request(url, headers={"User-Agent": "fmbench/0.1 (weights downloader)"})
    with urlopen(req) as resp:
        with target.open("wb") as out:
            shutil.copyfileobj(resp, out)

    if sha256:
        got = _sha256_file(target)
        if got.lower() != sha256.lower():
            raise RuntimeError(
                f"SHA256 mismatch for {target}.\n"
                f"Expected: {sha256}\n"
                f"Got:      {got}"
            )

    if extract:
        # Extract into local_dir/<archive_stem>/
        extract_dir = local_dir / (target.stem.replace(".tar", ""))
        extract_dir.mkdir(parents=True, exist_ok=True)

        if zipfile.is_zipfile(target):
            with zipfile.ZipFile(target, "r") as zf:
                zf.extractall(extract_dir)
        else:
            # tar / tar.gz / tgz
            try:
                with tarfile.open(target, "r:*") as tf:
                    tf.extractall(extract_dir)
            except tarfile.TarError as e:
                raise RuntimeError(
                    f"File {target} is not a supported archive for extract=True"
                ) from e
        return extract_dir

    return target


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
      - hf_repo_id: "org/repo"
      - hf_revision: "main" (optional)
    Or provide a direct URL:
      - weights_url: "https://.../checkpoint.pt"
      - weights_sha256: "<hex>" (recommended)
      - weights_extract: true (optional, for archives)
    """
    model_cfg = model_cfg or {}

    # Config override
    if "weights_url" in model_cfg:
        return {
            "type": "url",
            "url": model_cfg["weights_url"],
            "sha256": model_cfg.get("weights_sha256"),
            "extract": bool(model_cfg.get("weights_extract", False)),
        }
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

