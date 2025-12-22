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
from urllib.parse import urlparse, urlencode, parse_qsl, urlsplit, urlunsplit
from urllib.request import Request, urlopen, build_opener, HTTPCookieProcessor
from http.cookiejar import CookieJar
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


def _is_google_drive_url(url: str) -> bool:
    host = urlparse(url).netloc.lower()
    return host in {"drive.google.com", "drive.usercontent.google.com", "docs.google.com"}


def _append_query(url: str, extra: dict) -> str:
    parts = urlsplit(url)
    q = dict(parse_qsl(parts.query, keep_blank_values=True))
    q.update({k: str(v) for k, v in extra.items() if v is not None})
    return urlunsplit((parts.scheme, parts.netloc, parts.path, urlencode(q), parts.fragment))


def _open_with_drive_confirm(url: str):
    """
    Open a URL and transparently handle Google Drive's virus-scan warning page.

    Returns:
      (response, suggested_filename)
    """
    jar = CookieJar()
    opener = build_opener(HTTPCookieProcessor(jar))

    req = Request(url, headers={"User-Agent": "fmbench/0.1 (weights downloader)"})
    resp = opener.open(req)
    content_type = (getattr(resp, "headers", {}) or {}).get("Content-Type", "")

    # Drive may return an HTML interstitial (virus scan warning) for large files.
    if "text/html" in str(content_type).lower() and _is_google_drive_url(getattr(resp, "url", url)):
        html = resp.read().decode("utf-8", errors="replace")
        if 'id="download-form"' in html and 'name="confirm"' in html:
            import re

            action_m = re.search(r'action="([^"]+)"', html)
            id_m = re.search(r'name="id" value="([^"]+)"', html)
            export_m = re.search(r'name="export" value="([^"]+)"', html)
            confirm_m = re.search(r'name="confirm" value="([^"]+)"', html)
            uuid_m = re.search(r'name="uuid" value="([^"]+)"', html)

            if action_m and id_m and export_m and confirm_m:
                action = action_m.group(1)
                follow_url = _append_query(
                    action,
                    {
                        "id": id_m.group(1),
                        "export": export_m.group(1),
                        "confirm": confirm_m.group(1),
                        "uuid": uuid_m.group(1) if uuid_m else None,
                    },
                )
                resp2 = opener.open(
                    Request(follow_url, headers={"User-Agent": "fmbench/0.1 (weights downloader)"})
                )

                # Try infer filename from content-disposition
                cd = (getattr(resp2, "headers", {}) or {}).get("Content-Disposition")
                filename = None
                if cd and "filename=" in cd:
                    fn = cd.split("filename=", 1)[1].strip()
                    if fn.startswith('"') and fn.endswith('"'):
                        fn = fn[1:-1]
                    filename = fn
                return resp2, filename

        # If it's HTML but not the expected form, return it as a byte-stream so the caller
        # can persist it for debugging (hash/extract will likely fail downstream).
        import io

        return io.BytesIO(html.encode("utf-8")), None

    return resp, None


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

    if _is_google_drive_url(url):
        resp, drive_name = _open_with_drive_confirm(url)
        if (not filename) and drive_name:
            target = local_dir / drive_name
        with target.open("wb") as out:
            shutil.copyfileobj(resp, out)
    else:
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

    # Neuro (direct URLs: Google Drive file links)
    "brainjepa": {
        "type": "url",
        "url": "https://drive.google.com/uc?id=1jYfqOiKFBI95RLlDtPdomBr74zULg5HZ&export=download",
        "extract": False,
    },
    "brainharmony": {
        "type": "url",
        "url": "https://drive.google.com/uc?id=1gBptjXQJluuzBtV4y0IB_qBV1wSKWHUD&export=download",
        "extract": False,
    },
    "swift": {
        "type": "url",
        # SwiFT pretrained_models/contrastive_pretrained.ckpt (Drive)
        "url": "https://drive.google.com/uc?id=11u4GGeTB361X01sge86U7JbGyEzZC7KJ&export=download",
        "extract": False,
    },
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

