"""Model discovery for the panel's model picker: local cache + Hub search.

Both `local_models` (already on disk) and `search_hub_models` (Hugging Face
Hub lookup) only ever surface information — neither downloads anything. #33
lists a download manager as an explicit non-goal, and this stays on the right
side of that line: it is autocomplete for the free-text field, not a hub.
"""

from __future__ import annotations

from typing import Any

#: Repos that are cached but are not MLX text models we can serve. Filtering
#: these out keeps the picker from suggesting a model that will fail at load.
_EXCLUDE_MARKERS = (
    "clip",
    "bge",
    "bert",
    "whisper",
    "embed",
    "rerank",
    "vae",
    "sentence-transformers",
)


def _looks_servable(repo_id: str) -> bool:
    lowered = repo_id.lower()
    if any(marker in lowered for marker in _EXCLUDE_MARKERS):
        return False
    # VLMs load through mlx_vlm, not the text path this server uses.
    if "-vl-" in lowered or lowered.endswith("-vl"):
        return False
    return True


def local_models() -> list[dict[str, Any]]:
    """Cached models, largest-signal first (MLX community repos, then size).

    Returns ``[]`` on any failure. A missing or unreadable Hugging Face cache
    is normal — the model field accepts free text, so an empty picker costs the
    user nothing, while an exception here would break the whole panel.
    """
    try:
        from huggingface_hub import scan_cache_dir
    except ImportError:
        return []

    try:
        cache = scan_cache_dir()
    except Exception:
        return []

    models: list[dict[str, Any]] = []
    for repo in cache.repos:
        if getattr(repo, "repo_type", "model") != "model":
            continue
        if not _looks_servable(repo.repo_id):
            continue

        models.append(
            {
                "repo_id": repo.repo_id,
                "size_bytes": int(repo.size_on_disk),
                "size_label": _human_size(repo.size_on_disk),
                "is_mlx": "mlx-community/" in repo.repo_id.lower(),
            }
        )

    models.sort(key=lambda m: (not m["is_mlx"], -m["size_bytes"]))
    return models


def _human_size(num_bytes: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(num_bytes) < 1024:
            return f"{num_bytes:.0f} {unit}" if unit == "B" else f"{num_bytes:.1f} {unit}"
        num_bytes /= 1024
    return f"{num_bytes:.1f} PB"


def search_hub_models(query: str, limit: int = 20) -> list[dict[str, Any]]:
    """Search the Hugging Face Hub for text-generation models matching ``query``.

    This is discovery only, same as :func:`local_models` — it never downloads
    anything. Selecting a result just fills the free-text model field; the
    existing start-server path handles fetching the weights if needed.

    Returns ``[]`` on any failure (offline, rate-limited, huggingface_hub
    missing, or an empty/whitespace query) so a flaky network never breaks the
    panel.
    """
    query = query.strip()
    if not query:
        return []

    try:
        from huggingface_hub import HfApi
    except ImportError:
        return []

    try:
        api = HfApi()
        results = api.list_models(
            search=query,
            filter="text-generation",
            sort="downloads",
            limit=limit,
        )

        models: list[dict[str, Any]] = []
        for repo in results:
            repo_id = repo.id
            if not _looks_servable(repo_id):
                continue
            models.append(
                {
                    "repo_id": repo_id,
                    "downloads": getattr(repo, "downloads", None),
                    "is_mlx": "mlx-community/" in repo_id.lower(),
                }
            )
        return models
    except Exception:
        return []
