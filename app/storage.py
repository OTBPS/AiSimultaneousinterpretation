"""Keep the install footprint bounded and self-cleaning.

Requirement: updates overwrite in place and must never accumulate disk usage.
Three things in this app grow on their own if left alone:

1. `.ovcache` -- OpenVINO writes a compiled blob per (model, device, config).
   It is 3.1 GB for the two current models, and every model swap or driver
   update silently adds another set instead of replacing the old one.
2. logs.
3. nothing else: models live outside the project in D:\\AI\\Models and are
   referenced by path, never copied in.

`prune_cache` enforces a byte budget by deleting least-recently-used blobs,
and `invalidate_stale_cache` wipes the whole cache when the OpenVINO version
changes, because blobs from a previous runtime are dead weight that will never
be read again.
"""
from __future__ import annotations

import json
import logging
import shutil
from pathlib import Path

log = logging.getLogger("storage")

DEFAULT_CACHE_BUDGET_GB = 8.0
STAMP = "cache_stamp.json"


def dir_size(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _runtime_fingerprint() -> dict[str, str]:
    try:
        import openvino as ov
        ov_version = str(ov.__version__)
    except Exception:                                     # noqa: BLE001
        ov_version = "unknown"
    gpu = "unknown"
    try:
        import openvino as ov
        core = ov.Core()
        if "GPU" in core.available_devices:
            gpu = str(core.get_property("GPU", "FULL_DEVICE_NAME"))
    except Exception:                                     # noqa: BLE001
        pass
    return {"openvino": ov_version, "gpu": gpu}


def invalidate_stale_cache(cache_dir: Path) -> bool:
    """Drop the cache if the OpenVINO runtime or GPU changed.

    Old blobs are not reused after a runtime upgrade; without this they just
    sit there forever, which is exactly the accumulation we promised not to do.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stamp_path = cache_dir / STAMP
    current = _runtime_fingerprint()

    previous = None
    if stamp_path.exists():
        try:
            previous = json.loads(stamp_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            previous = None

    if previous is not None and previous == current:
        return False

    if previous is not None:
        freed = dir_size(cache_dir) / 1e9
        for item in cache_dir.iterdir():
            if item.name == STAMP:
                continue
            try:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            except OSError as e:
                log.warning("cannot remove stale cache entry %s: %s", item, e)
        log.info("runtime changed (%s -> %s); cleared %.1f GB of stale cache",
                 previous, current, freed)

    try:
        stamp_path.write_text(json.dumps(current, indent=2), encoding="utf-8")
    except OSError as e:
        log.warning("cannot write cache stamp: %s", e)
    return previous is not None


def prune_cache(cache_dir: Path, budget_gb: float = DEFAULT_CACHE_BUDGET_GB) -> int:
    """Evict least-recently-used blobs until the cache fits the budget.

    Returns the number of bytes freed. Safe to call at startup: a blob deleted
    by mistake only costs one recompile.
    """
    if not cache_dir.exists():
        return 0
    budget = int(budget_gb * 1e9)
    files = [f for f in cache_dir.rglob("*") if f.is_file() and f.name != STAMP]
    total = sum(f.stat().st_size for f in files)
    if total <= budget:
        return 0

    files.sort(key=lambda f: f.stat().st_atime)           # oldest access first
    freed = 0
    for f in files:
        if total - freed <= budget:
            break
        try:
            size = f.stat().st_size
            f.unlink()
            freed += size
        except OSError:
            continue
    if freed:
        log.info("pruned %.1f GB from the model cache (budget %.1f GB)",
                 freed / 1e9, budget_gb)
    return freed


def prepare(cache_dir: Path, budget_gb: float = DEFAULT_CACHE_BUDGET_GB) -> None:
    """Call once at startup, before any pipeline is constructed."""
    invalidate_stale_cache(cache_dir)
    prune_cache(cache_dir, budget_gb)
