"""The install footprint must stay bounded without manual cleanup."""
import json
import time

from app import storage


def _blob(path, name, size, atime=None):
    f = path / name
    f.write_bytes(b"\0" * size)
    if atime is not None:
        import os
        os.utime(f, (atime, atime))
    return f


def test_dir_size_counts_files(tmp_path):
    _blob(tmp_path, "a.bin", 1000)
    _blob(tmp_path, "b.bin", 2000)
    assert storage.dir_size(tmp_path) == 3000


def test_dir_size_of_missing_dir_is_zero(tmp_path):
    assert storage.dir_size(tmp_path / "nope") == 0


def test_first_run_writes_stamp_and_keeps_cache(tmp_path):
    _blob(tmp_path, "blob.bin", 1000)
    wiped = storage.invalidate_stale_cache(tmp_path)
    assert wiped is False, "nothing to invalidate on a first run"
    assert (tmp_path / storage.STAMP).exists()
    assert (tmp_path / "blob.bin").exists()


def test_unchanged_runtime_keeps_cache(tmp_path):
    storage.invalidate_stale_cache(tmp_path)
    _blob(tmp_path, "blob.bin", 1000)
    assert storage.invalidate_stale_cache(tmp_path) is False
    assert (tmp_path / "blob.bin").exists()


def test_changed_runtime_wipes_cache(tmp_path):
    storage.invalidate_stale_cache(tmp_path)
    _blob(tmp_path, "blob.bin", 1000)
    (tmp_path / storage.STAMP).write_text(
        json.dumps({"openvino": "1900.1", "gpu": "some other gpu"}),
        encoding="utf-8")

    assert storage.invalidate_stale_cache(tmp_path) is True
    assert not (tmp_path / "blob.bin").exists()
    assert (tmp_path / storage.STAMP).exists(), "stamp must be rewritten"


def test_prune_is_a_noop_under_budget(tmp_path):
    _blob(tmp_path, "a.bin", 1000)
    assert storage.prune_cache(tmp_path, budget_gb=1.0) == 0
    assert (tmp_path / "a.bin").exists()


def test_prune_evicts_least_recently_used(tmp_path):
    now = time.time()
    _blob(tmp_path, "old.bin", 600_000, atime=now - 10_000)
    _blob(tmp_path, "new.bin", 600_000, atime=now)

    freed = storage.prune_cache(tmp_path, budget_gb=0.001)   # 1 MB budget
    assert freed >= 600_000
    assert not (tmp_path / "old.bin").exists(), "LRU victim should go first"
    assert (tmp_path / "new.bin").exists()


def test_prune_never_deletes_the_stamp(tmp_path):
    storage.invalidate_stale_cache(tmp_path)
    _blob(tmp_path, "big.bin", 2_000_000)
    storage.prune_cache(tmp_path, budget_gb=0.0000001)
    assert (tmp_path / storage.STAMP).exists()


def test_prepare_is_idempotent(tmp_path):
    storage.prepare(tmp_path, budget_gb=8.0)
    size1 = storage.dir_size(tmp_path)
    storage.prepare(tmp_path, budget_gb=8.0)
    assert storage.dir_size(tmp_path) == size1
