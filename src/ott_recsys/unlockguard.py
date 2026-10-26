"""Guards for the one-shot test unlock.

This module is imported ONLY by scripts/unlock_test.py (a repository-wide static test in tests/test_module8.py fails if any
other file under src/ or scripts/ imports it or mentions the unlock script).

Rules enforced here:
  * the unlock log (reports/TEST_UNLOCK_LOG.json) must not exist in the working tree OR anywhere in git history;
  * a rerun after a crash is allowed only with --after-crash, only if the log says "started", and only if NO test metric
    was written or printed before the crash (the script sets test_metrics_emitted just before it emits any metric);
    otherwise the run counts as spent;
  * the pre-registration files must be committed and unchanged, and the working tree must be clean (real run);
  * the frozen artifacts must match the hashes recorded by scripts/prepare_test.py."""
import json
import os
import subprocess
import time
from pathlib import Path

from .hashing import hash_files, sha256_file  # noqa: F401  (re-exported for the unlock script)

LOG_NAME = "reports/TEST_UNLOCK_LOG.json"
REHEARSAL_LOG = "reports/rehearsal/UNLOCK_LOG_REHEARSAL.json"
PREREGISTERED = ("reports/module8_protocol.md", "reports/module8_predictions.md")


class UnlockRefused(RuntimeError):
    """Raised when a guard forbids the run."""


def _git(root: Path, *args: str) -> str:
    r = subprocess.run(["git", "-C", str(root), *args], capture_output=True, text=True)
    if r.returncode != 0:
        raise UnlockRefused(f"git {' '.join(args)} failed: {r.stderr.strip()}")
    return r.stdout


def in_git_history(root: Path, rel_path: str) -> bool:
    """True if any commit on any branch ever touched this path (deleting the file does not reopen the unlock)."""
    return bool(_git(root, "log", "--all", "--oneline", "--", rel_path).strip())


def read_log(path: Path):
    return json.loads(path.read_text()) if path.exists() else None


def check_not_spent(root: Path, log_rel: str, after_crash: bool, check_history: bool = True):
    """Return the previous log if a crash rerun is allowed, None for a first run; raise UnlockRefused otherwise."""
    path = Path(root) / log_rel
    prev = read_log(path)
    if prev is not None:
        if prev.get("status") == "done":
            raise UnlockRefused("the unlock already ran to completion; it can never be run again")
        if prev.get("test_metrics_emitted"):
            raise UnlockRefused("SPENT: a previous attempt emitted test metrics before it stopped; no rerun is allowed")
        if not after_crash:
            raise UnlockRefused("a previous attempt started but did not finish; rerun only with --after-crash "
                                "(allowed only because no test metric was written or printed)")
        return prev
    if check_history and in_git_history(Path(root), log_rel):
        raise UnlockRefused("the unlock log exists in git history: the unlock was already used (deleting the file does not reopen it)")
    if after_crash:
        raise UnlockRefused("--after-crash given but there is no previous started attempt")
    return None


def _write(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2, default=str)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def write_start_log(path: Path, info: dict, prev=None) -> dict:
    """First action of a run, BEFORE any test data is read."""
    log = {"status": "started", "test_metrics_emitted": False, "attempts": (prev["attempts"] + 1) if prev else 1,
           "first_started_at": prev["first_started_at"] if prev else time.strftime("%Y-%m-%d %H:%M:%S"),
           "last_started_at": time.strftime("%Y-%m-%d %H:%M:%S"), **info}
    _write(Path(path), log)
    return log


def mark_metrics_emitted(path: Path) -> None:
    """Called immediately before ANY test metric is printed or written; after this a crash means the run is spent."""
    log = read_log(Path(path))
    log["test_metrics_emitted"] = True
    _write(Path(path), log)


def mark_done(path: Path, report_path: str, report_sha256: str) -> None:
    log = read_log(Path(path))
    log.update({"status": "done", "finished_at": time.strftime("%Y-%m-%d %H:%M:%S"), "report": report_path, "report_sha256": report_sha256})
    _write(Path(path), log)


def clean_tree(root: Path) -> None:
    out = _git(Path(root), "status", "--porcelain").strip()
    if out:
        raise UnlockRefused("the working tree is not clean:\n" + out)


def preregistration(root: Path, files=PREREGISTERED) -> dict:
    """Every pre-registration file must be tracked, unmodified, and committed; returns their last-commit hashes."""
    root, out = Path(root), {}
    for f in files:
        if not (root / f).exists():
            raise UnlockRefused(f"pre-registration file missing: {f}")
        if not _git(root, "ls-files", "--", f).strip():
            raise UnlockRefused(f"pre-registration file is not committed: {f}")
        if _git(root, "status", "--porcelain", "--", f).strip():
            raise UnlockRefused(f"pre-registration file changed since its commit: {f}")
        out[f] = _git(root, "log", "-1", "--format=%H", "--", f).strip()
    return out


def verify_manifest(manifest_path: Path) -> dict:
    """The manifest written by scripts/prepare_test.py maps artifact path -> sha256; every file must still match."""
    m = json.loads(Path(manifest_path).read_text())
    bad = [p for p, h in m["artifacts"].items() if not Path(p).exists() or sha256_file(Path(p)) != h]
    if bad:
        raise UnlockRefused("frozen artifacts changed or missing since preparation: " + ", ".join(bad[:5]))
    return m
