"""Module 8 guard tests: the one-shot unlock cannot run twice, a crash rerun is allowed only before any metric was
emitted, deleting the log does not reopen it, and pre-registration / frozen artifacts must be intact."""
import json
import subprocess

import pytest

from ott_recsys import unlockguard as g


def _repo(tmp_path):
    def run(*a):
        subprocess.run(["git", "-C", str(tmp_path), *a], check=True, capture_output=True)
    run("init", "-q")
    run("config", "user.email", "t@example.com")
    run("config", "user.name", "t")
    (tmp_path / "reports").mkdir()
    return tmp_path, run


LOG = g.LOG_NAME


def test_first_run_ok_then_done_run_is_refused_forever(tmp_path):
    root, _ = _repo(tmp_path)
    assert g.check_not_spent(root, LOG, after_crash=False) is None
    log = g.write_start_log(root / LOG, {"git": "abc"})
    assert log["status"] == "started" and log["attempts"] == 1 and log["test_metrics_emitted"] is False
    g.mark_done(root / LOG, "reports/x.txt", "deadbeef")
    for flag in (False, True):
        with pytest.raises(g.UnlockRefused, match="already ran"):
            g.check_not_spent(root, LOG, after_crash=flag)


def test_crash_before_any_metric_allows_exactly_a_flagged_rerun(tmp_path):
    root, _ = _repo(tmp_path)
    g.write_start_log(root / LOG, {})
    with pytest.raises(g.UnlockRefused, match="--after-crash"):
        g.check_not_spent(root, LOG, after_crash=False)               # a bare rerun is refused
    prev = g.check_not_spent(root, LOG, after_crash=True)              # no metric was emitted -> allowed
    again = g.write_start_log(root / LOG, {}, prev)
    assert again["attempts"] == 2 and again["first_started_at"] == prev["first_started_at"]


def test_crash_after_a_metric_was_emitted_counts_as_spent(tmp_path):
    root, _ = _repo(tmp_path)
    g.write_start_log(root / LOG, {})
    g.mark_metrics_emitted(root / LOG)
    for flag in (False, True):
        with pytest.raises(g.UnlockRefused, match="SPENT"):
            g.check_not_spent(root, LOG, after_crash=flag)


def test_after_crash_without_a_previous_attempt_is_refused(tmp_path):
    root, _ = _repo(tmp_path)
    with pytest.raises(g.UnlockRefused, match="no previous"):
        g.check_not_spent(root, LOG, after_crash=True)


def test_deleting_the_log_does_not_reopen_the_unlock(tmp_path):
    root, run = _repo(tmp_path)
    g.write_start_log(root / LOG, {})
    g.mark_done(root / LOG, "r", "h")
    run("add", LOG)
    run("commit", "-q", "-m", "log")
    (root / LOG).unlink()
    run("add", "-A")
    run("commit", "-q", "-m", "delete the log")
    assert not (root / LOG).exists()
    with pytest.raises(g.UnlockRefused, match="git history"):
        g.check_not_spent(root, LOG, after_crash=False)
    assert g.check_not_spent(root, LOG, after_crash=False, check_history=False) is None   # rehearsal logs skip the history rule


def test_log_is_flushed_before_metrics_flag_and_survives_a_reload(tmp_path):
    root, _ = _repo(tmp_path)
    g.write_start_log(root / LOG, {"commit": "c1"})
    assert json.loads((root / LOG).read_text())["test_metrics_emitted"] is False
    g.mark_metrics_emitted(root / LOG)
    assert g.read_log(root / LOG)["test_metrics_emitted"] is True and g.read_log(root / LOG)["commit"] == "c1"


def test_preregistration_must_be_committed_and_unchanged(tmp_path):
    root, run = _repo(tmp_path)
    files = ("reports/p.md", "reports/q.md")
    for f in files:
        (root / f).write_text("v1")
    with pytest.raises(g.UnlockRefused, match="not committed"):
        g.preregistration(root, files)
    run("add", "-A")
    run("commit", "-q", "-m", "prereg")
    hashes = g.preregistration(root, files)
    assert set(hashes) == set(files) and all(len(h) == 40 for h in hashes.values())
    (root / files[0]).write_text("v2 edited after the commit")
    with pytest.raises(g.UnlockRefused, match="changed since"):
        g.preregistration(root, files)
    run("checkout", "--", files[0])                                       # restore the committed version
    assert g.preregistration(root, files)                                 # unchanged again -> accepted
    with pytest.raises(g.UnlockRefused, match="missing"):
        g.preregistration(root, files + ("reports/none.md",))


def test_clean_tree_check(tmp_path):
    root, run = _repo(tmp_path)
    (root / "a.txt").write_text("1")
    run("add", "-A")
    run("commit", "-q", "-m", "a")
    g.clean_tree(root)
    (root / "b.txt").write_text("stray file")
    with pytest.raises(g.UnlockRefused, match="not clean"):
        g.clean_tree(root)


def test_manifest_detects_changed_or_missing_artifacts(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"frozen-1")
    b.write_bytes(b"frozen-2")
    man = tmp_path / "manifest.json"
    man.write_text(json.dumps({"artifacts": g.hash_files([a, b])}))
    assert set(g.verify_manifest(man)["artifacts"]) == {str(a), str(b)}
    a.write_bytes(b"tampered")
    with pytest.raises(g.UnlockRefused, match="a.bin"):
        g.verify_manifest(man)
    a.write_bytes(b"frozen-1")
    b.unlink()
    with pytest.raises(g.UnlockRefused, match="b.bin"):
        g.verify_manifest(man)
