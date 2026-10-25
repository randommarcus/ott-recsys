"""Module 2b tests: scaling changes (fast top-K, chunking, block kNN), user
sampling reproducibility, and the memory guard."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ott_recsys.baselines import GlobalPop, ItemKNN, knn_neighbors
from ott_recsys.config import Config
from ott_recsys.data import load_raw
from ott_recsys.eval import _top_one, evaluate, paired_diff

ROOT = Path(__file__).resolve().parents[1]


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label", "timestamp"]).assign(hour=0)


def test_fast_topk_matches_full_stable_sort_with_ties():
    rng = np.random.default_rng(0)
    for _ in range(50):
        s = rng.integers(0, 6, size=300).astype(float)  # many ties
        s[rng.choice(300, 40, replace=False)] = -np.inf
        for k in (1, 10, 50, 299, 300):
            assert _top_one(s, k).tolist() == np.argsort(-s, kind="stable")[:k].tolist()


def test_evaluate_is_chunk_invariant():
    rng = np.random.default_rng(1)
    rows = [(u, int(i), int(rng.random() < 0.6), 10 * j)
            for u in range(1, 31) for j, i in enumerate(rng.choice(np.arange(1, 61), 12, replace=False))]
    df = _df(rows)
    fit, ev = df[df.timestamp < 80], df[df.timestamp >= 80]
    cat = np.arange(1, 61)
    m = GlobalPop()
    m.fit(fit, cat)
    a = evaluate(m, fit, ev, cat, ks=(5, 20), chunk=7)
    b = evaluate(m, fit, ev, cat, ks=(5, 20), chunk=1000)
    pd.testing.assert_frame_equal(a.per_user, b.per_user)
    assert (a.top == b.top).all()


def test_knn_shared_neighbours_equal_direct_fit():
    rng = np.random.default_rng(2)
    rows = [(u, int(i), 1, 0) for u in range(1, 40) for i in rng.choice(np.arange(1, 30), 8, replace=False)]
    fit = _df(rows)
    cat = np.arange(1, 30)
    state = knn_neighbors(fit, cat, [0.0, 10.0], k_max=12)
    users = np.arange(1, 40)
    for sh, k in [(0.0, 5), (10.0, 12), (10.0, 3)]:
        shared = ItemKNN().use(state, sh, k).score(users)
        direct = ItemKNN(k=k, shrink=sh)
        direct.fit(fit, cat)
        np.testing.assert_allclose(shared, direct.score(users), rtol=1e-5, atol=1e-7)


# ---------------------------------------------------------------- sampling
def _fake_dataset(root: Path, n_users=40):
    d = root / "raw" / "fake"
    d.mkdir(parents=True)
    rng = np.random.default_rng(3)
    rows = [(u, int(m), 4.0, 1_000_000_000 + 100 * j)
            for u in range(1, n_users + 1)
            for j, m in enumerate(rng.choice(np.arange(1, 31), 20 + u % 5, replace=False))]
    rows += [(999, 1, 3.0, 1_000_000_000)]  # user with 1 rating: never eligible
    pd.DataFrame(rows, columns=["userId", "movieId", "rating", "timestamp"]).to_csv(d / "ratings.csv", index=False)
    pd.DataFrame({"movieId": range(1, 41), "title": "t", "genres": "g"}).to_csv(d / "movies.csv", index=False)
    pd.DataFrame({"movieId": range(1, 41), "imdbId": "1", "tmdbId": 1}).to_csv(d / "links.csv", index=False)
    return Config(dataset_name="fake", raw_dir=root / "raw", processed_dir=root / "proc",
                  sample_users=10, sample_seed=7)


def test_user_sampling_is_reproducible_and_keeps_full_histories(tmp_path):
    cfg = _fake_dataset(tmp_path)
    r1, m1, _ = load_raw(cfg)
    r2, _, _ = load_raw(cfg)
    assert r1.equals(r2)
    saved = json.loads((cfg.out_dir / "sampled_users.json").read_text())
    assert saved["seed"] == 7 and saved["n_users"] == 10 and 999 not in saved["user_ids"]
    assert sorted(r1.userId.unique()) == saved["user_ids"]
    full = pd.read_csv(cfg.dataset_dir / "ratings.csv")
    for u in saved["user_ids"]:  # every rating of a sampled user is kept
        assert (full.userId == u).sum() == (r1.userId == u).sum()
    assert set(m1.movieId) == set(r1.movieId)  # catalog shrinks to rated movies
    other = Config(**{**cfg.__dict__, "sample_seed": 8})
    assert sorted(load_raw(other)[0].userId.unique()) != saved["user_ids"]


# ---------------------------------------------------------------- memory guard
def test_memory_guard_kills_a_runaway_process(tmp_path):
    hog = tmp_path / "hog.py"
    hog.write_text("import time\nx = bytearray(600 * 1024 * 1024)\nx[::4096] = b'a' * len(x[::4096])\ntime.sleep(30)\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "guarded_run.py"), "--limit-gb", "0.3", str(hog)],
                       capture_output=True, text=True, timeout=25)
    assert r.returncode == 42 and "MEMORY GUARD TRIPPED" in r.stdout


def test_memory_guard_passes_a_normal_process(tmp_path):
    ok = tmp_path / "ok.py"
    ok.write_text("print('fine')\n")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "guarded_run.py"), "--limit-gb", "2", str(ok)],
                       capture_output=True, text=True, timeout=25)
    assert r.returncode == 0 and "fine" in r.stdout


def test_paired_diff_hand_checked():
    a, b = pd.Series([0.5, 0.3, 0.2, np.nan]), pd.Series([0.4, 0.3, 0.0, 0.9])
    m, lo, hi = paired_diff(a, b, n_boot=500)  # diffs: 0.1, 0.0, 0.2 (NaN row ignored)
    assert abs(m - 0.1) < 1e-12 and lo <= m <= hi
    same = pd.Series(np.random.default_rng(0).random(200))
    m, lo, hi = paired_diff(same + 0.05, same)  # constant shift: interval collapses on 0.05
    assert abs(m - 0.05) < 1e-12 and lo > 0.049 and hi < 0.051
