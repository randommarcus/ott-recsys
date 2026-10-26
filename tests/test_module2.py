"""Module 2 tests: per-user split, metrics (hand-checked), baselines, test lock."""
import ast
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from ott_recsys.baselines import GlobalPop, ItemKNN
from ott_recsys.config import Config
from ott_recsys.data import TestLocked, UserSplit, per_user_split
from ott_recsys.eval import cold_report, evaluate, summarize
from ott_recsys.experiments import run_validation

cfg = Config()
SRC = Path(__file__).resolve().parents[1] / "src" / "ott_recsys"


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label", "timestamp"]).assign(hour=0)


def _synthetic(n_users=30, n_items=40, per_user=20, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(1, n_users + 1):
        items = rng.choice(np.arange(1, n_items + 1), per_user, replace=False)
        for j, i in enumerate(items):
            rows.append((u, int(i), int(rng.random() < 0.5), 1_000_000 + 1000 * j))
    return _df(rows)


# ---------------------------------------------------------------- split
def test_split_sizes_and_order():
    df = pd.concat([_synthetic(per_user=20), _df([(99, i, 1, 100 + i) for i in range(1, 11)])])
    train, val, test = per_user_split(df, cfg)
    for u in [1, 99]:
        n = (df.user_id == u).sum()
        assert (len(train[train.user_id == u]), len(val[val.user_id == u]), len(test[test.user_id == u])) == \
               ((14, 2, 4) if n == 20 else (7, 1, 2))
        assert train[train.user_id == u].timestamp.max() <= val[val.user_id == u].timestamp.min()
        assert val[val.user_id == u].timestamp.max() <= test[test.user_id == u].timestamp.min()


def test_split_drops_users_below_minimum():
    df = pd.concat([_synthetic(n_users=3), _df([(50, i, 1, i) for i in range(1, 10)])])  # 9 rows
    parts = per_user_split(df, cfg)
    assert all(50 not in p.user_id.values for p in parts)


def test_split_is_deterministic_and_ties_break_by_item_id():
    df = _df([(1, i, 1, 500) for i in range(1, 21)])  # all timestamps tied
    a = per_user_split(df, cfg)
    b = per_user_split(df.sample(frac=1, random_state=3), cfg)  # shuffled input
    for x, y in zip(a, b):
        pd.testing.assert_frame_equal(x.reset_index(drop=True), y.reset_index(drop=True))
    assert a[2].item_id.tolist() == [17, 18, 19, 20]  # highest ids are 'latest'


def test_real_split_has_no_future_leakage():
    d = cfg.out_dir / "user_split"
    if not (d / "train.parquet").exists():
        pytest.skip("run scripts/build_user_split.py first")
    tr, va = pd.read_parquet(d / "train.parquet"), pd.read_parquet(d / "val.parquet")
    te = pd.read_parquet(d / "test.parquet")  # structural check only: timestamps
    g = lambda x, f: x.groupby("user_id").timestamp.agg(f)
    assert (g(tr, "max") <= g(va, "min")).all() and (g(va, "max") <= g(te, "min")).all()


# ---------------------------------------------------------------- metrics
class FixedScores:
    def __init__(self, table):
        self.table = table

    def fit(self, fit_df, catalog):
        pass

    def score(self, users, hours=None):
        return np.array([self.table[u] for u in users], dtype=float)


def test_metrics_hand_checked():
    catalog = np.array([1, 2, 3, 4, 5])
    # fit data: user 1 saw item 5; items 1,2,3,5 are seen by someone -> item 4 is COLD
    fit = _df([(1, 5, 1, 1), (3, 1, 1, 1), (3, 2, 1, 1), (3, 3, 1, 1)])
    ev = _df([(1, 1, 1, 9), (1, 3, 1, 9), (1, 2, 0, 9),     # user 1 likes {1,3}
              (2, 2, 1, 9), (2, 4, 1, 9)])                   # user 2 likes {2,4(cold)}
    scores = {1: [0.9, 0.8, 0.7, 0.6, 1.0],   # item 5 seen -> removed; ranking 1,2,3,4
              2: [0.1, 0.9, 0.05, 0.8, 0.01]}  # ranking 2,4,1,3,5
    res = evaluate(FixedScores(scores), fit, ev, catalog, ks=(2, 3), ndcg_k=3, cov_k=2)
    pu = res.per_user
    # user 1: top2={1,2} hits 1/2; top3={1,2,3} hits 2/2
    assert pu.loc[1, "all_recall@2"] == pytest.approx(0.5)
    assert pu.loc[1, "all_recall@3"] == pytest.approx(1.0)
    # NDCG@3 user 1: DCG = 1/log2(2) + 1/log2(4) = 1.5 ; IDCG = 1 + 1/log2(3)
    assert pu.loc[1, "all_ndcg@3"] == pytest.approx(1.5 / (1 + 1 / np.log2(3)))
    # user 2: hits at rank 1 and 2 -> perfect
    assert pu.loc[2, "all_recall@2"] == pytest.approx(1.0) and pu.loc[2, "all_ndcg@3"] == pytest.approx(1.0)
    # warm/cold views: user 2's warm truth {2}, cold truth {4}; user 1 has no cold truth
    assert pu.loc[2, "warm_recall@2"] == pytest.approx(1.0) and pu.loc[2, "cold_recall@2"] == pytest.approx(1.0)
    assert np.isnan(pu.loc[1, "cold_recall@2"])
    # coverage@2: top2 sets {1,2} and {2,4} -> {1,2,4} / 5 items
    assert res.coverage == pytest.approx(0.6)
    s = summarize(pu, res.coverage, n_boot=0)
    assert s.loc["all_recall@2", "mean"] == pytest.approx(0.75)
    assert s.loc["cold_recall@2", "n_users"] == 1


def test_seen_items_are_never_recommended():
    catalog = np.arange(1, 6)
    fit = _df([(1, 5, 1, 1), (1, 4, 1, 1)])
    ev = _df([(1, 1, 1, 9)])
    res = evaluate(FixedScores({1: [0, 0, 0, 0.9, 1.0]}), fit, ev, catalog, ks=(3,), ndcg_k=3)
    assert not set(res.top[0, :3]) & {3, 4}  # catalog indices of items 4 and 5


def test_cold_report():
    fit = _df([(1, 1, 1, 1)])
    ev = _df([(1, 1, 1, 2), (1, 2, 1, 2)])
    r = cold_report(fit, ev)
    assert (r["items_not_in_fit"], r["likes_on_new_items"]) == (1, 1)


# ---------------------------------------------------------------- baselines
def test_item_knn_recommends_co_liked_item():
    fit = _df([(1, 1, 1, 1), (1, 2, 1, 1), (2, 1, 1, 1), (2, 2, 1, 1), (3, 1, 1, 1), (3, 3, 1, 1), (4, 3, 1, 1)])
    m = ItemKNN(k=5, shrink=0.0)
    catalog = np.array([1, 2, 3, 4])  # item 4 is never liked by anyone
    m.fit(fit, catalog)
    s = m.score(np.array([3]))[0]  # user 3 liked items 1 and 3; item 2 is co-liked with item 1
    assert s[1] > 0.1 and s[3] < 1e-5


def test_global_pop_orders_by_likes():
    fit = _df([(1, 1, 1, 1), (2, 1, 1, 1), (3, 2, 1, 1), (3, 3, 0, 1)])
    m = GlobalPop()
    m.fit(fit, np.array([1, 2, 3]))
    assert m.score(np.array([1]))[0].tolist() == [2, 1, 0]


# ---------------------------------------------------------------- test lock
def test_test_portion_is_locked():
    def boom():
        raise AssertionError("test portion was loaded")
    split = UserSplit(_df([(1, 1, 1, 1)]), _df([(1, 2, 1, 2)]), boom)
    with pytest.raises(TestLocked):
        _ = split.test


def test_validation_pipeline_never_touches_test():
    full = _synthetic(n_users=40, n_items=50, per_user=20)
    train, val, _ = per_user_split(full, cfg)

    def boom():
        raise AssertionError("tuning code loaded the test portion")
    split = UserSplit(train, val, boom)
    movies = pd.DataFrame({"movieId": np.arange(1, 51), "title": "t", "genres": "g"})
    grids = {"hour_pop": {"window": [0], "lam": [50.0]}, "item_knn": {"k": [5], "shrink": [0.0]}}
    out = run_validation(split, movies, grids=grids, n_shuffle=1, n_boot=0)  # must not raise
    assert "item_knn" in out["tables"]


def test_tuning_code_never_references_test_split():
    """Static guard: tuning/eval/baseline code and run script must not use
    .test or unlock_test."""
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    files = [SRC / "experiments.py", SRC / "eval.py", SRC / "baselines.py", SRC / "content.py",
             SRC / "newrelease.py", SRC / "tmdb.py", SRC / "twotower.py", SRC / "export.py", SRC / "ncf.py", SRC / "rerank.py", SRC / "context.py", SRC / "rankfeat.py", SRC / "ranker.py", SRC / "bandit.py", SRC / "banditdata.py", scripts / "run_module2.py", scripts / "run_module3.py",
             scripts / "build_embeddings.py", scripts / "fetch_tmdb.py", scripts / "train_two_tower.py",
             scripts / "run_module4.py", scripts / "run_module4_all.py", scripts / "run_module4b.py", scripts / "build_candidates.py", scripts / "train_ncf.py",
             scripts / "run_module5.py", scripts / "run_module5_all.py", scripts / "build_ranker_features.py",
             scripts / "run_module6.py", scripts / "run_module6_all.py", scripts / "run_module7.py",
             scripts / "run_module7_ranker.py", scripts / "run_module7_all.py", scripts / "run_module7b.py"]
    for f in files:
        tree = ast.parse(f.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in {"test", "unlock_test", "_test_loader"}:
                pytest.fail(f"{f.name}:{node.lineno} touches the locked test portion")
