"""The numpy tree evaluator (what Lambda runs) must reproduce xgboost's own predictions."""
from pathlib import Path

import numpy as np
import pytest

from ott_recsys.serving.trees import TreeEnsemble

from .serving_real import HAVE_REAL, OUT

xgb = pytest.importorskip("xgboost")


def test_matches_xgboost_on_a_small_ranker_with_missing_values(tmp_path):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(3000, 12)).astype(np.float32)
    X[rng.random(X.shape) < 0.25] = np.nan  # lots of missing values: default directions must be followed
    X[:, 3] = np.round(X[:, 3])  # ties at split points
    y = (np.nan_to_num(X[:, 0]) + 0.5 * np.nan_to_num(X[:, 1]) + rng.normal(size=3000) > 0.8).astype(int)
    qid = np.repeat(np.arange(60), 50)
    m = xgb.XGBRanker(objective="rank:ndcg", n_estimators=40, max_depth=5, learning_rate=0.1, random_state=0, tree_method="hist")
    m.fit(X, y, qid=qid)
    p = tmp_path / "m.json"
    m.get_booster().save_model(str(p))
    ens = TreeEnsemble.from_xgboost_json(p)
    Xt = rng.normal(size=(5000, 12)).astype(np.float32)
    Xt[rng.random(Xt.shape) < 0.25] = np.nan
    Xt[:, 3] = np.round(Xt[:, 3])
    assert np.allclose(ens.predict(Xt), m.predict(Xt), atol=1e-5)
    ens.save_npz(tmp_path / "m.npz")
    assert np.array_equal(TreeEnsemble.load_npz(tmp_path / "m.npz").predict(Xt), ens.predict(Xt))


@pytest.mark.skipif(not HAVE_REAL, reason="Module 6/8 artifacts not built")
@pytest.mark.parametrize("name,n_trees", [("M2", 351), ("M0", 188)])
def test_matches_xgboost_on_200k_real_feature_rows(name, n_trees):
    z = np.load(OUT / "ranker" / "features.npz")
    X, cols = z["X"], [str(c) for c in z["cols"]]
    b = xgb.Booster()
    b.load_model(str(OUT / "test_prep" / "frozen" / f"final_{name}.json"))
    names = list(b.feature_names)
    sel = np.random.default_rng(1).choice(len(X), 200_000, replace=False)
    Xs = X[sel][:, [cols.index(n) for n in names]]
    ens = TreeEnsemble.from_xgboost_json(OUT / "test_prep" / "frozen" / f"final_{name}.json")
    assert ens.n_trees == n_trees and ens.feature_names == names
    ref = b.predict(xgb.DMatrix(Xs, feature_names=names))
    got = ens.predict(Xs)
    assert np.abs(got - ref).max() < 1e-4, np.abs(got - ref).max()
    # ordering inside a user is what matters: check the rank order agrees on the first 20 users' rows
    rows = np.flatnonzero(np.isin(z["row_user"], np.arange(20)))
    Xu = X[rows][:, [cols.index(n) for n in names]]
    ru = z["row_user"][rows]
    pr, pg = b.predict(xgb.DMatrix(Xu, feature_names=names)), ens.predict(Xu)
    for u in range(20):
        m = ru == u
        assert (np.argsort(-pr[m], kind="stable")[:20] == np.argsort(-pg[m], kind="stable")[:20]).all()
