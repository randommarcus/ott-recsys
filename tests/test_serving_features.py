"""The single-request feature code that runs in Lambda must agree with the batch code the rankers were trained with (rankfeat.build_features),
and the live (search) candidate lists must agree with the precomputed ones. Real-data tests are skipped when the serving artifacts are not built."""
import numpy as np
import pandas as pd
import pytest

from ott_recsys import testprep
from ott_recsys.data import load_user_split
from ott_recsys.rankfeat import build_features
from ott_recsys.serving.bundle import Bundle
from ott_recsys.serving.features import build_request_features, matrix
from ott_recsys.serving.search import NumpySearch, live_candidates
from ott_recsys.serving_offline import record_at

from .serving_real import CFG, HAVE_REAL, OUT

SV = OUT / "serving"
pytestmark = pytest.mark.skipif(not (HAVE_REAL and (SV / "users.npz").exists()), reason="serving artifacts not built (scripts/build_serving_artifacts.py)")


@pytest.fixture(scope="module")
def world():
    b = Bundle(SV / "bundle")
    a = dict(np.load(SV / "users.npz"))
    split = load_user_split(CFG)
    history = testprep.history_frame(split.train, split.val, rehearsal=False)
    lists = np.load(OUT / "test_prep" / "real" / "lists.npz")
    return b, a, history, lists


def _sample(a, n=300):
    no_vec = np.flatnonzero(~a["has_ttvec"])[:10]
    rest = np.random.default_rng(0).choice(len(a["user_id"]), n - len(no_vec), replace=False)
    return np.unique(np.concatenate([no_vec, rest]))


@pytest.mark.parametrize("gap_s", [600, 3 * 86400])  # inside a session / a new session
def test_features_match_the_batch_builder(world, gap_s):
    b, a, history, lists = world
    sel = _sample(a)
    users, t_req = a["user_id"][sel], a["last_ts"][sel] + gap_s
    ref = build_features(history, users, t_req, lists["cand"][sel].astype(np.int64), 200, b.catalog, b.genre_m, b.genre_names, b.emb,
                         b.rel_ts, lists["tt_score"][sel], lists["ncf_fold"][sel], lists["ncf_fold_user_known"][sel], b.known)
    cols = ref["cols"]
    worst, n_rows = 0.0, 0
    for r, i in enumerate(sel):
        mine = build_request_features(b, record_at(a, i), a["cand"][i], int(t_req[r]))
        Xm = matrix(mine, cols)
        Xr = ref["X"][ref["row_user"] == r]
        assert Xm.shape == Xr.shape, (r, Xm.shape, Xr.shape)
        assert np.array_equal(mine["_cand"], ref["cand_f"][r][ref["cand_f"][r] >= 0]), "kept candidates differ"
        assert np.array_equal(np.isnan(Xm), np.isnan(Xr)), f"NaN pattern differs for user {users[r]}"
        both = ~np.isnan(Xr)
        worst = max(worst, float(np.abs(Xm[both] - Xr[both]).max(initial=0.0)))
        n_rows += len(Xr)
    # popularity, flags, hour, session fields are exact; scores and similarities agree to float rounding
    assert worst < 5e-3, worst
    print(f"gap {gap_s}s: {len(sel)} users, {n_rows} rows, largest absolute difference {worst:.2e}")


def test_ranking_is_identical_to_the_batch_ranking(world):
    from ott_recsys.serving.trees import TreeEnsemble
    b, a, history, lists = world
    sel = _sample(a, 120)
    users, t_req = a["user_id"][sel], a["last_ts"][sel] + 600
    ref = build_features(history, users, t_req, lists["cand"][sel].astype(np.int64), 200, b.catalog, b.genre_m, b.genre_names, b.emb,
                         b.rel_ts, lists["tt_score"][sel], lists["ncf_fold"][sel], lists["ncf_fold_user_known"][sel], b.known)
    m2 = b.rankers["M2"]
    s_ref = m2.predict(matrix({n: ref["X"][:, j] for j, n in enumerate(ref["cols"])} | {"_cand": ref["X"][:, 0]}, m2.feature_names))
    same = 0
    for r, i in enumerate(sel):
        mine = build_request_features(b, record_at(a, i), a["cand"][i], int(t_req[r]))
        s_mine = m2.predict(matrix(mine, m2.feature_names))
        s_r = s_ref[ref["row_user"] == r]
        top = lambda s: np.argsort(-s, kind="stable")[:20]
        same += set(top(s_mine)) == set(top(s_r))
    assert same >= 0.98 * len(sel), f"top-20 sets agree for only {same} of {len(sel)} users"


def test_live_candidates_match_precomputed_lists(world):
    b, a, _, _ = world
    s = NumpySearch(b)
    sel = np.random.default_rng(1).choice(len(a["user_id"]), 400, replace=False)
    sel = np.unique(np.concatenate([sel, np.flatnonzero(~a["has_ttvec"])[:5]]))
    same_main = same_lane = 0
    for i in sel:
        rec = record_at(a, i)
        cand, info = live_candidates(b, rec, s)
        off = rec["cand"]
        same_main += set(cand[:200][cand[:200] >= 0]) == set(off[:200][off[:200] >= 0])
        same_lane += set(cand[200:][cand[200:] >= 0]) == set(off[200:][off[200:] >= 0])
    # differences can only come from tied scores at the cut-off
    assert same_main >= 0.99 * len(sel) and same_lane >= 0.97 * len(sel), (same_main, same_lane, len(sel))
