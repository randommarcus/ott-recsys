"""Module 6 tests: request-time leakage proofs, features, folds, simulated device and synthetic labels."""
import numpy as np
import pandas as pd
import pytest

from ott_recsys.context import (DEVICES, compact, device_onehot, genre_affinity, genre_matrix, pop_asof, pop_keys,
                                request_times, session_state, shuffled_device, simulate_device, synthetic_truth)
from ott_recsys.rankfeat import build_features
from ott_recsys.ranker import GROUPS, group_importance, oof_scores, user_folds


# ------------------------------------------------------------ request time and session state
def test_request_time_uses_timestamps_only():
    val = pd.DataFrame({"user_id": [1, 1, 2], "item_id": [5, 6, 7], "label": [1, 0, 1], "timestamp": [300, 200, 900]})
    scrambled = val.assign(item_id=[9, 9, 9], label=[0, 0, 0])
    assert request_times(val, [1, 2]).tolist() == [200, 900] == request_times(scrambled, [1, 2]).tolist()


def test_session_state_hand_checked():
    ts = np.array([0, 60, 120, 5000, 5060])
    lab = np.array([1, 0, 1, 1, 0])
    s = session_state(ts, lab, 5100)                    # 40 s after the last -> in session; chain = last two
    assert (s["n"], s["length"], s["in_session"], s["start"]) == (5, 2, 1.0, 3)
    assert s["mins_since_last"] == pytest.approx(40 / 60) and s["like_rate"] == pytest.approx(0.5)
    s = session_state(ts, lab, 5060 + 1801)             # more than 30 minutes of silence -> not in a session
    assert (s["length"], s["in_session"]) == (0, 0.0) and np.isnan(s["like_rate"])
    s = session_state(ts, lab, 0)                       # nothing strictly before
    assert s["n"] == 0 and np.isnan(s["mins_since_last"])
    s = session_state(ts, lab, 60)                      # the interaction AT t_req is excluded (strictly before)
    assert s["n"] == 1 and s["length"] == 1


def test_session_state_future_invariance_and_ties():
    ts, lab = np.array([0, 100, 200, 400]), np.array([1, 1, 0, 1])
    base = session_state(ts, lab, 500)
    ts2 = np.append(ts, [500, 500, 650, 9000])          # same-second tie and later rows
    lab2 = np.append(lab, [1, 0, 1, 0])
    assert session_state(ts2, lab2, 500) == base


# ------------------------------------------------------------ as-of popularity
def test_pop_asof_strictly_before_and_future_invariant():
    item = np.array([0, 0, 0, 1, 1]); ts = np.array([10, 20, 30, 5, 99])
    keys, start = pop_keys(item, ts, 3)
    assert pop_asof(keys, start, np.array([0, 0, 0, 1, 1, 2]), np.array([10, 11, 31, 5, 100, 50])).tolist() == [0, 1, 3, 0, 2, 0]
    keys2, start2 = pop_keys(np.append(item, [0, 1]), np.append(ts, [1000, 2000]), 3)  # later likes
    assert pop_asof(keys2, start2, np.array([0, 1]), np.array([31, 100])).tolist() == pop_asof(keys, start, np.array([0, 1]), np.array([31, 100])).tolist()


def test_compact_preserves_order_and_pads():
    m = np.array([[10, 11, 12, 13]]); keep = np.array([[True, False, True, False]])
    assert compact(m, keep, -1).tolist() == [[10, 12, -1, -1]]
    assert np.isnan(compact(m.astype(float), keep, np.nan)[0, 2:]).all()


def test_genre_matrix_and_affinity():
    movies = pd.DataFrame({"movieId": [1, 2, 3], "title": "t", "genres": ["Drama|Comedy", "Drama", "(no genres listed)"]})
    m, names = genre_matrix(movies, np.array([1, 2, 3]))
    assert names == ["Comedy", "Drama"] and m.tolist() == [[1, 1], [0, 1], [0, 0]]
    share = np.array([0.25, 0.75])                      # user: 25% comedy, 75% drama
    assert genre_affinity(share, m).tolist() == pytest.approx([0.5, 0.75, 0.0])


# ------------------------------------------------------------ build_features: leakage proofs on synthetic data
def _world(seed=0):
    rng = np.random.default_rng(seed)
    n_items, n_users, W = 12, 4, 8
    catalog = np.arange(100, 100 + n_items)
    movies = pd.DataFrame({"movieId": catalog, "title": "t", "genres": ["Drama", "Comedy|Drama", "Action", "Drama"] * 3})
    gm, names = genre_matrix(movies, catalog)
    emb = rng.normal(size=(n_items, 6)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rows = []
    for u in range(n_users):
        t = 1_000 + u * 7
        for k in range(12):
            rows.append((u + 1, int(catalog[rng.integers(0, n_items)]), int(rng.random() < 0.6), t + k * 40))
    train = pd.DataFrame(rows, columns=["user_id", "item_id", "label", "timestamp"])
    users = np.arange(1, n_users + 1)
    t_req = train.groupby("user_id")["timestamp"].max().to_numpy() + 20          # request just after the last train row
    cand = np.array([rng.permutation(n_items)[:W] for _ in range(n_users)])
    rel = np.full(n_items, 0.0); rel[3] = float(t_req.max() + 10)                 # catalog idx 3 is released after everyone's request
    rel[5] = np.nan                                                               # unknown release date
    kw = dict(users=users, t_req=t_req, cand=cand, n_main=5, catalog=catalog, genre_m=gm, genre_names=names, emb_c=emb, rel_ts=rel,
              tt_score=rng.random((n_users, W)), ncf_logit=rng.normal(size=(n_users, W)), ncf_user_known=np.ones(n_users),
              ncf_item_known_cat=np.ones(n_items, bool))
    return train, kw


def test_future_invariance_of_every_feature():
    train, kw = _world()
    base = build_features(train, **kw)
    t_late = int(kw["t_req"].max()) + 5
    extra = pd.DataFrame({"user_id": [1, 2, 3, 4] * 3, "item_id": [100, 101, 102, 103] * 3, "label": [1, 0, 1, 1] * 3,
                          "timestamp": [t_late + 11 * i for i in range(12)]})   # after EVERY user's request
    aug = build_features(pd.concat([train, extra], ignore_index=True), **kw)
    assert base["cols"] == aug["cols"] and np.array_equal(base["X"], aug["X"], equal_nan=True)
    assert (base["cand_f"] == aug["cand_f"]).all()


def test_same_second_and_own_later_rows_never_reach_user_side_features():
    train, kw = _world(1)
    base = build_features(train, **kw)
    r = 1
    extra = pd.DataFrame({"user_id": [r + 1] * 3, "item_id": [101, 104, 105], "label": [1, 1, 1],
                          "timestamp": [int(kw["t_req"][r]), int(kw["t_req"][r]), int(kw["t_req"][r]) + 500]})  # tie + later
    aug = build_features(pd.concat([train, extra], ignore_index=True), **kw)
    cols = base["cols"]
    keep_cols = [i for i, c in enumerate(cols) if c != "pop_asof"]  # pop_asof may count OTHER users' earlier likes by design
    rows_u = base["row_user"] == r
    assert np.array_equal(base["X"][:, keep_cols], aug["X"][:, keep_cols], equal_nan=True)
    # pop_asof for this user's request time is unaffected too, because the extra likes are not strictly before it
    pi = cols.index("pop_asof")
    assert np.array_equal(base["X"][rows_u, pi], aug["X"][rows_u, pi])


def test_release_filter_keeps_unknown_dates_and_compacts_rows():
    train, kw = _world(2)
    out = build_features(train, **kw)
    for r in range(len(kw["users"])):
        row = out["cand_f"][r]
        kept = row[row >= 0]
        assert (row[len(kept):] == -1).all()                                       # padding only at the end
        assert 3 not in kept.tolist()                                              # unreleased title removed everywhere
    original = kw["cand"]
    assert out["removed_future"] == int((original == 3).sum())
    if (original == 5).any():
        assert (out["cand_f"] == 5).sum() == (original == 5).sum()                  # NaN release date kept


def test_feature_columns_lane_flag_and_tt_rank():
    train, kw = _world(3)
    out = build_features(train, **kw)
    cols, X = out["cols"], out["X"]
    lane, rank = X[:, cols.index("lane")], X[:, cols.index("tt_rank")]
    assert set(np.unique(lane)).issubset({0.0, 1.0}) and np.isnan(rank[lane == 1]).all()
    for r in range(len(kw["users"])):
        rr = rank[(out["row_user"] == r) & (lane == 0)]
        assert rr.tolist() == list(range(len(rr)))                                  # 0,1,2,... among kept main-lane slots
    assert X.shape[1] == len(cols) and X.dtype == np.float32


# ------------------------------------------------------------ folds and ranker
def test_user_folds_partition_and_determinism():
    f = user_folds(103, 5, 0)
    assert sorted(np.bincount(f).tolist()) == [20, 20, 21, 21, 21] and (f == user_folds(103, 5, 0)).all()
    assert not (f == user_folds(103, 5, 1)).all()


def test_oof_scores_cover_every_row_once_and_use_the_informative_feature():
    rng = np.random.default_rng(0)
    U, W = 120, 15
    row_user = np.repeat(np.arange(U), W)
    sig = rng.normal(size=U * W)
    X = np.column_stack([sig, rng.normal(size=U * W), rng.normal(size=U * W)]).astype(np.float32)
    y = (sig + 0.3 * rng.normal(size=U * W) > 1.2).astype(int)
    s, best, imp = oof_scores(X, y, row_user, U, ["good", "noise1", "noise2"], k=3, log=lambda m: None, params=dict(n_estimators=60))
    assert not np.isnan(s).any() and len(best) == 3
    assert imp["good"] > 0.5 > max(imp["noise1"], imp["noise2"])
    assert np.corrcoef(s, sig)[0, 1] > 0.5


def test_group_importance_sums_groups():
    out = group_importance({"tt_score": 0.4, "tt_rank": 0.1, "hour": 0.05, "dev_mobile": 0.01, "g_Drama": 0.02})
    assert out["two-tower"] == pytest.approx(0.5) and out["hour / day-of-week"] == pytest.approx(0.05)
    assert out["device (SIMULATED)"] == pytest.approx(0.01) and out["item genre one-hots"] == pytest.approx(0.02)


# ------------------------------------------------------------ simulated device and synthetic labels
def test_device_is_independent_of_behaviour_and_shuffle_keeps_marginals():
    n = 60_000
    d = simulate_device(n, 0)
    assert np.abs(np.bincount(d, minlength=4) / n - np.array([0.40, 0.35, 0.15, 0.10])).max() < 0.01
    trait = np.random.default_rng(5).normal(size=n)                      # any behaviour-like user trait
    assert max(abs(np.corrcoef(trait, device_onehot(d)[:, k])[0, 1]) for k in range(4)) < 0.02
    sh = shuffled_device(d, 1)
    assert np.bincount(sh, minlength=4).tolist() == np.bincount(d, minlength=4).tolist() and not (sh == d).all()
    assert device_onehot(d).sum(1).tolist() == [1.0] * n


def test_synthetic_truth_only_drops_mobile_drama_no_comedy_likes():
    rng = np.random.default_rng(0)
    n_users, n_items = 4000, 30
    drama_nc = np.zeros(n_items, bool); drama_nc[:10] = True
    truth = [rng.choice(n_items, 12, replace=False) for _ in range(n_users)]
    device = simulate_device(n_users, 3)
    new, removed = synthetic_truth(truth, device, drama_nc, 0.6, 4)
    mobile = np.array([DEVICES[d] == "mobile" for d in device])
    eligible = dropped = 0
    for t, nt, m in zip(truth, new, mobile):
        assert set(nt.tolist()).issubset(set(t.tolist()))                    # likes are only ever removed
        gone = set(t.tolist()) - set(nt.tolist())
        assert all(drama_nc[i] for i in gone)                                # only Drama-without-Comedy titles
        if not m:
            assert not gone                                                  # non-mobile users untouched
        else:
            eligible += int(drama_nc[t].sum()); dropped += len(gone)
    assert removed == dropped and abs(dropped / eligible - 0.6) < 0.03


def test_static_groups_cover_known_feature_names():
    names = {n for v in GROUPS.values() for n in v}
    assert {"tt_score", "ncf_logit", "pop_asof", "hour", "session_len", "dev_mobile"}.issubset(names)
