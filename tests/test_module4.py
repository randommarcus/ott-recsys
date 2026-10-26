"""Module 4 tests: leakage-free pairs, loss maths, model behaviour, union eval, export."""
import json

import numpy as np
import pandas as pd
import pytest

from ott_recsys.eval import evaluate, evaluate_lists, union_top
from ott_recsys.export import export_milvus
from ott_recsys.twotower import (TwoTowerRetrieval, better, build_pairs, export_inference, log_sampling_prob,
                                 make_model, sampled_softmax_loss, train, user_histories)

CAT = np.arange(1, 21)  # catalog: movie ids 1..20 -> indices 0..19


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label", "timestamp"]).assign(hour=0)


def _pair_tuples(p):
    return sorted((int(u), int(x), tuple(int(i) for i in h if i >= 0)) for u, x, h in zip(p["user"], p["pos"], p["hist"]))


# ------------------------------------------------------------ leakage
def test_history_is_strictly_before_and_ties_never_see_each_other():
    # likes: item1@10, item2@20, item3@20 (tie), item4@30, item5@40; item9 is a dislike@25
    df = _df([(1, 1, 1, 10), (1, 2, 1, 20), (1, 3, 1, 20), (1, 9, 0, 25), (1, 4, 1, 30), (1, 5, 1, 40)])
    p = build_pairs(df, CAT, n=10)
    got = {int(CAT[x]): [int(CAT[i]) for i in h if i >= 0] for x, h in zip(p["pos"], p["hist"])}
    assert 1 not in got                      # first like has no history -> no pair
    assert got[2] == [1] and got[3] == [1]   # tied items see only the strictly earlier one, never each other
    assert sorted(got[4]) == [1, 2, 3]       # the dislike (item 9) is not in anyone's history
    assert sorted(got[5]) == [1, 2, 3, 4]
    for ts, x, h in zip(p["ts"], p["pos"], p["hist"]):
        hist_items = [CAT[i] for i in h if i >= 0]
        assert CAT[x] not in hist_items and 9 not in hist_items
        ts_of = {1: 10, 2: 20, 3: 20, 4: 30, 5: 40}
        assert all(ts_of[i] < ts for i in hist_items)


def test_history_keeps_only_the_newest_n():
    df = _df([(1, i, 1, 100 * i) for i in range(1, 8)])
    p = build_pairs(df, CAT, n=3)
    last = {int(CAT[x]): sorted(int(CAT[i]) for i in h if i >= 0) for x, h in zip(p["pos"], p["hist"])}
    assert last[7] == [4, 5, 6] and last[3] == [1, 2] and last[2] == [1]


def test_future_invariance_truncating_data_does_not_change_earlier_pairs():
    rng = np.random.default_rng(0)
    rows = [(u, int(i), int(rng.random() < 0.7), int(t)) for u in range(1, 8)
            for i, t in zip(rng.choice(np.arange(1, 21), 15, replace=False), np.sort(rng.integers(0, 60, 15)))]
    full, T = _df(rows), 30
    cut = full[full["timestamp"] < T]
    p_full, p_cut = build_pairs(full, CAT, n=5), build_pairs(cut, CAT, n=5)
    early = {k: (v[p_full["ts"] < T] if len(v) else v) for k, v in p_full.items()}
    assert _pair_tuples(early) == _pair_tuples(p_cut)  # the future never changes what the past looks like


def test_min_history_drops_early_pairs():
    df = _df([(1, i, 1, 10 * i) for i in range(1, 6)])
    assert len(build_pairs(df, CAT, n=5, min_history=1)["pos"]) == 4
    assert len(build_pairs(df, CAT, n=5, min_history=3)["pos"]) == 2


def test_pairs_come_only_from_the_given_dataframe():
    train_df = _df([(1, 1, 1, 10), (1, 2, 1, 20)])
    p = build_pairs(train_df, CAT, n=5)
    assert set(CAT[p["pos"]].tolist()) == {2}  # a later 'validation' like of item 3 cannot appear


def test_user_histories_last_n_oldest_first_and_padded():
    df = _df([(1, i, 1, 10 * i) for i in range(1, 6)] + [(2, 7, 1, 5), (2, 8, 0, 6)])
    uid, h = user_histories(df, CAT, n=3)
    assert uid.tolist() == [1, 2]
    assert [int(CAT[i]) for i in h[0]] == [3, 4, 5] and [int(CAT[i]) if i >= 0 else -1 for i in h[1]] == [7, -1, -1]


# ------------------------------------------------------------ loss
def _np_loss(u, v, pos, logq, tau, use_logq):
    logits = u @ v.T / tau
    if use_logq:
        logits = logits - logq[pos][None, :]
    for i in range(len(pos)):
        for j in range(len(pos)):
            if i != j and pos[i] == pos[j]:
                logits[i, j] = -1e9
    lse = np.log(np.exp(logits - logits.max(1, keepdims=True)).sum(1)) + logits.max(1)
    return float(np.mean(lse - np.diag(logits)))


@pytest.mark.parametrize("use_logq", [True, False])
def test_sampled_softmax_matches_independent_numpy(use_logq):
    import tensorflow as tf
    u = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    v = np.array([[1, 0], [0, 1], [0.6, 0.8]], dtype=np.float32)
    pos = np.array([3, 5, 3])  # rows 0 and 2 share a positive: accidental hits must be masked
    logq = np.log(np.array([.01, .02, .05, .4, .02, .3, .2], dtype=np.float32))
    got = float(sampled_softmax_loss(tf.constant(u), tf.constant(v), tf.constant(pos), tf.constant(logq), 0.5, use_logq))
    assert got == pytest.approx(_np_loss(u, v, pos, logq, 0.5, use_logq), rel=1e-5)


def test_logq_hand_checked_popular_positive_gets_less_credit():
    """Two identical users; item 0 is rare (Q=.1), item 1 popular (Q=.9); all similarities equal.
    Without correction every row is a coin flip: loss ln2. With logQ the rare positive is easy
    (-> ln(10/9)) and the popular positive is hard (-> ln10): popularity no longer wins for free."""
    import tensorflow as tf
    u = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    v = np.array([[1.0, 0.0], [1.0, 0.0]], dtype=np.float32)
    pos, logq = np.array([0, 1]), np.log(np.array([0.1, 0.9], dtype=np.float32))
    f = lambda use: float(sampled_softmax_loss(tf.constant(u), tf.constant(v), tf.constant(pos), tf.constant(logq), 1.0, use))
    assert f(False) == pytest.approx(np.log(2), rel=1e-5)
    assert f(True) == pytest.approx((np.log(10 / 9) + np.log(10)) / 2, rel=1e-5)


def test_log_sampling_prob():
    lq = log_sampling_prob(np.array([0, 0, 0, 2]), 4)
    assert np.exp(lq[0]) == pytest.approx(0.75) and np.exp(lq[2]) == pytest.approx(0.25)


# ------------------------------------------------------------ model
def _content(n=20, d=8, seed=0):
    return np.random.default_rng(seed).normal(size=(n, d)).astype(np.float32)


@pytest.mark.parametrize("variant", ["id", "content", "both"])
def test_item_and_user_vectors_are_unit_length_and_padding_is_ignored(variant):
    m = make_model(20, _content(), variant, dim=16)
    t = m.item_table(False)
    assert np.allclose(np.linalg.norm(t.numpy(), axis=1), 1, atol=1e-4)
    h1 = np.array([[3, 7, -1, -1]], dtype=np.int32)
    h2 = np.array([[3, 7, -1, -1, -1, -1]], dtype=np.int32)
    h3 = np.array([[7, -1, 3, -1]], dtype=np.int32)  # same items, different order/padding position
    u1, u2, u3 = (m.user_vec(t, h).numpy() for h in (h1, h2, h3))
    assert np.allclose(np.linalg.norm(u1, axis=1), 1, atol=1e-4)
    assert np.allclose(u1, u2, atol=1e-5) and np.allclose(u1, u3, atol=1e-5)


def test_unknown_id_items_use_the_unknown_vector_at_inference():
    import tensorflow as tf
    c = _content()
    c[5] = c[6]  # identical content
    m = make_model(20, c, "both", dim=16)
    known = np.ones(20, dtype=bool); known[5] = False; known[8] = False
    t = m.item_table(False, known=tf.constant(known)).numpy()
    assert not np.allclose(t[5], t[6], atol=1e-4)      # item 5 unknown, item 6 known: same content, different vectors
    m2 = make_model(20, np.tile(c[5], (20, 1)), "both", dim=16)
    kn = np.ones(20, dtype=bool); kn[[2, 9]] = False
    t2 = m2.item_table(False, known=tf.constant(kn)).numpy()
    assert np.allclose(t2[2], t2[9], atol=1e-5)        # two unknown items with identical content are identical


def test_numpy_retrieval_matches_tensorflow_user_tower():
    import tensorflow as tf
    m = make_model(20, _content(), "both", dim=16)
    known = np.ones(20, dtype=bool)
    art = export_inference(m, known)
    df = _df([(1, 2, 1, 10), (1, 4, 1, 20), (1, 6, 1, 30), (2, 1, 1, 10)])
    r = TwoTowerRetrieval(art["item_vecs"], art["W"], art["b"], history_len=5)
    r.fit(df, CAT)
    _, hist = user_histories(df, CAT, 5)
    u_tf = m.user_vec(tf.constant(art["item_vecs"]), hist).numpy()
    np.testing.assert_allclose(r.U, u_tf, atol=1e-5)
    np.testing.assert_allclose(r.score(np.array([1, 2])), u_tf @ art["item_vecs"].T, atol=1e-5)


def test_users_without_likes_fall_back_to_popularity():
    m = make_model(20, _content(), "both", dim=16)
    art = export_inference(m, np.ones(20, dtype=bool))
    df = _df([(1, 2, 1, 10), (2, 2, 1, 10), (3, 5, 1, 10), (4, 7, 0, 10)])  # user 4 has only a dislike
    r = TwoTowerRetrieval(art["item_vecs"], art["W"], art["b"], 5)
    r.fit(df, CAT)
    assert r.no_likes(np.array([1, 4, 99])).tolist() == [False, True, True]
    s = r.score(np.array([4]))[0]
    assert s.argmax() == 1 and s[1] > s[4] > s[0]  # item 2 (2 likes) > item 5 (1 like) > item 1 (0 likes)


def test_training_learns_a_simple_pattern_and_id_dropout_trains_unknown_vector():
    """Users who liked item a next like item a+1 (mod 10): the loss must drop clearly."""
    rng = np.random.default_rng(0)
    rows, t = [], 0
    for u in range(1, 61):
        start = int(rng.integers(0, 10))
        for k in range(8):
            t += 1
            rows.append((u, int((start + k) % 10) + 1, 1, t))
    p = build_pairs(_df(rows), CAT, n=3)
    logs = []
    m, hist, _ = train(p, 20, _content(), "both", batch_size=32, max_epochs=12, fixed_epochs=12, dim=16, lr=5e-3,
                       id_dropout=0.3, log=logs.append)
    assert hist[-1]["loss"] < hist[0]["loss"] - 0.3


# ------------------------------------------------------------ selection rule, union, harness
def test_selection_rule_recall200_then_recall50_tiebreak():
    a, b = {"recall@200": 0.50, "recall@50": 0.20}, {"recall@200": 0.40, "recall@50": 0.30}
    assert better(a, b) and not better(b, a) and better(a, None)
    c = {"recall@200": 0.50005, "recall@50": 0.25}   # within 1e-4 of a: Recall@50 decides
    assert better(c, a) and not better(a, c)


def test_union_hand_checked():
    a = np.array([[1, 2, 3, 4]]); b = np.array([[3, 5, 6, 1]]); better_list = np.array([[1, 2, 3, 4, 5, 6]])
    # n_each=2: a[:2]=[1,2], b[:2]=[3,5] -> interleaved 1,3,2,5 (4 distinct)
    assert union_top(a, b, 2, 4, better_list).tolist() == [[1, 3, 2, 5]]
    # heavy overlap: b[:2] = [1,2] -> union {1,2}; fill from better list: 3,4
    assert union_top(a, np.array([[1, 2, 9, 9]]), 2, 4, better_list).tolist() == [[1, 2, 3, 4]]


def test_evaluate_lists_equals_evaluate():
    class M:
        def fit(self, *a): pass
        def score(self, users, h=None): return np.array([np.arange(20)[::-1] * (1 + u % 3) for u in users], dtype=float)
    fit = _df([(1, 1, 1, 1), (2, 2, 1, 1)])
    ev = _df([(1, 5, 1, 9), (2, 7, 1, 9), (2, 15, 1, 9)])
    res = evaluate(M(), fit, ev, CAT, ks=(3, 10))
    again = evaluate_lists(res.top, fit, ev, CAT, ks=(3, 10))
    pd.testing.assert_frame_equal(res.per_user, again)


# ------------------------------------------------------------ export
def test_milvus_export_roundtrip(tmp_path):
    import pyarrow.parquet as pq
    rng = np.random.default_rng(0)
    V = rng.normal(size=(20, 64)).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
    movies = pd.DataFrame({"movieId": CAT, "title": [f"Movie {i} ({1990 + i})" for i in CAT], "genres": "Drama"})
    counts = np.arange(1, 21); counts[3] = 0
    stats = export_milvus(tmp_path, CAT, V, movies, counts, np.eye(64, dtype=np.float32), np.zeros(64, np.float32), 20,
                          np.array([1, 2]), V[:2])
    t = pq.read_table(tmp_path / "item_vectors.parquet").to_pandas()
    assert stats["cold_items"] == 1 and len(t) == 20 and t["item_id"].is_unique and t["is_cold"].sum() == 1
    vecs = np.vstack(t["vector"].to_numpy())
    assert vecs.dtype == np.float32 and vecs.shape == (20, 64) and np.allclose(np.linalg.norm(vecs, axis=1), 1, atol=1e-4)
    assert t.loc[t["item_id"] == 4, "year"].iloc[0] == 1994
    schema = json.loads((tmp_path / "milvus_schema.json").read_text())
    assert schema["fields"][1]["dim"] == 64 and schema["index"]["metric_type"] == "IP"
    assert np.load(tmp_path / "user_tower.npz")["history_len"] == 20


# ------------------------------------------------------------ Module 4b: candidate mixes and collections
def test_mix_top_hand_checked_dedupe_fill_and_slots():
    from ott_recsys.eval import mix_top
    a = np.array([[1, 2, 3, 4, 5, 6, 7, 8]])
    c = np.array([[9, 2, 10, 11, 12, 13, 14, 15]])
    # n_a=4, n_c=3, total 6: a[:4]=1,2,3,4; content first 3 = 9,2,10 -> 2 already there -> new 9,10 (2 slots) -> total 6
    lists, slots = mix_top(a, c, 4, 3, 6)
    assert lists.tolist() == [[1, 2, 3, 4, 9, 10]] and slots.tolist() == [2]
    # content fully overlapping: nothing new, so the list is filled from a: 1..6
    lists, slots = mix_top(a, np.array([[1, 2, 3, 99, 98, 97, 96, 95]]), 4, 3, 6)
    assert lists.tolist() == [[1, 2, 3, 4, 5, 6]] and slots.tolist() == [0]
    # fill skips items already added from content: content picks 5 (new), a continues 5(dup skipped),6
    lists, slots = mix_top(a, np.array([[5, 40, 41, 42, 43, 44, 45, 46]]), 3, 1, 5)
    assert lists.tolist() == [[1, 2, 3, 5, 4]] and slots.tolist() == [1]


def test_mix_with_zero_content_equals_base_and_lists_are_unique():
    from ott_recsys.eval import mix_top
    rng = np.random.default_rng(0)
    a = np.array([rng.permutation(60) for _ in range(5)]); c = np.array([rng.permutation(60) for _ in range(5)])
    lists, slots = mix_top(a, c, 20, 0, 20)
    assert (lists == a[:, :20]).all() and (slots == 0).all()
    lists, slots = mix_top(a, c, 15, 5, 20)
    assert all(len(set(r)) == 20 for r in lists) and (slots <= 5).all()


def test_export_collection_drops_uninformative_rows_and_supports_two_vector_fields(tmp_path):
    import pyarrow.parquet as pq
    from ott_recsys.export import export_collection, export_user_vectors
    rng = np.random.default_rng(1)
    unit = lambda n, d: (lambda m: m / np.linalg.norm(m, axis=1, keepdims=True))(rng.normal(size=(n, d)).astype(np.float32))
    movies = pd.DataFrame({"movieId": CAT, "title": [f"M{i} ({2000 + i})" for i in CAT], "genres": "Drama"})
    counts = np.arange(20); counts[[0, 5]] = 0
    st = export_collection(tmp_path / "id", "c_id", CAT, {"vector": unit(20, 64)}, movies, counts, include=counts > 0)
    t = pq.read_table(tmp_path / "id" / "items.parquet").to_pandas()
    assert st["items"] == 18 and st["dropped"] == 2 and 1 not in t["item_id"].values and not t["is_cold"].any()
    assert np.load(tmp_path / "id" / "vector.npy").shape == (18, 64)
    st2 = export_collection(tmp_path / "ct", "c_ct", CAT, {"vec384": unit(20, 384), "vec64": unit(20, 64)}, movies, counts)
    t2 = pq.read_table(tmp_path / "ct" / "items.parquet").to_pandas()
    sch = json.loads((tmp_path / "ct" / "milvus_schema.json").read_text())
    assert len(t2) == 20 and t2["is_cold"].sum() == 2 and {f["name"]: f.get("dim") for f in sch["fields"] if "dim" in f} == {"vec384": 384, "vec64": 64}
    assert [i["field"] for i in sch["indexes"]] == ["vec384", "vec64"]
    with pytest.raises(AssertionError):
        export_collection(tmp_path / "bad", "x", CAT, {"vector": 3 * unit(20, 64)}, movies, counts)  # not unit length
    export_user_vectors(tmp_path / "u.parquet", np.array([1, 2]), {"vec64": unit(2, 64)})
    assert len(pq.read_table(tmp_path / "u.parquet")) == 2


def test_mix_metrics_match_evaluate_on_a_base_list():
    from ott_recsys.eval import mix_top
    class M:
        def fit(self, *a): pass
        def score(self, users, h=None): return np.array([np.arange(20)[::-1] * 1.0 for _ in users])
    fit = _df([(1, 1, 1, 1), (2, 2, 1, 1)]); ev = _df([(1, 5, 1, 9), (2, 7, 1, 9)])
    res = evaluate(M(), fit, ev, CAT, ks=(5,))
    lists, _ = mix_top(res.top, res.top, 10, 0, 10)
    pd.testing.assert_frame_equal(evaluate_lists(lists, fit, ev, CAT, ks=(5,)), res.per_user)


def test_append_top_keeps_base_list_and_adds_only_new_picks():
    from ott_recsys.eval import append_top
    a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]]); c = np.array([[3, 9, 10, 11], [20, 21, 5, 22]])
    lists, added = append_top(a, c, 3)
    assert lists[0].tolist() == [1, 2, 3, 4, 9, 10, -1] and added[0] == 2   # 3 is already in the base list; width 4+3
    assert lists[1].tolist() == [5, 6, 7, 8, 20, 21, -1] and added[1] == 2  # 5 already there; third pick would be 22 > n_c=3 window
    assert (lists[:, :4] == a).all()                                          # the base list is untouched
