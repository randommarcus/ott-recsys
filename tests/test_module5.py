"""Module 5 tests: NCF negative sampling, scorer, unknown handling, re-ranking metrics (hand-checked),
tie-breaking, blend, NaN-share table, candidate lists."""
import numpy as np
import pandas as pd
import pytest

from ott_recsys.ncf import (NCFScorer, better, build_training_data, make_model, model_arrays, sample_negatives,
                            score_candidates, train)
from ott_recsys.rerank import (all_orders, bucket_recall, build_candidate_lists, hit_matrix, metrics_from_ranks,
                               nan_share, rank_positions)


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label"])


# ------------------------------------------------------------ training data and sampler
def test_training_data_uses_only_given_likes_and_defines_known_ids():
    df = _df([(1, 10, 1), (1, 11, 0), (2, 10, 1), (3, 12, 0), (4, 12, 1)])
    d = build_training_data(df)
    assert d["users"].tolist() == [1, 2, 4] and d["items"].tolist() == [10, 12]   # user 3 / item 11 have no likes -> unknown
    assert len(d["pos_u"]) == 3 and d["item_counts"].tolist() == [2, 1]


def test_negatives_never_hit_a_users_own_positives_and_have_k_per_positive():
    rng = np.random.default_rng(0)
    n_users, n_items = 30, 40
    pos = [(u, i) for u in range(n_users) for i in rng.choice(n_items, 10, replace=False)]
    pu, pi = np.array([p[0] for p in pos], np.int32), np.array([p[1] for p in pos], np.int32)
    counts = np.bincount(pi, minlength=n_items)
    nu, ni = sample_negatives(pu, pi, n_items, 6, 0.5, counts, rng)
    positive_keys = set(zip(pu.tolist(), pi.tolist()))
    assert not any((u, i) in positive_keys for u, i in zip(nu.tolist(), ni.tolist()))
    assert len(nu) >= 0.99 * len(pu) * 6 and ni.max() < n_items and ni.min() >= 0


def test_negative_sampling_distributions_popularity_vs_uniform():
    n_items, n = 20, 200_000
    rng = np.random.default_rng(1)
    pu = np.arange(n, dtype=np.int32)                      # one positive per user, so exclusions barely matter
    pi = rng.integers(0, n_items, n).astype(np.int32)
    counts = np.arange(1, n_items + 1) * 10
    _, ni_pop = sample_negatives(pu, pi, n_items, 1, 1.0, counts, rng)
    target = counts.astype(float) ** 0.75
    target /= target.sum()
    assert np.abs(np.bincount(ni_pop, minlength=n_items) / len(ni_pop) - target).max() < 0.01
    _, ni_uni = sample_negatives(pu, pi, n_items, 1, 0.0, counts, rng)
    assert np.abs(np.bincount(ni_uni, minlength=n_items) / len(ni_uni) - 1 / n_items).max() < 0.01
    _, ni_mix = sample_negatives(pu, pi, n_items, 4, 0.5, counts, rng)  # 2 popularity + 2 uniform draws per positive
    assert len(ni_mix) > 3.9 * n


# ------------------------------------------------------------ scorer
def test_numpy_scorer_matches_tensorflow_and_unknowns_are_nan():
    import tensorflow as tf
    nu, ni = 12, 15
    m = make_model(nu, ni)
    users, items = np.arange(100, 100 + nu), np.arange(500, 500 + ni)
    sc = NCFScorer(model_arrays(m), users, items)
    u = np.random.default_rng(0).integers(0, nu, 50)
    i = np.random.default_rng(1).integers(0, ni, 50)
    tf_out = m(tf.constant(u, tf.int32), tf.constant(i, tf.int32)).numpy()
    np.testing.assert_allclose(sc.score(users[u], items[i]), tf_out, atol=1e-5)
    s = sc.score(np.array([100, 999, 100, 999]), np.array([500, 500, 888, 888]))
    assert not np.isnan(s[0]) and np.isnan(s[1:]).all()
    uk, ik = sc.known(np.array([100, 999, 100]), np.array([500, 500, 888]))
    assert uk.tolist() == [True, False, True] and ik.tolist() == [True, True, False]


def test_score_candidates_marks_padding_and_unknown_as_nan():
    m = make_model(3, 4)
    sc = NCFScorer(model_arrays(m), np.array([1, 2, 3]), np.array([10, 20, 30, 40]))
    catalog = np.array([10, 20, 30, 40, 50])          # item 50 has no embedding
    out = score_candidates(sc, np.array([1, 9]), np.array([[0, 4, -1], [0, 1, 2]]), catalog)
    assert not np.isnan(out[0, 0]) and np.isnan(out[0, 1]) and np.isnan(out[0, 2])   # unknown item, padding
    assert np.isnan(out[1]).all()                                                   # unknown user


def test_shuffled_users_is_a_permutation_of_user_embeddings_only():
    m = make_model(20, 10)
    sc = NCFScorer(model_arrays(m), np.arange(20), np.arange(10))
    sh = sc.shuffled_users(0)
    assert sorted(map(tuple, np.round(sh.a["ug"], 6))) == sorted(map(tuple, np.round(sc.a["ug"], 6)))
    assert (sh.a["ig"] == sc.a["ig"]).all() and (sh.a["W0"] == sc.a["W0"]).all()
    assert not np.allclose(sh.score(np.arange(20), np.zeros(20, int)), sc.score(np.arange(20), np.zeros(20, int)))
    np.testing.assert_array_equal(sc.shuffled_users(0).a["ug"], sh.a["ug"])      # deterministic


def test_ncf_learns_group_structure():
    rng = np.random.default_rng(0)
    rows = [(u, int(i), 1) for u in range(40) for i in rng.choice(np.arange(0, 10) + (0 if u < 20 else 10), 6, replace=False)]
    data = build_training_data(_df(rows))
    sc, hist, _ = train(data, neg_k=4, pop_frac=0.5, batch=256, fixed_epochs=40, log=lambda m: None)  # ~5 steps/epoch on this toy set
    same = sc.score(np.repeat(np.arange(0, 20), 10), np.tile(np.arange(0, 10), 20)).mean()
    other = sc.score(np.repeat(np.arange(0, 20), 10), np.tile(np.arange(10, 20), 20)).mean()
    assert same > other + 3.0 and hist[-1]["loss"] < hist[0]["loss"] - 0.3


def test_selection_rule_ndcg_then_recall():
    a, b = {"ndcg@10": 0.10, "recall@20": 0.2}, {"ndcg@10": 0.09, "recall@20": 0.3}
    assert better(a, b) and not better(b, a) and better(a, None)
    c = {"ndcg@10": 0.10005, "recall@20": 0.25}
    assert better(c, a) and not better(a, c)


# ------------------------------------------------------------ re-ranking metrics (hand-checked)
def test_metrics_hand_checked_with_nan_and_padding():
    cand = np.array([[5, 3, 8, 1, 9, -1]])
    hit = hit_matrix(cand, [np.array([3, 9, 100])])           # truth has 3 likes; 100 is not in the list
    ncf = np.array([[0.1, 0.9, np.nan, 0.5, 0.7, np.nan]])
    pop = np.zeros(200)
    r = all_orders(cand, hit, pop, {"x": ncf})["ncf_x"]
    ordered = cand[0][np.argsort(r[0])]
    assert ordered[:5].tolist() == [3, 9, 1, 5, 8] and ordered[5] == -1   # scored by logit, NaN item after, padding last
    m = metrics_from_ranks(r, hit, np.array([3]), [7]).iloc[0]
    d1, d2, d3 = 1 / np.log2(2), 1 / np.log2(3), 1 / np.log2(4)
    assert m["e2e_ndcg10"] == pytest.approx((d1 + d2) / (d1 + d2 + d3))   # ideal for 3 likes has 3 hits
    assert m["e2e_recall20"] == pytest.approx(2 / 3)
    assert m["rr_ndcg10"] == pytest.approx(1.0) and m["rr_recall20"] == pytest.approx(1.0)  # restricted to the 2 in-list likes


def test_ties_do_not_follow_the_original_list_order_and_are_seed_deterministic():
    cand = np.arange(60)[None, :]
    hit = np.zeros((1, 60), bool)
    same = np.zeros((1, 60))
    r1 = all_orders(cand, hit, np.zeros(60), {"x": same}, seed=1)["ncf_x"]
    r2 = all_orders(cand, hit, np.zeros(60), {"x": same}, seed=1)["ncf_x"]
    r3 = all_orders(cand, hit, np.zeros(60), {"x": same}, seed=2)["ncf_x"]
    assert (r1 == r2).all() and not (r1 == r3).all() and not (r1[0] == np.arange(60)).all()


def test_ncf_order_ignores_candidate_list_order_when_scores_distinct():
    rng = np.random.default_rng(0)
    cand = rng.permutation(100)[None, :50]
    score = rng.random((1, 50))
    hit = np.zeros((1, 50), bool)
    perm = rng.permutation(50)
    top = lambda c, s: set(c[0][np.argsort(all_orders(c, hit, np.zeros(100), {"x": s})["ncf_x"][0])][:10])
    assert top(cand, score) == top(cand[:, perm], score[:, perm])


def test_unknown_user_falls_back_to_popularity_order():
    cand = np.array([[0, 1, 2, 3]])
    pop = np.array([5.0, 50.0, 1.0, 20.0])
    r = all_orders(cand, np.zeros((1, 4), bool), pop, {"x": np.full((1, 4), np.nan)})["ncf_x"]
    assert cand[0][np.argsort(r[0])].tolist() == [1, 3, 0, 2]


def test_blend_hand_checked():
    cand = np.array([[0, 1, 2, 3]])                          # tt order: a=0, b=1, c=2, d=3
    ncf = np.array([[0.8, 0.5, 0.9, 0.1]])                   # ncf order: c, a, b, d
    r = all_orders(cand, np.zeros((1, 4), bool), np.zeros(4), {"x": ncf})
    # blend = 0.5*tt_rank/3 + 0.5*ncf_rank/3: a .167, b .5, c .333, d 1.0 -> a, c, b, d
    assert cand[0][np.argsort(r["blend_x"][0])].tolist() == [0, 2, 1, 3]


def test_rank_positions_orders_valid_then_scored_then_nan():
    valid = np.array([[True, True, True, False]])
    r = rank_positions(valid, np.array([[1.0, np.nan, 3.0, 9.0]]), np.zeros((1, 4)))
    assert r[0].tolist() == [1, 2, 0, 3]


# ------------------------------------------------------------ NaN share, buckets, candidate lists
def test_nan_share_by_lane_hand_checked():
    cand = np.array([[1, 2, 3, 4]])                           # slots 0-1 main lane, 2-3 new-release lane
    hit = np.array([[True, False, True, True]])
    item_known = np.array([True, True, True, False, True])   # indexed by catalog idx: catalog idx 3 is unknown
    out = nan_share(cand, hit, 2, item_known, np.array([True]))
    assert out["main_lane"] == {"likes_inside": 1, "on_item_nan": 0, "on_user_nan": 0, "on_either_nan": 0, "slots": 2, "slots_item_nan": 0}
    assert out["new_release_lane"]["likes_inside"] == 2 and out["new_release_lane"]["on_item_nan"] == 1
    assert out["new_release_lane"]["slots_item_nan"] == 1
    out2 = nan_share(cand, hit, 2, item_known, np.array([False]))   # user unknown: every in-list like is unscorable
    assert out2["main_lane"]["on_either_nan"] == 1 and out2["new_release_lane"]["on_either_nan"] == 2


def test_bucket_recall_separates_items_without_train_likes():
    cand = np.array([[0, 1, 2, 3, 4, 5]])
    counts = np.array([0, 3, 6, 9, 12, 15])
    hit = np.array([[True, True, True, True, True, True]])
    rank_best = {"m": np.arange(6)[None, :].astype(np.int16)}         # everything in the top 20
    names, out = bucket_recall(rank_best, hit, cand, counts, k=20)
    assert out["_n_positives"][0] == 1 and sum(out["_n_positives"]) == 6 and all(v == 1.0 for v in out["m"])


def test_candidate_lists_append_lane_without_taking_main_slots():
    main = np.array([[1, 2, 3, 4, 5]])
    lane = np.array([[3, 9, 8, 7]])                                # 3 is already in the main list
    cand, added = build_candidate_lists(main, lane, n_main=5, n_lane=3)
    assert cand.shape == (1, 8) and cand[0, :5].tolist() == [1, 2, 3, 4, 5] and cand[0, 5:].tolist() == [9, 8, -1] and added[0] == 2
    assert len(set(cand[0][cand[0] >= 0])) == 7                    # no duplicates
