"""Module 8 tests: NCF fold-in, frozen scorers, candidate lists for the test preparation. No test data is ever involved."""
import numpy as np
import pandas as pd
import pytest

from ott_recsys import testprep
from ott_recsys.baselines import ContentRetrieval, ItemKNN, knn_neighbors
from ott_recsys.eval import evaluate
from ott_recsys.frozen import FrozenContent, FrozenPop, FrozenTwoTower, knn_state_for_history
from ott_recsys.ncf import NCFScorer, build_training_data, fold_in_users, train
from ott_recsys.rerank import build_candidate_lists
from ott_recsys.twotower import TwoTowerRetrieval


def _df(rows):
    return pd.DataFrame(rows, columns=["user_id", "item_id", "label", "timestamp"])


# ------------------------------------------------------------ NCF fold-in
def _frozen_ncf():
    rng = np.random.default_rng(0)
    rows = [(u, int(i), 1, 0) for u in range(1, 41) for i in rng.choice(np.arange(0, 10) + (0 if u <= 20 else 10), 6, replace=False)]
    sc, _, _ = train(build_training_data(_df(rows).assign(label=1)), neg_k=4, pop_frac=0.5, batch=256, fixed_epochs=30, log=lambda m: None)
    return sc


def test_fold_in_updates_users_only_and_gives_new_users_embeddings():
    frozen = _frozen_ncf()
    new_user, group_a = 999, np.arange(0, 10)
    rng = np.random.default_rng(1)
    pos = [(999, int(i)) for i in rng.choice(group_a, 6, replace=False)] + [(999, 777)]        # 777 = item the frozen model has never seen
    pos += [(u, int(i)) for u in range(1, 6) for i in rng.choice(group_a, 4, replace=False)]    # extra likes for existing users
    folded, hist = fold_in_users(frozen, [p[0] for p in pos], [p[1] for p in pos], epochs=25, batch=128, lr=0.05, seed=3, log=lambda m: None)  # toy data has ~50 steps, so a larger lr here
    assert 999 in set(folded.user_ids) and 999 not in set(frozen.user_ids)
    for k in ("ig", "im", "W0", "b0", "W1", "b1", "W2", "b2", "Wo", "bo"):                       # every item/MLP parameter is untouched
        assert np.array_equal(folded.a[k], frozen.a[k])
    assert np.array_equal(folded.item_ids, frozen.item_ids)
    uf, uo = folded._u.get_indexer(np.arange(1, 6)), frozen._u.get_indexer(np.arange(1, 6))
    assert not np.allclose(folded.a["ug"][uf], frozen.a["ug"][uo])                              # existing users' embeddings did move
    same = folded.score(np.full(10, 999), np.arange(0, 10)).mean()
    other = folded.score(np.full(10, 999), np.arange(10, 20)).mean()
    assert same > other + 1.0 and hist[-1]["loss"] < hist[0]["loss"]                           # the new user learned their group
    assert np.isnan(folded.score(np.array([999]), np.array([777]))[0])                         # the unknown item still cannot be scored


def test_fold_in_is_deterministic():
    frozen = _frozen_ncf()
    pos_u, pos_i = [999] * 5 + [1] * 5, [0, 1, 2, 3, 4, 0, 1, 2, 3, 4]
    a, _ = fold_in_users(frozen, pos_u, pos_i, epochs=3, batch=64, seed=7, log=lambda m: None)
    b, _ = fold_in_users(frozen, pos_u, pos_i, epochs=3, batch=64, seed=7, log=lambda m: None)
    assert np.array_equal(a.a["ug"], b.a["ug"]) and np.array_equal(a.a["um"], b.a["um"])


# ------------------------------------------------------------ frozen scorers equal the originals
def _world(n_items=60, n_users=12, seed=0):
    rng = np.random.default_rng(seed)
    catalog = np.arange(100, 100 + n_items)
    V = rng.normal(size=(n_items, 8)).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
    emb = rng.normal(size=(n_items, 8)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    rows = [(u, int(catalog[rng.integers(0, n_items)]), int(rng.random() < 0.6), 10 * k) for u in range(1, n_users) for k in range(15)]
    rows += [(n_users + 5, int(catalog[0]), 0, 1)]                                                   # a user with no likes at all
    return catalog, V, emb, _df(rows), rng


def test_frozen_two_tower_and_content_match_the_fitted_originals():
    catalog, V, emb, hist, rng = _world()
    W, b = rng.normal(size=(8, 8)).astype(np.float32), np.zeros(8, np.float32)
    tt = TwoTowerRetrieval(V, W, b, 20)
    tt.fit(hist, catalog)
    ids = np.array(sorted(tt.u_index))
    counts = np.bincount(pd.Index(catalog).get_indexer(hist.loc[hist.label == 1, "item_id"]), minlength=len(catalog)).astype(float)
    ft = FrozenTwoTower(V, ids, tt.U[[tt.u_index[u] for u in ids]], counts)
    users = np.array([1, 2, 5, 17, 999])                                                              # includes a no-likes user and an unknown user
    np.testing.assert_allclose(ft.score(users), tt.score(users), atol=1e-6)
    ct = ContentRetrieval(emb)
    ct.fit(hist, catalog)
    cids = np.array(sorted(ct.u_index))
    fc = FrozenContent(emb, cids, ct.profiles[[ct.u_index[u] for u in cids]], counts)
    np.testing.assert_allclose(fc.score(users), ct.score(users), atol=1e-6)
    np.testing.assert_allclose(FrozenPop(counts).score(users[:2]), np.tile(counts, (2, 1)))


def test_frozen_knn_uses_frozen_neighbours_but_the_new_history():
    catalog, V, emb, hist, rng = _world(n_items=40, n_users=30)
    state = knn_neighbors(hist, catalog, [0.0], 10)
    orig = ItemKNN().use(state, 0.0, 10)
    counts = np.bincount(pd.Index(catalog).get_indexer(hist.loc[hist.label == 1, "item_id"]), minlength=len(catalog))
    fs = knn_state_for_history(state["neighbors"][0.0][0], state["neighbors"][0.0][1], hist, catalog, counts)
    same = ItemKNN().use(fs, 0.0, 10)
    users = np.array([1, 2, 3])
    np.testing.assert_allclose(same.score(users), orig.score(users), atol=1e-6)                      # same history -> identical scores
    liked_before = set(hist[(hist.user_id == 1) & (hist.label == 1)].item_id)
    new_item = next(int(c) for c in catalog if int(c) not in liked_before)
    hist2 = pd.concat([hist, _df([(1, new_item, 1, 999)])], ignore_index=True)
    fs2 = knn_state_for_history(state["neighbors"][0.0][0], state["neighbors"][0.0][1], hist2, catalog, counts)
    moved = ItemKNN().use(fs2, 0.0, 10).score(users)
    assert not np.allclose(moved[0], orig.score(users)[0])                                           # user 1's new like changes their scores
    np.testing.assert_allclose(moved[1:], orig.score(users)[1:], atol=1e-6)                          # nobody else changes


# ------------------------------------------------------------ candidate lists
def test_candidate_lists_match_the_module5_harness_path():
    rng = np.random.default_rng(2)
    n_items = 320
    catalog = np.arange(1000, 1000 + n_items)
    V = rng.normal(size=(n_items, 8)).astype(np.float32); V /= np.linalg.norm(V, axis=1, keepdims=True)
    emb = rng.normal(size=(n_items, 8)).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    users = np.arange(1, 25)
    rows = [(u, int(catalog[i]), int(rng.random() < 0.7), 5 * k) for u in users for k, i in enumerate(rng.choice(n_items // 2, 20, replace=False))]
    hist = _df(rows)                                                                                 # users only touch the first half of the catalog
    W, b = rng.normal(size=(8, 8)).astype(np.float32), np.zeros(8, np.float32)
    tt = TwoTowerRetrieval(V, W, b, 20)
    tt.fit(hist, catalog)
    ct = ContentRetrieval(emb)
    ct.fit(hist, catalog)
    counts = np.bincount(pd.Index(catalog).get_indexer(hist.loc[hist.label == 1, "item_id"]), minlength=n_items).astype(float)
    lane_pool = np.flatnonzero(counts == 0)
    seen = testprep.seen_map(hist, catalog)
    tt_ids, ct_ids = np.array(sorted(tt.u_index)), np.array(sorted(ct.u_index))
    cand = testprep.candidate_lists(users, seen, V, tt_ids, tt.U[[tt.u_index[u] for u in tt_ids]], ct_ids, ct.profiles[[ct.u_index[u] for u in ct_ids]],
                                    emb, lane_pool, counts, chunk=7)
    ev = _df([(u, int(catalog[0]), 1, 9999) for u in users])
    main = evaluate(tt, hist, ev, catalog, ks=(10, 50, 200))
    lane = evaluate(ct, hist, ev, catalog, ks=(10, 50, 200), candidate_items=catalog[lane_pool])
    expected, _ = build_candidate_lists(main.top, lane.top, 200, 30)
    assert cand.shape == (len(users), 230) and (cand == expected).all()
    assert all(not set(cand[r][cand[r] >= 0]) & set(seen[u]) for r, u in enumerate(users))        # nothing already seen is recommended
    assert set(cand[:, 200:][cand[:, 200:] >= 0]) <= set(lane_pool)                                 # lane picks come only from the lane pool


def test_two_tower_scores_are_nan_for_the_lane_and_unknown_items():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(20, 4)).astype(np.float32)
    known = np.ones(20, bool); known[3] = False
    cand = np.full((1, 230), -1); cand[0, :3] = [1, 3, 5]; cand[0, 200] = 7                       # a main slot with unknown item 3 and one lane slot
    s = testprep.two_tower_scores(np.array([1]), cand, V, np.array([1]), np.ones((1, 4), np.float32), known)
    assert s[0, 0] == pytest.approx(V[1].sum()) and np.isnan(s[0, 1]) and s[0, 2] == pytest.approx(V[5].sum())
    assert np.isnan(s[0, 200]) and np.isnan(s[0, 3])


def test_history_frame_rehearsal_is_train_only_real_adds_validation():
    tr, va = _df([(1, 1, 1, 1), (1, 2, 1, 2)]), _df([(1, 3, 1, 3)])
    assert testprep.history_frame(tr, va, rehearsal=True)["item_id"].tolist() == [1, 2]
    assert testprep.history_frame(tr, va, rehearsal=False)["item_id"].tolist() == [1, 2, 3]
