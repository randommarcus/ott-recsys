"""Module 7 tests: bandit strategies, simulator invariants, metrics, prior fitting, reserved slots.
THIS IS A SIMULATION, NOT AN A/B TEST."""
import numpy as np
import pandas as pd
import pytest

from ott_recsys.bandit import (displayed_metrics, display_with_reserved, draw_truth, eb_weight, eligible_range, estimate,
                               fit_prior, gini, make_arms, mean_ci, oracle_expected, run_metrics, session_starts, simulate,
                               strategy_keys)


def _arms(release, a=None, b=None, mu=None, g=0.45):
    n = len(release)
    arms = make_arms(np.arange(n), np.array(release, float), np.full(n, 10), np.full(n, 5), np.full(n, g) if mu is None else mu, g)
    if a is not None:
        arms["a"], arms["b"] = np.array(a, float), np.array(b, float)
    return arms


# ------------------------------------------------------------ requests, arms, eligibility
def test_session_starts_hand_checked():
    val = pd.DataFrame({"user_id": [1, 1, 1, 2, 2], "timestamp": [100, 160, 160 + 1801, 50, 60]})
    assert session_starts(val).tolist() == [50, 100, 1961]   # user 1: 100 and 1961 start sessions (gap > 30 min); user 2: 50


def test_make_arms_sorts_by_release_filters_and_builds_the_posterior():
    arms = make_arms(np.array([10, 11, 12]), np.array([300.0, 100.0, np.nan]), np.array([4, 20, 9]), np.array([2, 10, 3]),
                     np.array([.5, .5, .5]), 0.4, pseudo=5.0, min_ratings=5)
    assert arms["item_id"].tolist() == [11]                          # item 10 has < 5 ratings, item 12 has no release date
    assert arms["a"][0] == pytest.approx(0.4 * 5 + 10) and arms["b"][0] == pytest.approx(0.6 * 5 + 20 - 10)


def test_eligible_range_is_the_arms_on_the_shelf():
    rel = np.array([0.0, 100.0, 200.0, 300.0])
    day = 86400
    ts = np.array([150.0, 250.0, 100.0 + 2 * day])      # the last request is two days after every release: nothing on the shelf
    lo, hi = eligible_range(rel, ts, shelf_days=1)
    assert lo.tolist() == [0, 0, 4] and hi.tolist() == [2, 3, 4]
    for t, a, b in zip(ts, lo, hi):                       # agrees with a brute-force scan
        assert list(range(a, b)) == [i for i, r in enumerate(rel) if r <= t < r + day]


# ------------------------------------------------------------ strategies
def test_ucb1_index_hand_checked_and_unseen_arms_come_first():
    n, s = np.array([[2.0, 4.0, 0.0]]), np.array([[1.0, 3.0, 0.0]])
    keys = strategy_keys("ucb1", n, s, None, np.zeros(3), 0.0, 0.1, 0.45, np.random.default_rng(0))
    N = 6.0
    assert keys[0, 0] == pytest.approx(1 / 2 + np.sqrt(2 * np.log(N) / 2)) and keys[0, 1] == pytest.approx(3 / 4 + np.sqrt(2 * np.log(N) / 4))
    assert keys[0, 2] > 1e8                                           # the unseen arm outranks every seen arm


def test_greedy_uses_the_prior_mean_for_unseen_arms_and_ucb_prior_formula():
    n, s = np.array([[10.0, 0.0]]), np.array([[3.0, 0.0]])
    g = strategy_keys("greedy", n, s, None, np.zeros(2), 0.0, 0.1, 0.45, np.random.default_rng(0))
    assert g[0, 0] == pytest.approx(0.3, abs=1e-6) and g[0, 1] == pytest.approx(0.45, abs=1e-6)
    mu, w = np.array([0.6, 0.2]), 8.0
    k = strategy_keys("ucb1_prior", n, s, None, mu, w, 0.1, 0.45, np.random.default_rng(0))
    for j in range(2):
        n_eff = n[0, j] + w
        assert k[0, j] == pytest.approx((s[0, j] + w * mu[j]) / n_eff + np.sqrt(2 * np.log(10.0) / n_eff), abs=1e-6)


def test_epsilon_greedy_explores_about_eps_of_the_time():
    S, m = 20000, 4
    n, s = np.ones((S, m)) * 5, np.tile([4.0, 0.0, 0.0, 0.0], (S, 1))
    keys = strategy_keys("eps_greedy", n, s, None, np.zeros(m), 0.0, 0.2, 0.45, np.random.default_rng(1))
    exploit_choice_rate = (keys.argmax(1) == 0).mean()                # exploit -> arm 0; explore -> uniform over 4 arms
    assert exploit_choice_rate == pytest.approx(0.8 + 0.2 / 4, abs=0.02)


def test_thompson_concentrates_with_data_and_is_uniform_without():
    rng = np.random.default_rng(0)
    big = strategy_keys("thompson", np.array([[1000.0, 1000.0]]), np.array([[800.0, 200.0]]), None, np.zeros(2), 0, 0.1, 0.45, rng)
    assert big[0, 0] > big[0, 1]
    draws = strategy_keys("thompson", np.zeros((50000, 1)), np.zeros((50000, 1)), None, np.zeros(1), 0, 0.1, 0.45, rng)
    assert draws.mean() == pytest.approx(0.5, abs=0.01)               # Beta(1, 1)


def test_estimate_matches_each_strategys_belief():
    n, s, mu = np.array([[10.0, 0.0]]), np.array([[3.0, 0.0]]), np.array([0.6, 0.2])
    assert estimate("ucb1", n, s, mu, 8.0, 0.45).tolist() == [[0.3, 0.45]]
    assert estimate("ucb1_prior", n, s, mu, 8.0, 0.45)[0].tolist() == pytest.approx([(3 + 8 * 0.6) / 18, 0.2])
    assert estimate("prior_only", n, s, mu, 8.0, 0.45)[0].tolist() == [0.6, 0.2]


# ------------------------------------------------------------ simulator invariants
def test_no_arm_is_shown_before_release_or_after_its_shelf_life():
    arms = _arms([100.0, 200.0, 300.0])
    day = 86400
    req = np.array([50, 150, 150, 250, 350, 350, 350], dtype=float)  # t=50: nothing released yet
    res = simulate(arms, req, "random", k=1, shelf_days=100 / day, n_seeds=5, seed=0)
    imp = res["impressions"]
    assert (imp[:, 0] == 2).all() and (imp[:, 1] == 1).all() and (imp[:, 2] == 3).all()   # exactly the requests inside each arm's window
    assert res["served_requests"] == 6


def test_oracle_has_zero_regret_and_random_matches_the_mean_eligible_rate():
    rng = np.random.default_rng(0)
    arms = _arms(np.sort(rng.uniform(0, 1000, 40)), a=rng.uniform(1, 8, 40), b=rng.uniform(1, 8, 40))
    req = np.sort(rng.uniform(1000, 1100, 4000))
    p = draw_truth(arms, 6, 3)
    orc = oracle_expected(arms, req, 2, 365, p)
    res = simulate(arms, req, "oracle", k=2, shelf_days=365, n_seeds=6, seed=3, p=p)
    assert np.allclose(res["expected"], orc, atol=1e-5)
    rnd = simulate(arms, req, "random", k=2, shelf_days=365, n_seeds=6, seed=3, p=p)
    assert rnd["expected"].sum() / (2 * len(req) * 6) == pytest.approx(p.mean(), abs=0.02)
    assert (orc >= rnd["expected"] - 1e-6).all()


def test_same_seed_is_identical_different_seed_differs_and_truth_is_shared_across_strategies():
    arms = _arms(np.arange(20.0), a=np.full(20, 2.0), b=np.full(20, 2.0))
    req = np.linspace(30, 200, 500)
    a1 = simulate(arms, req, "ucb1", 2, 365, 4, 7)
    a2 = simulate(arms, req, "ucb1", 2, 365, 4, 7)
    b = simulate(arms, req, "ucb1", 2, 365, 4, 8)
    c = simulate(arms, req, "thompson", 2, 365, 4, 7)
    assert np.array_equal(a1["reward"], a2["reward"]) and not np.array_equal(a1["reward"], b["reward"])
    assert np.array_equal(a1["p"], c["p"])                            # common random numbers: same hidden like-rates per seed


def test_learning_strategies_beat_random_on_a_clear_problem():
    arms = _arms(np.zeros(12), a=np.r_[np.full(2, 60.0), np.full(10, 20.0)], b=np.r_[np.full(2, 20.0), np.full(10, 60.0)])
    req = np.full(3000, 10.0)
    rew = {s: simulate(arms, req, s, 2, 365, 10, 1, w=6.0)["reward"].sum() for s in ("random", "ucb1", "thompson", "eps_greedy")}
    assert rew["ucb1"] > rew["random"] * 1.3 and rew["thompson"] > rew["random"] * 1.3 and rew["eps_greedy"] > rew["random"]


def test_time_to_find_top_arms_prior_only_with_good_vs_misleading_prior():
    n = 12
    a = np.r_[2000.0, np.ones(n - 1)]
    b = np.r_[1.0, np.full(n - 1, 2000.0)]                      # arm 0 is the one clearly best arm
    good = _arms(np.zeros(n), a=a, b=b, mu=np.r_[0.9, np.full(n - 1, 0.1)])
    bad = _arms(np.zeros(n), a=a, b=b, mu=np.r_[0.1, np.full(n - 1, 0.9)])
    req = np.full(300, 10.0)
    kw = dict(k=2, shelf_days=365, n_seeds=3, seed=0, probe_every=50, top_m=1, top_hits=1, min_pool=1)
    assert (simulate(good, req, "prior_only", **kw)["found_at"] == 0).all()    # found at the very first probe
    assert np.isinf(simulate(bad, req, "prior_only", **kw)["found_at"]).all()  # never: the prior is wrong and never learns


# ------------------------------------------------------------ metrics
def test_gini_hand_checked_and_run_metrics_shapes():
    assert gini(np.ones(10)) == pytest.approx(0.0, abs=1e-12)
    assert gini(np.r_[np.zeros(9), 5.0]) == pytest.approx(0.9)       # (n-1)/n when one arm gets everything
    arms = _arms(np.zeros(15), a=np.full(15, 3.0), b=np.full(15, 3.0))
    req = np.full(400, 5.0)
    res = simulate(arms, req, "ucb1", 2, 365, 5, 0)
    m = run_metrics(res, oracle_expected(arms, req, 2, 365, res["p"]))
    assert (m["impressions"] == 800).all() and (m["regret"] >= -1e-3).all() and m["gini"].shape == (5,) and set(m["cum_at"]) == {0.1, 0.25, 0.5, 1.0}


def test_mean_ci():
    m, lo, hi = mean_ci(np.array([1.0, 2.0, 3.0, 4.0]))
    assert m == 2.5 and lo < m < hi


# ------------------------------------------------------------ content prior
def test_empirical_bayes_weight_recovers_the_truth():
    rng = np.random.default_rng(0)
    mu = rng.uniform(0.25, 0.75, 6000)
    n = rng.integers(20, 200, 6000)
    p = rng.beta(10.0 * mu, 10.0 * (1 - mu))
    assert eb_weight(mu, n, rng.binomial(n, p)) == pytest.approx(10.0, rel=0.25)


def test_fit_prior_learns_a_linear_signal_from_embeddings():
    rng = np.random.default_rng(1)
    emb = rng.normal(size=(1500, 12)).astype(np.float32)
    true_p = 1 / (1 + np.exp(-(emb[:, 0] * 0.8)))
    n = np.full(1500, 60)
    likes = rng.binomial(n, true_p)
    model, w, corr = fit_prior(emb, np.arange(1500), n, likes)
    assert corr > 0.6 and w > 3 and model.predict(emb[:3]).shape == (3,)


# ------------------------------------------------------------ reserving slots
def test_reserved_slots_shift_positions_exactly_and_never_count_as_hits():
    hits = np.zeros((1, 25), bool)
    hits[0, [0, 1, 4, 9, 17]] = True                                 # main-lane hits at ranks 1, 2, 5, 10, 18
    end, res_end = display_with_reserved(hits, [19, 20])
    assert res_end.sum() == 2 and not end[0, [18, 19]].any() and (end[0, :18] == hits[0, :18]).all()   # first 18 positions untouched
    inside, res_in = display_with_reserved(hits, [5, 10])
    # free positions (0-based) are 0-3, 5-8, 10-19: main ranks 1,2 stay; rank 5 -> position 5; rank 10 -> 11; rank 18 -> 19
    assert np.flatnonzero(inside[0]).tolist() == [0, 1, 5, 11, 19]
    assert not inside[0, [4, 9]].any() and res_in.sum() == 2         # a reserved slot is never a main-lane hit


def test_displayed_metrics_hand_checked_and_end_reservation_keeps_ndcg_at_10():
    hits = np.zeros((1, 25), bool)
    hits[0, [0, 2]] = True
    nd, rc = displayed_metrics(hits[:, :20], np.array([2]))
    d1, d3 = 1.0, 1 / np.log2(4)
    assert nd[0] == pytest.approx((d1 + d3) / (1 + 1 / np.log2(3))) and rc[0] == pytest.approx(1.0)
    end, _ = display_with_reserved(hits, [19, 20])
    nd_end, _ = displayed_metrics(end, np.array([2]))
    assert nd_end[0] == nd[0]                                         # NDCG@10 is unchanged when the reserved slots are at the end
    hits2 = np.zeros((1, 25), bool); hits2[0, 3] = True              # a hit at main rank 4 moves to position 5 behind a reserved slot 2
    mid, _ = display_with_reserved(hits2, [2])
    assert displayed_metrics(mid, np.array([1]))[0][0] == pytest.approx(1 / np.log2(6))
    assert displayed_metrics(hits2[:, :20], np.array([1]))[0][0] == pytest.approx(1 / np.log2(5))
