"""Module 7b tests: tunable UCB, UCB1-Tuned, second-half windowing and first-half-only parameter selection.
THIS IS A SIMULATION, NOT AN A/B TEST."""
import numpy as np
import pytest

from ott_recsys.bandit import (make_arms, oracle_expected, pick_best, select_param, simulate, strategy_keys, window_metrics)


def _arms(n_arms, a, b, release=None):
    rel = np.zeros(n_arms) if release is None else np.asarray(release, float)
    arms = make_arms(np.arange(n_arms), rel, np.full(n_arms, 10), np.full(n_arms, 5), np.full(n_arms, 0.45), 0.45)
    arms["a"], arms["b"] = np.broadcast_to(np.asarray(a, float), n_arms).copy(), np.broadcast_to(np.asarray(b, float), n_arms).copy()
    return arms


# ------------------------------------------------------------ strategy formulas
def test_ucb_c_hand_checked_unseen_first_and_sqrt2_equals_ucb1():
    n, s = np.array([[2.0, 4.0, 0.0]]), np.array([[1.0, 3.0, 0.0]])
    k = strategy_keys("ucb_c", n, s, None, np.zeros(3), 0.0, 0.1, 0.45, np.random.default_rng(0), c=0.3)
    N = 6.0                                                                    # total impressions on the shelf
    assert k[0, 0] == pytest.approx(0.5 + 0.3 * np.sqrt(np.log(N) / 2)) and k[0, 1] == pytest.approx(0.75 + 0.3 * np.sqrt(np.log(N) / 4))
    assert k[0, 2] > 1e8                                                       # unseen arms are still tried first
    big = strategy_keys("ucb_c", n, s, None, np.zeros(3), 0.0, 0.1, 0.45, np.random.default_rng(5), c=np.sqrt(2.0))
    u1 = strategy_keys("ucb1", n, s, None, np.zeros(3), 0.0, 0.1, 0.45, np.random.default_rng(5))
    assert np.allclose(big[:, :2], u1[:, :2]) and (big[:, 2] > 1e8).all()      # c = sqrt(2) is exactly UCB1


def test_ucb_tuned_hand_checked_both_branches_of_the_min():
    # n=4, s=2, total 4: ln4/4 = 0.34657; V = 0.25 + sqrt(2*0.34657) = 1.0826 > 1/4 -> bonus sqrt(0.34657 * 0.25)
    k = strategy_keys("ucb_tuned", np.array([[4.0]]), np.array([[2.0]]), None, np.zeros(1), 0.0, 0.1, 0.45, np.random.default_rng(0))
    assert k[0, 0] == pytest.approx(0.5 + np.sqrt(np.log(4) / 4 * 0.25))
    # n=400, s=0: ln400/400 = 0.014979; V = sqrt(2*0.014979) = 0.17308 < 1/4 -> bonus sqrt(0.014979 * V)
    k = strategy_keys("ucb_tuned", np.array([[400.0]]), np.array([[0.0]]), None, np.zeros(1), 0.0, 0.1, 0.45, np.random.default_rng(0))
    assert k[0, 0] == pytest.approx(np.sqrt(np.log(400) / 400 * (np.sqrt(2 * np.log(400) / 400))))


def test_ucb_c_prior_formula():
    n, s, mu, w = np.array([[10.0, 0.0]]), np.array([[3.0, 0.0]]), np.array([0.6, 0.2]), 8.0
    k = strategy_keys("ucb_c_prior", n, s, None, mu, w, 0.1, 0.45, np.random.default_rng(0), c=0.4)
    for j in range(2):
        n_eff = n[0, j] + w
        assert k[0, j] == pytest.approx((s[0, j] + w * mu[j]) / n_eff + 0.4 * np.sqrt(np.log(10.0) / n_eff), abs=1e-6)


def test_smaller_c_explores_less_on_a_clear_problem():
    arms = _arms(30, np.r_[np.full(3, 60.0), np.full(27, 20.0)], np.r_[np.full(3, 20.0), np.full(27, 60.0)])
    req = np.full(3000, 10.0)
    like = {c: simulate(arms, req, "ucb_c", 2, 365, 8, 3, c=c)["reward"].sum() for c in (0.05, np.sqrt(2.0))}
    assert like[0.05] > like[np.sqrt(2.0)]                                     # textbook exploration wastes impressions here


# ------------------------------------------------------------ second-half windowing
def test_window_metrics_count_only_the_second_half():
    arms = _arms(6, 2.0, 2.0)
    req = np.full(400, 10.0)
    res = simulate(arms, req, "ucb_c", 2, 365, 4, 1, c=0.5, snapshot_at=200)
    orc = oracle_expected(arms, req, 2, 365, res["p"])
    m = window_metrics(res, orc, 200)
    assert (m["impressions"] == 2 * 200).all()                                          # exactly the second half's 200 requests x 2 slots
    assert np.allclose(m["likes"], res["reward"][200:].sum(0))
    assert np.array_equal(res["impressions_at_snapshot"].sum(1), np.full(4, 400.0))  # 200 requests x 2 slots before the snapshot
    assert (m["regret"] >= -1e-3).all() and m["gini"].shape == (4,)
    assert simulate(arms, req, "ucb_c", 2, 365, 2, 1)["impressions_at_snapshot"] is None


# ------------------------------------------------------------ parameter selection
def test_pick_best_flags_grid_edges():
    grid = [0.1, 0.2, 0.4]
    assert pick_best(grid, [0.5, 0.6, 0.55]) == (0.2, None)
    assert pick_best(grid, [0.7, 0.6, 0.55]) == (0.1, "lower")
    assert pick_best(grid, [0.5, 0.6, 0.9]) == (0.4, "upper")


def test_select_param_uses_only_the_requests_it_is_given_and_prefers_small_c():
    arms = _arms(20, np.r_[np.full(2, 70.0), np.full(18, 20.0)], np.r_[np.full(2, 20.0), np.full(18, 70.0)])
    first_half = np.full(1500, 10.0)
    grid = [0.02, 0.1, 0.5, 1.0, np.sqrt(2.0)]
    best, edge, scores = select_param(arms, first_half, "ucb_c", "c", grid, 2, 365, 6, 101)
    assert best < 1.0 and set(scores) == set(grid) and all(len(v) == 6 for v in scores.values())
    # identical when the same first half is passed again (nothing from any second half can reach the selection)
    again = select_param(arms, first_half.copy(), "ucb_c", "c", grid, 2, 365, 6, 101)
    assert again[0] == best and all(np.array_equal(scores[g], again[2][g]) for g in grid)
    # a different seed draws a different hidden truth, so scores differ (selection seeds are not the evaluation seeds)
    other = select_param(arms, first_half, "ucb_c", "c", grid, 2, 365, 6, 202)
    assert not np.array_equal(scores[grid[0]], other[2][grid[0]])
