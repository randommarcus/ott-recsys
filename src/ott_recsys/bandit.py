"""Semi-synthetic bandit simulation for the new-release lane.

THIS IS A SIMULATION, NOT AN A/B TEST. Standard replay (Li et al. 2010) cannot be used on MovieLens:
the ratings were not logged by a random policy (users chose what to rate) and with ~3,300 arms almost every
logged event would be discarded. Instead: arms enter at their release date, each arm has a hidden like-rate
drawn from a Beta posterior built from its REAL future ratings, rewards are Bernoulli, and the request stream
uses the real session-start timestamps. The reward is "liked, given rated"; it ignores click probability.

Common random numbers: for one seed every strategy sees the same hidden like-rates and the same reward coins,
so differences between strategies are paired.

Never touches the test portion (tests/test_module2.py scans it)."""
import numpy as np
import pandas as pd

STRATEGIES = ("random", "greedy", "eps_greedy", "ucb1", "ucb1_prior", "thompson", "prior_only")
NEVER = np.inf


def session_starts(val_df: pd.DataFrame, gap_s: int = 1800) -> np.ndarray:
    """Sorted timestamps of the first interaction of every session (30-minute rule) = the request stream."""
    d = val_df.sort_values(["user_id", "timestamp"], kind="mergesort")
    gap = d.groupby("user_id")["timestamp"].diff()
    return np.sort(d.loc[gap.isna() | (gap > gap_s), "timestamp"].to_numpy().astype(np.int64))


def make_arms(item_ids, release_ts, n, likes, prior_mu, global_rate: float, pseudo: float = 5.0, min_ratings: int = 1) -> dict:
    """Arms with >= min_ratings window ratings, sorted by release time. Hidden like-rate posterior =
    Beta(rate*pseudo + likes, (1-rate)*pseudo + n - likes): real ratings, shrunk toward the global rate."""
    item_ids, release_ts, n, likes, prior_mu = (np.asarray(a) for a in (item_ids, release_ts, n, likes, prior_mu))
    keep = (n >= min_ratings) & np.isfinite(release_ts)
    order = np.argsort(release_ts[keep], kind="stable")
    sel = lambda a: a[keep][order]
    return {"item_id": sel(item_ids), "release_ts": sel(release_ts).astype(np.float64), "n": sel(n), "likes": sel(likes),
            "a": global_rate * pseudo + sel(likes), "b": (1 - global_rate) * pseudo + sel(n) - sel(likes), "prior_mu": sel(prior_mu),
            "global_rate": global_rate}


def eligible_range(release_ts: np.ndarray, t, shelf_days: float):
    """Arms are sorted by release time, so the arms on the shelf at time t form one contiguous range [lo, hi)."""
    hi = np.searchsorted(release_ts, t, side="right")
    lo = np.searchsorted(release_ts, np.asarray(t) - shelf_days * 86400, side="right")
    return lo, hi


def draw_truth(arms: dict, n_seeds: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, 0]).beta(arms["a"], arms["b"], size=(n_seeds, len(arms["a"])))


def draw_coins(n_req: int, k: int, n_seeds: int, seed: int) -> np.ndarray:
    return np.random.default_rng([seed, 1]).random((n_req, k, n_seeds))


def strategy_keys(strategy: str, n, s, p_true, mu, w: float, eps: float, m0: float, rng) -> np.ndarray:
    """Score of each eligible arm for each seed; the top-k keys are shown. n, s, p_true: [seeds, eligible arms]."""
    S, m = n.shape
    if strategy == "random":
        return rng.random((S, m))
    noise = rng.random((S, m)) * 1e-9  # random tie-breaking
    if strategy == "oracle":
        return p_true + noise
    if strategy == "prior_only":
        return np.broadcast_to(mu, (S, m)) + noise
    if strategy == "thompson":
        return rng.beta(1.0 + s, 1.0 + n - s)
    N = np.maximum(n.sum(1, keepdims=True), 2.0)  # impressions so far among the arms on the shelf
    if strategy in ("greedy", "eps_greedy"):
        keys = np.where(n > 0, s / np.maximum(n, 1), m0) + noise
        if strategy == "eps_greedy":
            explore = rng.random(S) < eps
            keys = np.where(explore[:, None], rng.random((S, m)), keys)
        return keys
    if strategy == "ucb1":
        idx = s / np.maximum(n, 1) + np.sqrt(2.0 * np.log(N) / np.maximum(n, 1))
        return np.where(n == 0, 1e9 + rng.random((S, m)), idx)  # every unseen arm is tried first
    if strategy == "ucb1_prior":
        n_eff = n + w  # the content prior counts as w pseudo-ratings at the prior mean
        return (s + w * mu) / n_eff + np.sqrt(2.0 * np.log(N) / n_eff) + noise
    raise ValueError(strategy)


def estimate(strategy: str, n, s, mu, w: float, m0: float):
    """The strategy's own current belief about each arm's like-rate (used for 'time to find the top arms')."""
    if strategy == "prior_only":
        return np.broadcast_to(mu, n.shape)
    if strategy == "ucb1_prior":
        return (s + w * mu) / (n + w)
    return np.where(n > 0, s / np.maximum(n, 1), m0)


def oracle_expected(arms: dict, req_t, k: int, shelf_days: float, p: np.ndarray) -> np.ndarray:
    """[requests, seeds] expected reward of always showing the true best k eligible arms."""
    lo, hi = eligible_range(arms["release_ts"], req_t, shelf_days)
    out = np.zeros((len(req_t), p.shape[0]))
    for r, (a, b) in enumerate(zip(lo, hi)):
        m = b - a
        if m:
            kk = min(k, m)
            out[r] = np.partition(p[:, a:b], m - kk, axis=1)[:, m - kk:].sum(1)
    return out


def simulate(arms: dict, req_t, strategy: str, k: int, shelf_days: float, n_seeds: int, seed: int, w: float = 0.0,
             eps: float = 0.1, probe_every: int = 100, top_m: int = 10, top_hits: int = 5, min_pool: int = 50, p=None, coins=None) -> dict:
    A, R = len(arms["a"]), len(req_t)
    p = draw_truth(arms, n_seeds, seed) if p is None else p
    coins = draw_coins(R, k, n_seeds, seed) if coins is None else coins
    mu, m0 = arms["prior_mu"], arms["global_rate"]
    lo_all, hi_all = eligible_range(arms["release_ts"], req_t, shelf_days)
    n, s = np.zeros((n_seeds, A)), np.zeros((n_seeds, A))
    rng = np.random.default_rng([seed, 2, STRATEGIES.index(strategy) if strategy in STRATEGIES else 99])
    rows = np.arange(n_seeds)[:, None]
    reward, expected = np.zeros((R, n_seeds), np.float32), np.zeros((R, n_seeds), np.float32)
    found_at = np.full(n_seeds, NEVER)
    served = 0
    for r in range(R):
        lo, hi = lo_all[r], hi_all[r]
        m = hi - lo
        if m == 0:
            continue
        served += 1
        kk = min(k, m)
        keys = strategy_keys(strategy, n[:, lo:hi], s[:, lo:hi], p[:, lo:hi], mu[lo:hi], w, eps, m0, rng)
        cols = lo + np.argpartition(-keys, kk - 1, axis=1)[:, :kk]
        pc = np.take_along_axis(p, cols, axis=1)
        rew = coins[r, :kk, :].T < pc
        n[rows, cols] += 1
        s[rows, cols] += rew
        reward[r], expected[r] = rew.sum(1), pc.sum(1)
        if r % probe_every == 0 and m >= max(top_m, min_pool):  # found >= top_hits of the true top-m? (only once the shelf is big enough that chance is low)
            true_top = np.zeros((n_seeds, m), bool)
            np.put_along_axis(true_top, np.argpartition(-p[:, lo:hi], top_m - 1, axis=1)[:, :top_m], True, axis=1)
            est = estimate(strategy, n[:, lo:hi], s[:, lo:hi], mu[lo:hi], w, m0) + np.random.default_rng(r).random((n_seeds, m)) * 1e-9
            mine = np.argpartition(-est, top_m - 1, axis=1)[:, :top_m]
            hits = np.take_along_axis(true_top, mine, axis=1).sum(1)
            found_at = np.where(np.isinf(found_at) & (hits >= top_hits), r, found_at)
    return {"reward": reward, "expected": expected, "impressions": n, "found_at": found_at, "p": p, "served_requests": served}


# ---------------------------------------------------------------- metrics
def gini(x: np.ndarray) -> float:
    x = np.sort(np.asarray(x, dtype=float))
    if x.sum() == 0:
        return 0.0
    n = len(x)
    return float(2.0 * (np.arange(1, n + 1) * x).sum() / (n * x.sum()) - (n + 1) / n)


def run_metrics(res: dict, oracle_exp: np.ndarray, checkpoints=(0.10, 0.25, 0.50, 1.0)) -> dict:
    """Per-seed metrics from one simulation run."""
    R = res["reward"].shape[0]
    cum, cum_exp, cum_or = res["reward"].cumsum(0), res["expected"].cumsum(0), oracle_exp.cumsum(0)
    imp = res["impressions"]
    p = res["p"]
    top_arms = p >= np.quantile(p, 0.9, axis=1, keepdims=True)
    S = imp.shape[0]
    return {"likes": cum[-1], "impressions": imp.sum(1), "regret": (cum_or - cum_exp)[-1],
            "cum_at": {c: cum[max(int(c * R) - 1, 0)] for c in checkpoints},
            "regret_at": {c: (cum_or - cum_exp)[max(int(c * R) - 1, 0)] for c in checkpoints},
            "gini": np.array([gini(imp[i]) for i in range(S)]), "never_shown": (imp == 0).mean(1),
            "top10pct_share": (imp * top_arms).sum(1) / np.maximum(imp.sum(1), 1), "found_at": res["found_at"]}


def mean_ci(x: np.ndarray):
    x = np.asarray(x, dtype=float)
    m, se = x.mean(), x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
    return float(m), float(m - 1.96 * se), float(m + 1.96 * se)


# ---------------------------------------------------------------- content prior (fitted on PRE-cutoff movies only)
def eb_weight(mu, n, likes, bounds=(0.5, 500.0)) -> float:
    """Empirical-Bayes prior weight w (pseudo-ratings): maximise the beta-binomial likelihood of the observed
    likes given each movie's prior mean mu, true rate ~ Beta(w*mu, w*(1-mu))."""
    from scipy.optimize import minimize_scalar
    from scipy.stats import betabinom
    mu = np.clip(np.asarray(mu, dtype=float), 0.02, 0.98)
    f = lambda lw: -betabinom.logpmf(likes, n, np.exp(lw) * mu, np.exp(lw) * (1 - mu)).sum()
    return float(np.exp(minimize_scalar(f, bounds=(np.log(bounds[0]), np.log(bounds[1])), method="bounded").x))


def fit_prior(emb: np.ndarray, item_idx, n, likes, alpha: float = 10.0, seed: int = 0):
    """Ridge from centered text embeddings to like-rate, fitted on pre-cutoff movies only. Returns the model,
    the empirical-Bayes weight (from cross-validated predictions) and the cross-validated correlation."""
    from sklearn.linear_model import Ridge
    from sklearn.model_selection import KFold, cross_val_predict
    X, rate = emb[np.asarray(item_idx)], np.asarray(likes) / np.asarray(n)
    cv = np.clip(cross_val_predict(Ridge(alpha=alpha), X, rate, cv=KFold(5, shuffle=True, random_state=seed)), 0.05, 0.95)
    return Ridge(alpha=alpha).fit(X, rate), eb_weight(cv, n, likes), float(np.corrcoef(cv, rate)[0, 1])


# ---------------------------------------------------------------- reserving slots in a top-20 list
def display_with_reserved(main_hits: np.ndarray, reserved_positions, n_slots: int = 20):
    """main_hits [users, >= n_slots]: hit flags of main-lane items in ranking order. Reserved (1-based) slot
    positions get no main item; the remaining positions are filled in ranking order. Returns the displayed
    hit matrix [users, n_slots] (a reserved slot is never a main-lane hit) and the reserved mask."""
    reserved = np.zeros(n_slots, bool)
    reserved[np.asarray(reserved_positions, dtype=int) - 1] = True
    free = np.flatnonzero(~reserved)
    out = np.zeros((len(main_hits), n_slots), dtype=bool)
    out[:, free] = main_hits[:, :len(free)]
    return out, reserved


def displayed_metrics(disp: np.ndarray, n_truth: np.ndarray):
    """End-to-end NDCG@10 and Recall@20 of a displayed list (same definitions as Modules 5-6)."""
    disc = 1.0 / np.log2(np.arange(2, 12))
    ideal = np.concatenate([[0.0], np.cumsum(disc)])
    dcg = (disp[:, :10] * disc).sum(1)
    nt = np.asarray(n_truth)
    return dcg / ideal[np.minimum(nt, 10)], disp[:, :20].sum(1) / nt
