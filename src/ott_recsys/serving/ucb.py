"""UCB selection for the new-release slots, with the SAME rule as ott_recsys.bandit.strategy_keys("ucb_c"):
index = s/n + c*sqrt(ln N / n), N = max(total impressions on the eligible shelf, 2); an arm never shown has index 1e9 (tried first, random order).
c = 0.2 is the pre-registered constant. Counts n, s live in the DynamoDB table ott-bandit; s counts SIMULATED likes (SIMULATION, NOT AN A/B TEST)."""
import numpy as np

C_PREREGISTERED = 0.2


def ucb_keys(n: np.ndarray, s: np.ndarray, c: float = C_PREREGISTERED, rng: np.random.Generator | None = None) -> np.ndarray:
    n, s = np.asarray(n, float), np.asarray(s, float)
    N = max(n.sum(), 2.0)
    idx = s / np.maximum(n, 1.0) + c * np.sqrt(np.log(N) / np.maximum(n, 1.0))
    jitter = (rng or np.random.default_rng()).random(len(n))
    return np.where(n == 0, 1e9 + jitter, idx)


def pick(n, s, k: int, c: float = C_PREREGISTERED, rng: np.random.Generator | None = None) -> np.ndarray:
    """Positions (into the arm arrays) of the k arms to show, best first."""
    keys = ucb_keys(n, s, c, rng)
    k = min(k, len(keys))
    return np.argsort(-keys, kind="stable")[:k]
