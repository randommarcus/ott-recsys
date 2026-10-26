"""Request-time context features and simulated device, with an explicit no-future-information rule.

LEAKAGE RULE. A user's "request time" is the timestamp of their first interaction in the evaluation
period. Only that TIME is used (never its item or label). Every context feature is computed from
interactions with timestamp STRICTLY LESS than the request time, so interactions at the same second,
and everything after, can never reach a feature.

DEVICE IS SIMULATED. MovieLens has no device. Everything about devices in this module is synthetic and
must be labelled as such wherever it is reported.

Never touches the test portion (tests/test_module2.py scans it)."""
import numpy as np
import pandas as pd

SESSION_GAP_S = 1800  # 30 minutes, as in Module 1
DEVICES = ("tv", "mobile", "web", "tablet")
DEVICE_P = (0.40, 0.35, 0.15, 0.10)


def request_times(val: pd.DataFrame, users) -> np.ndarray:
    """First evaluation-period timestamp per user. Uses the timestamp column only."""
    return val.groupby("user_id")["timestamp"].min().reindex(users).to_numpy().astype(np.int64)


def session_state(ts: np.ndarray, label: np.ndarray, t_req: int, gap_s: int = SESSION_GAP_S) -> dict:
    """Session state at the request, from interactions strictly before t_req (ts sorted ascending).
    With bulk rating, "session" means "the user is in the middle of a rating spree"."""
    n = int(np.searchsorted(ts, t_req, side="left"))  # count of interactions with ts < t_req
    if n == 0:
        return {"n": 0, "mins_since_last": np.nan, "in_session": 0.0, "length": 0, "start": 0, "like_rate": np.nan}
    since = t_req - ts[n - 1]
    if since > gap_s:
        return {"n": n, "mins_since_last": since / 60.0, "in_session": 0.0, "length": 0, "start": n, "like_rate": np.nan}
    breaks = np.flatnonzero(np.diff(ts[:n]) > gap_s)
    start = int(breaks[-1] + 1) if len(breaks) else 0
    return {"n": n, "mins_since_last": since / 60.0, "in_session": 1.0, "length": n - start, "start": start,
            "like_rate": float(label[start:n].mean())}


def pop_keys(item_idx: np.ndarray, ts: np.ndarray, n_items: int):
    """Sorted (item, timestamp) keys of train likes, for popularity "as of" a time."""
    keys = np.sort((item_idx.astype(np.int64) << 32) | ts.astype(np.int64))
    start = np.searchsorted(keys, np.arange(n_items + 1, dtype=np.int64) << 32)
    return keys, start


def pop_asof(keys: np.ndarray, start: np.ndarray, item_idx: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Number of train likes of each item with timestamp strictly before t (same shape as item_idx)."""
    key = (item_idx.astype(np.int64) << 32) | np.asarray(t, dtype=np.int64)
    return np.searchsorted(keys, key, side="left") - start[item_idx]


def genre_matrix(movies: pd.DataFrame, catalog: np.ndarray):
    """[n_items, n_genres] multi-hot (float32) aligned with the catalog, and the genre names."""
    g = movies.set_index("movieId").reindex(catalog)["genres"].fillna("(no genres listed)")
    names = sorted({x for s in g for x in s.split("|") if x != "(no genres listed)"})
    pos = {n: i for i, n in enumerate(names)}
    m = np.zeros((len(catalog), len(names)), dtype=np.float32)
    for r, s in enumerate(g):
        for x in s.split("|"):
            if x in pos:
                m[r, pos[x]] = 1.0
    return m, names


def genre_affinity(share: np.ndarray, item_multihot: np.ndarray) -> np.ndarray:
    """Mean of the user's genre share over the item's genres. share [G]; item_multihot [..., G] -> [...]."""
    return (item_multihot @ share) / np.maximum(item_multihot.sum(-1), 1.0)


def compact(mat: np.ndarray, keep: np.ndarray, fill):
    """Move kept entries to the front of each row (order preserved); the rest becomes `fill`."""
    order = np.argsort(~keep, axis=1, kind="stable")
    out = np.take_along_axis(mat, order, axis=1)
    k = np.take_along_axis(keep, order, axis=1)
    return np.where(k, out, fill)


# ---------------------------------------------------------------- simulated device
def simulate_device(n_users: int, seed: int = 0) -> np.ndarray:
    """SIMULATED device code per request, independent of everything else."""
    return np.random.default_rng(seed).choice(len(DEVICES), n_users, p=DEVICE_P)


def device_onehot(codes: np.ndarray) -> np.ndarray:
    return (codes[:, None] == np.arange(len(DEVICES))[None, :]).astype(np.float32)


def shuffled_device(codes: np.ndarray, seed: int = 1) -> np.ndarray:
    """Control: the same device column reassigned among users (same marginals, no link to behaviour)."""
    return np.random.default_rng(seed).permutation(codes)


def synthetic_truth(truth: list, device: np.ndarray, drama_no_comedy: np.ndarray, drop_p: float = 0.6, seed: int = 2):
    """SYNTHETIC SCENARIO (pipeline demonstration, not a finding): on a mobile request each validation like of a
    Drama-without-Comedy title is removed with probability drop_p. Only labels change; likes are only ever removed."""
    rng = np.random.default_rng(seed)
    out, removed = [], 0
    for t, d in zip(truth, device):
        if DEVICES[d] == "mobile":
            drop = drama_no_comedy[t] & (rng.random(len(t)) < drop_p)
            removed += int(drop.sum())
            out.append(t[~drop])
        else:
            out.append(t)
    return out, removed
