"""Baseline recommenders. All share the interface used by eval.evaluate:
fit(fit_df, catalog) and score(user_ids, query_hours) -> [n_users, n_items]."""
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix


def _like_counts(fit_df: pd.DataFrame, catalog: np.ndarray) -> np.ndarray:
    idx = pd.Index(catalog).get_indexer(fit_df.loc[fit_df["label"] == 1, "item_id"])
    assert (idx >= 0).all(), "fit item missing from catalog"
    return np.bincount(idx, minlength=len(catalog)).astype(np.float64)


class RandomRec:
    name = "random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, fit_df, catalog):
        self.n = len(catalog)

    def score(self, user_ids, query_hours=None):
        return np.random.default_rng(self.seed).random((len(user_ids), self.n))


class GlobalPop:
    """Rank by number of likes (rating >= 4) in the fitting data."""
    name = "global_pop"

    def fit(self, fit_df, catalog):
        self.g = _like_counts(fit_df, catalog)

    def score(self, user_ids, query_hours=None):
        return np.tile(self.g, (len(user_ids), 1))


class HourPop:
    """Popularity at the user's request hour, smoothed toward global popularity.
    p_h(i) = (C_h(i) + lam * p_global(i)) / (N_h + lam), where C_h pools likes
    from hours within +-window (circular). shuffle_hours=True permutes the hour
    labels as a control: if it scores the same, the hour carries no signal."""
    name = "hour_pop"

    def __init__(self, lam: float = 500.0, window: int = 1, shuffle_hours: bool = False, seed: int = 0):
        self.lam, self.window, self.shuffle, self.seed = lam, window, shuffle_hours, seed

    def fit(self, fit_df, catalog):
        likes = fit_df[fit_df["label"] == 1]
        hours = likes["hour"].to_numpy()
        if self.shuffle:
            hours = np.random.default_rng(self.seed).permutation(hours)
        idx = pd.Index(catalog).get_indexer(likes["item_id"])
        counts = np.zeros((24, len(catalog)))
        np.add.at(counts, (hours, idx), 1.0)
        pooled = sum(np.roll(counts, s, axis=0) for s in range(-self.window, self.window + 1))
        p_global = counts.sum(axis=0) / counts.sum()
        n_h = pooled.sum(axis=1, keepdims=True)
        self.table = (pooled + self.lam * p_global) / (n_h + self.lam)  # [24, n_items]

    def score(self, user_ids, query_hours):
        return self.table[np.asarray(query_hours).astype(int)]


def knn_neighbors(fit_df: pd.DataFrame, catalog: np.ndarray, shrinks, k_max: int, block: int = 500) -> dict:
    """Item-item cosine neighbours from the binary user x item likes matrix.
    Works in blocks of items so the (items x items) table never exists in full.
    For every shrink value, keeps each item's k_max best neighbours, sorted by
    similarity (so any smaller k is just a prefix)."""
    users = np.sort(fit_df["user_id"].unique())
    u_index = {u: i for i, u in enumerate(users)}
    likes = fit_df[fit_df["label"] == 1]
    rows = likes["user_id"].map(u_index).to_numpy()
    cols = pd.Index(catalog).get_indexer(likes["item_id"])
    assert (cols >= 0).all(), "fit item missing from catalog"
    n_i = len(catalog)
    k_max = min(k_max, n_i - 1)
    X = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), n_i))
    XT = X.T.tocsr()
    cnt = np.asarray(X.sum(axis=0)).ravel().astype(np.float32)
    root = np.sqrt(cnt)
    out = {sh: (np.zeros((n_i, k_max), dtype=np.int32), np.zeros((n_i, k_max), dtype=np.float32))
           for sh in shrinks}
    for a in range(0, n_i, block):
        b = min(a + block, n_i)
        co = (XT[a:b] @ X).toarray()  # co-like counts, block x items
        co[np.arange(b - a), np.arange(a, b)] = 0.0  # an item is not its own neighbour
        norm = root[a:b, None] * root[None, :]
        for sh in shrinks:
            sim = np.divide(co, norm + sh, out=np.zeros_like(co), where=(norm + sh) > 0)
            part = np.argpartition(-sim, k_max - 1, axis=1)[:, :k_max]
            vals = np.take_along_axis(sim, part, axis=1)
            order = np.argsort(-vals, axis=1, kind="stable")
            out[sh][0][a:b] = np.take_along_axis(part, order, axis=1)
            out[sh][1][a:b] = np.take_along_axis(vals, order, axis=1)
        del co, norm
    g = _like_counts(fit_df, catalog)
    return {"X": X, "u_index": u_index, "neighbors": out, "k_max": k_max,
            "tiebreak": 1e-6 * g / max(g.max(), 1.0)}  # users with no likes fall back to popularity


class ItemKNN:
    """Item-item cosine similarity with shrinkage for rarely-liked items.
    score(u, j) = sum of sim(i, j) over the user's liked items i that are among
    j's k nearest neighbours."""
    name = "item_knn"

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, fit_df, catalog):
        self.use(knn_neighbors(fit_df, catalog, [self.shrink], self.k), self.shrink, self.k)

    def use(self, state: dict, shrink: float, k: int):
        """Adopt precomputed neighbours (lets tuning share one expensive pass)."""
        self.shrink, self.k = shrink, k
        idx, sim = state["neighbors"][shrink]
        k = min(k, idx.shape[1])
        n_i = idx.shape[0]
        S = csr_matrix((sim[:, :k].ravel(), idx[:, :k].ravel(), np.arange(0, n_i * k + 1, k)),
                       shape=(n_i, n_i))
        self.ST = S.T.tocsr()  # score = X_u @ S.T
        self.X, self.u_index, self.tiebreak = state["X"], state["u_index"], state["tiebreak"]
        return self

    def score(self, user_ids, query_hours=None):
        rows = np.array([self.u_index.get(u, -1) for u in user_ids])
        Xu = self.X[np.where(rows >= 0, rows, 0)]
        Xu = Xu.multiply((rows >= 0)[:, None].astype(np.float32)).tocsr()
        return (Xu @ self.ST).toarray() + self.tiebreak
