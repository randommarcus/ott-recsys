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


class ItemKNN:
    """Item-item cosine similarity on the binary user x item likes matrix, with
    shrinkage for rarely-liked items. score(u, j) = sum of sim(i, j) over the
    user's liked items i among j's k nearest neighbours."""
    name = "item_knn"

    def __init__(self, k: int = 50, shrink: float = 10.0):
        self.k, self.shrink = k, shrink

    def fit(self, fit_df, catalog):
        users = np.sort(fit_df["user_id"].unique())
        self.u_index = {u: i for i, u in enumerate(users)}
        likes = fit_df[fit_df["label"] == 1]
        rows = likes["user_id"].map(self.u_index).to_numpy()
        cols = pd.Index(catalog).get_indexer(likes["item_id"])
        assert (cols >= 0).all()
        n_i = len(catalog)
        X = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), n_i))
        co = (X.T @ X).tocoo()
        keep = co.row != co.col
        cnt = np.asarray(X.sum(axis=0)).ravel()
        sim = co.data[keep] / (np.sqrt(cnt[co.row[keep]] * cnt[co.col[keep]]) + self.shrink)
        S = csr_matrix((sim, (co.row[keep], co.col[keep])), shape=(n_i, n_i))
        r, c, d = [], [], []  # keep each item's top-k neighbours
        for j in range(n_i):
            a, b = S.indptr[j], S.indptr[j + 1]
            vals, nbr = S.data[a:b], S.indices[a:b]
            if len(vals) > self.k:
                top = np.argpartition(-vals, self.k)[: self.k]
                vals, nbr = vals[top], nbr[top]
            r.append(np.full(len(vals), j)); c.append(nbr); d.append(vals)
        self.S = csr_matrix((np.concatenate(d), (np.concatenate(r), np.concatenate(c))), shape=(n_i, n_i))
        self.X = X
        g = _like_counts(fit_df, catalog)
        self.tiebreak = 1e-6 * g / max(g.max(), 1.0)  # users with no likes fall back to popularity

    def score(self, user_ids, query_hours=None):
        rows = np.array([self.u_index.get(u, -1) for u in user_ids])
        Xu = self.X[np.where(rows >= 0, rows, 0)].toarray() * (rows >= 0)[:, None]
        return (csr_matrix(Xu) @ self.S.T).toarray() + self.tiebreak
