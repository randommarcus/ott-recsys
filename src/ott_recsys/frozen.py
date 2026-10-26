"""Scorers built from SAVED arrays only (no fitting): they load vectors/profiles that scripts/prepare_test.py computed from
train+validation histories and score users against the whole catalog. The harness (eval.evaluate) calls .score(user_ids).
Never touches the test portion (tests/test_module2.py scans it)."""
import numpy as np
import pandas as pd


class FrozenTwoTower:
    """Item vectors frozen; user vectors precomputed from history. Users without a vector fall back to popularity."""
    name = "two_tower"

    def __init__(self, item_vecs, user_ids, user_vecs, pop):
        self.V, self.pop = item_vecs, pop / max(pop.max(), 1.0)
        self.u_index = {u: i for i, u in enumerate(user_ids)}
        self.U = user_vecs

    def score(self, user_ids, query_hours=None):
        rows = np.array([self.u_index.get(u, -1) for u in user_ids])
        s = self.U[np.maximum(rows, 0)] @ self.V.T
        s[rows < 0] = self.pop[None, :]
        return s


class FrozenContent:
    """Centered-SBERT retrieval: profile (mean of liked-item vectors, unit length) . item vectors, popularity tie-break."""
    name = "content"

    def __init__(self, emb, user_ids, profiles, pop):
        self.emb, self.profiles = emb, profiles
        self.u_index = {u: i for i, u in enumerate(user_ids)}
        self.tiebreak = 1e-6 * pop / max(pop.max(), 1.0)

    def score(self, user_ids, query_hours=None):
        rows = np.array([self.u_index.get(u, -1) for u in user_ids])
        P = np.where((rows >= 0)[:, None], self.profiles[np.maximum(rows, 0)], 0.0)
        return P @ self.emb.T + self.tiebreak


class FrozenPop:
    name = "popularity"

    def __init__(self, counts):
        self.g = np.asarray(counts, dtype=np.float64)

    def score(self, user_ids, query_hours=None):
        return np.tile(self.g, (len(user_ids), 1))


def knn_state_for_history(neighbors_idx, neighbors_sim, history_df, catalog, train_counts) -> dict:
    """Assemble an ItemKNN state: FROZEN neighbour lists (computed from train) + a user x item likes matrix built from the
    given history (train+validation). No similarities are recomputed."""
    from scipy.sparse import csr_matrix
    users = np.sort(history_df["user_id"].unique())
    likes = history_df[history_df["label"] == 1]
    rows = pd.Index(users).get_indexer(likes["user_id"])
    cols = pd.Index(catalog).get_indexer(likes["item_id"])
    X = csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(len(users), len(catalog)))
    g = np.asarray(train_counts, dtype=np.float64)
    return {"X": X, "u_index": {u: i for i, u in enumerate(users)}, "neighbors": {0.0: (neighbors_idx, neighbors_sim)},
            "k_max": neighbors_idx.shape[1], "tiebreak": 1e-6 * g / max(g.max(), 1.0)}
