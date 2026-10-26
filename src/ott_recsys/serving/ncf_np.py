"""NeuMF forward pass in numpy for ONE user against many items (same maths as ott_recsys.ncf.NCFScorer.score, no pandas).

The user side (two 32-d vectors, after the offline fold-in) comes from the user's DynamoDB record. NaN = unknown user or item."""
import numpy as np

N_LAYERS = 3  # MLP 64-32-16


class NCFForward:
    def __init__(self, arrays: dict, item_ids, catalog):
        self.a = arrays
        pos = {int(m): i for i, m in enumerate(item_ids)}
        self.row_of_cat = np.array([pos.get(int(m), -1) for m in catalog], dtype=np.int32)  # catalog index -> NCF item row (-1 = unknown)

    def score(self, ug, um, cat_idx: np.ndarray) -> np.ndarray:
        cat_idx = np.asarray(cat_idx)
        out = np.full(len(cat_idx), np.nan, dtype=np.float32)
        if ug is None or um is None:
            return out
        rows = np.where(cat_idx >= 0, self.row_of_cat[np.maximum(cat_idx, 0)], -1)
        ok = np.flatnonzero(rows >= 0)
        if not len(ok):
            return out
        a, i = self.a, rows[ok]
        g = np.asarray(ug, np.float32)[None, :] * a["ig"][i]
        h = np.concatenate([np.broadcast_to(np.asarray(um, np.float32), (len(i), len(um))), a["im"][i]], axis=1)
        for l in range(N_LAYERS):
            h = np.maximum(h @ a[f"W{l}"] + a[f"b{l}"], 0.0)
        out[ok] = (np.concatenate([g, h], axis=1) @ a["Wo"] + a["bo"])[:, 0]
        return out
