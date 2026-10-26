"""Pure-numpy evaluator for xgboost tree ensembles saved as JSON (rank:ndcg models).

Why: xgboost needs scipy and is too big for a Lambda zip (see reports/module9_plan.md). Rules copied from xgboost: go LEFT when
feature < split_condition (float32 comparison); a missing (NaN) value follows the node's default direction; the score is the base score
plus the sum of the leaf values. tests/test_serving_trees.py checks it against xgboost's own predictions."""
import json
from pathlib import Path

import numpy as np


class TreeEnsemble:
    def __init__(self, left, right, feat, cond, dleft, base_score: float, feature_names):
        self.left, self.right, self.feat, self.cond, self.dleft = left, right, feat, cond, dleft  # all [n_trees, max_nodes]
        self.base_score = float(base_score)
        self.feature_names = list(feature_names)

    @classmethod
    def from_xgboost_json(cls, path) -> "TreeEnsemble":
        m = json.loads(Path(path).read_text())["learner"]
        trees = m["gradient_booster"]["model"]["trees"]
        n_max = max(len(t["left_children"]) for t in trees)
        T = len(trees)
        left, right = np.full((T, n_max), -1, np.int32), np.full((T, n_max), -1, np.int32)
        feat, cond = np.zeros((T, n_max), np.int32), np.zeros((T, n_max), np.float32)
        dleft = np.zeros((T, n_max), bool)
        for i, t in enumerate(trees):
            k = len(t["left_children"])
            left[i, :k], right[i, :k] = t["left_children"], t["right_children"]
            feat[i, :k], cond[i, :k] = t["split_indices"], t["split_conditions"]  # a leaf's value lives in split_conditions
            dleft[i, :k] = np.asarray(t["default_left"], dtype=bool)
        base = str(m["learner_model_param"]["base_score"]).strip("[]")
        return cls(left, right, feat, cond, dleft, float(base), m.get("feature_names", []))

    def save_npz(self, path) -> None:
        np.savez_compressed(path, left=self.left, right=self.right, feat=self.feat, cond=self.cond, dleft=self.dleft,
                            base_score=np.float64(self.base_score), feature_names=np.array(self.feature_names))

    @classmethod
    def load_npz(cls, path) -> "TreeEnsemble":
        z = np.load(path)
        return cls(z["left"], z["right"], z["feat"], z["cond"], z["dleft"], float(z["base_score"]), [str(x) for x in z["feature_names"]])

    @property
    def n_trees(self) -> int:
        return self.left.shape[0]

    def predict(self, X: np.ndarray, chunk: int = 20000) -> np.ndarray:
        X = np.ascontiguousarray(X, dtype=np.float32)
        out = np.empty(len(X), dtype=np.float32)
        t = np.arange(self.n_trees)[:, None]
        for a in range(0, len(X), chunk):
            xb = X[a:a + chunk]
            R = len(xb)
            rows = np.arange(R)[None, :]
            node = np.zeros((self.n_trees, R), np.int32)
            while True:
                lc = self.left[t, node]
                leaf = lc < 0
                if leaf.all():
                    break
                x = xb[rows, self.feat[t, node]]
                go_left = np.where(np.isnan(x), self.dleft[t, node], x < self.cond[t, node])
                node = np.where(leaf, node, np.where(go_left, lc, self.right[t, node]))
            out[a:a + R] = (self.cond[t, node].sum(0, dtype=np.float64) + self.base_score).astype(np.float32)
        return out
