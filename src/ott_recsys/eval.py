"""Evaluation harness, written once and reused by every model.

Generic: works on plain columns user_id, item_id, label. A model needs
    .fit(fit_df, catalog)                      -> None
    .score(user_ids, query_hours)              -> array [n_users, len(catalog)]
Higher score = recommended earlier. Items a user already saw in fit_df are
removed before ranking.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd

KS = (10, 50, 200)
VIEWS = ("all", "warm", "cold")  # cold = truth items never seen in fit_df


@dataclass
class EvalResult:
    per_user: pd.DataFrame  # one row per evaluated user, NaN if user has no truth in that view
    coverage: float         # share of the catalog appearing in anyone's top-cov_k
    top: np.ndarray         # [n_users, 200] ranked catalog indices
    users: np.ndarray
    catalog: np.ndarray


def _top_one(s: np.ndarray, k: int) -> np.ndarray:
    """Indices of the k highest scores; ties go to the lower index (identical
    to a full stable sort, but only sorts the candidates near the cut)."""
    n = len(s)
    if k < n:
        thr = np.partition(s, n - k)[n - k]  # k-th largest value
        cand = np.flatnonzero(s >= thr)
        if len(cand) <= 20 * k:
            return cand[np.argsort(-s[cand], kind="stable")][:k]
    return np.argsort(-s, kind="stable")[:k]


def rank_top(scores: np.ndarray, seen: list[np.ndarray], k_max: int) -> np.ndarray:
    """Top-k_max catalog indices per user with already-seen items removed.
    Deterministic: ties break toward the lower item index."""
    out = np.empty((len(scores), k_max), dtype=np.int64)
    for i, row in enumerate(scores):
        s = np.asarray(row, dtype=np.float64).copy()
        s[seen[i]] = -np.inf
        out[i] = _top_one(s, k_max)
    return out


def evaluate(model, fit_df: pd.DataFrame, eval_df: pd.DataFrame, catalog: np.ndarray,
             query_hours: pd.Series | None = None, ks=KS, ndcg_k: int = 10,
             cov_k: int = 10, chunk: int = 500, candidate_items: np.ndarray | None = None) -> EvalResult:
    cat_index = pd.Index(catalog)
    truth_df = eval_df[eval_df["label"] == 1]
    users = np.sort(truth_df["user_id"].unique())
    truth = {u: cat_index.get_indexer(g["item_id"].to_numpy())
             for u, g in truth_df.groupby("user_id")}
    assert all((t >= 0).all() for t in truth.values()), "truth item missing from catalog"
    warm_items = set(cat_index.get_indexer(fit_df["item_id"].unique()))
    seen_map = {u: cat_index.get_indexer(g["item_id"].to_numpy()) for u, g in fit_df.groupby("user_id")}
    seen = [seen_map.get(u, np.empty(0, dtype=np.int64)) for u in users]
    if candidate_items is not None:  # rank only this pool: everything else is excluded like seen items
        outside = np.setdiff1d(np.arange(len(catalog)), cat_index.get_indexer(candidate_items))
        seen = [np.union1d(x, outside) for x in seen]

    hours = None if query_hours is None else query_hours.reindex(users).to_numpy()
    k_max = min(max(max(ks), ndcg_k, cov_k), len(catalog))
    top = np.empty((len(users), k_max), dtype=np.int64)
    for a in range(0, len(users), chunk):  # score in chunks: users x items may not fit in RAM
        b = min(a + chunk, len(users))
        scores = model.score(users[a:b], None if hours is None else hours[a:b])
        assert scores.shape == (b - a, len(catalog)), "score matrix has wrong shape"
        top[a:b] = rank_top(scores, seen[a:b], k_max)
        del scores

    discount = 1.0 / np.log2(np.arange(2, ndcg_k + 2))
    cols = {f"{v}_{m}": np.full(len(users), np.nan)
            for v in VIEWS for m in [f"recall@{k}" for k in ks] + [f"ndcg@{ndcg_k}"]}
    for r, u in enumerate(users):
        t_all = set(truth[u].tolist())
        views = {"all": t_all,
                 "warm": {x for x in t_all if x in warm_items},
                 "cold": {x for x in t_all if x not in warm_items}}
        for v, t in views.items():
            if not t:
                continue
            for k in ks:
                cols[f"{v}_recall@{k}"][r] = len(t.intersection(top[r, :k].tolist())) / len(t)
            gains = np.array([1.0 if x in t else 0.0 for x in top[r, :ndcg_k]])
            idcg = discount[: min(len(t), ndcg_k)].sum()
            cols[f"{v}_ndcg@{ndcg_k}"][r] = float((gains * discount).sum() / idcg)

    coverage = len(np.unique(top[:, :cov_k])) / len(catalog)
    return EvalResult(pd.DataFrame(cols, index=users), coverage, top, users, catalog)


def summarize(per_user: pd.DataFrame, coverage: float, n_boot: int = 1000, seed: int = 0) -> pd.DataFrame:
    """Mean per metric with a 95% bootstrap interval over users."""
    rng = np.random.default_rng(seed)
    rows = {}
    for col in per_user.columns:
        v = per_user[col].dropna().to_numpy()
        if len(v) == 0:
            rows[col] = (np.nan, np.nan, np.nan, 0)
            continue
        if n_boot:
            means = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
            lo, hi = np.percentile(means, [2.5, 97.5])
        else:
            lo = hi = np.nan
        rows[col] = (v.mean(), lo, hi, len(v))
    out = pd.DataFrame(rows, index=["mean", "lo", "hi", "n_users"]).T
    out.loc["coverage@10"] = (coverage, np.nan, np.nan, np.nan)
    return out


def cold_report(fit_df: pd.DataFrame, eval_df: pd.DataFrame) -> dict:
    """How much of the evaluation portion is about items not seen while fitting."""
    seen = set(fit_df["item_id"])
    new = ~eval_df["item_id"].isin(seen)
    pos = eval_df["label"] == 1
    return {"rows": len(eval_df), "items": eval_df["item_id"].nunique(),
            "items_not_in_fit": int(eval_df.loc[new, "item_id"].nunique()),
            "rows_on_new_items": int(new.sum()), "share_rows_on_new_items": float(new.mean()),
            "likes": int(pos.sum()), "likes_on_new_items": int((new & pos).sum())}


def first_hour(df: pd.DataFrame) -> pd.Series:
    """Query hour per user = hour of their first interaction in df. Only the
    timestamp is used, never the label or the item."""
    d = df.sort_values(["user_id", "timestamp", "item_id"])
    return d.groupby("user_id")["hour"].first()


def paired_diff(a: pd.Series, b: pd.Series, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    """Mean of (a - b) over users with a 95% bootstrap interval. Both models are
    scored on the same users, so this is much sharper than comparing two
    separate intervals."""
    d = (a - b).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n_boot, len(d)))].mean(axis=1)
    return float(d.mean()), *map(float, np.percentile(means, [2.5, 97.5]))
