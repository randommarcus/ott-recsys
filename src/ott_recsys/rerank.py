"""Evaluate orderings of candidate lists (the 230-slot lists: 200 main-lane + up to 30 new-release-lane).

Everything here works on catalog indices; candidate lists are [users, width] int arrays padded with -1.
A 'score' is higher-is-better; NaN means "cannot score" and ranks after every scored item. Exact ties are
broken at random (seeded), never by the original list order, so the retriever's order cannot leak in.
Never touches the test portion (tests/test_module2.py scans it)."""
import numpy as np
import pandas as pd

from .eval import append_top

DISC = 1.0 / np.log2(np.arange(2, 12))
IDCG = np.concatenate([[0.0], np.cumsum(DISC)])  # IDCG[k] = best possible DCG@10 with k positives


def build_candidate_lists(main_top: np.ndarray, lane_top: np.ndarray, n_main: int = 200, n_lane: int = 30):
    """Main-lane top n_main, then up to n_lane lane picks not already there (appended, never replacing)."""
    return append_top(main_top[:, :n_main], lane_top, n_lane)


def hit_matrix(cand: np.ndarray, truth: list) -> np.ndarray:
    hit = np.zeros(cand.shape, dtype=bool)
    for r, t in enumerate(truth):
        hit[r] = np.isin(cand[r], t) & (cand[r] >= 0)
    return hit


def rank_positions(valid: np.ndarray, score: np.ndarray, rand: np.ndarray) -> np.ndarray:
    """rank[u, j] = position (0 = best) of candidate j. Order: valid first, scored before NaN, higher score
    first, then the random tie-break."""
    U, W = valid.shape
    rank = np.empty((U, W), dtype=np.int16)
    nan = np.isnan(score) | ~valid
    s = np.where(nan, 0.0, score)
    ar = np.arange(W)
    for r in range(U):
        rank[r, np.lexsort((rand[r], -s[r], nan[r], ~valid[r]))] = ar
    return rank


def apply_user_fallback(sc: np.ndarray, valid: np.ndarray, pop: np.ndarray) -> np.ndarray:
    """Users with nothing scorable (every candidate NaN) fall back to popularity scores."""
    sc = sc.astype(float).copy()
    unknown_user = (np.isnan(sc) | ~valid).all(1)
    sc[unknown_user] = pop[unknown_user]
    return sc


def all_orders(cand, hit, pop_counts, ncf_scores: dict | None = None, seed: int = 0) -> dict:
    """Rank positions for: tt (the list's own order), pop, random, oracle, and for each NCF score matrix
    'ncf_<name>' plus 'blend_<name>' (untuned 50/50 mix of tt-rank and NCF-rank). A user NCF cannot score at
    all (every candidate NaN) falls back to popularity order."""
    valid = cand >= 0
    U, W = cand.shape
    rand = np.random.default_rng(seed).random((U, W))
    pop = np.where(valid, pop_counts[np.maximum(cand, 0)], np.nan).astype(float)
    n_valid = np.maximum(valid.sum(1, keepdims=True) - 1, 1)
    out = {"tt": rank_positions(valid, np.tile(-np.arange(W, dtype=float), (U, 1)), rand),
           "pop": rank_positions(valid, pop, rand),
           "random": rank_positions(valid, np.zeros((U, W)), rand),
           "oracle": rank_positions(valid, hit.astype(float), rand)}
    for name, sc in (ncf_scores or {}).items():
        out[f"ncf_{name}"] = rank_positions(valid, apply_user_fallback(sc, valid, pop), rand)
        blend = -(0.5 * out["tt"] / n_valid + 0.5 * out[f"ncf_{name}"] / n_valid)
        out[f"blend_{name}"] = rank_positions(valid, blend, rand)
    return out


def metrics_from_ranks(rank: np.ndarray, hit: np.ndarray, n_truth: np.ndarray, users) -> pd.DataFrame:
    """NDCG@10 and Recall@20 in two views.
    e2e: every user; truth = ALL their validation likes (capped by what retrieval put in the list).
    rr : users with >= 1 liked item inside the list; truth restricted to the list (pure ordering quality)."""
    H = np.zeros(hit.shape, dtype=float)
    np.put_along_axis(H, rank.astype(np.int64), hit.astype(float), axis=1)
    k10 = min(10, H.shape[1])
    dcg = (H[:, :k10] * DISC[:k10]).sum(1)
    n_in = H.sum(1)
    top20 = H[:, :20].sum(1)
    nt = np.asarray(n_truth)
    rr = n_in > 0
    return pd.DataFrame({"e2e_ndcg10": dcg / IDCG[np.minimum(nt, 10)], "e2e_recall20": top20 / nt,
                         "rr_ndcg10": np.divide(dcg, IDCG[np.minimum(n_in, 10).astype(int)], out=np.full(len(dcg), np.nan), where=rr),
                         "rr_recall20": np.where(rr, top20 / np.maximum(n_in, 1), np.nan)}, index=users)


def nan_share(cand, hit, n_main: int, item_known_cat: np.ndarray, user_known: np.ndarray) -> dict:
    """Among validation likes INSIDE the lists, the share NCF cannot score, split by lane.
    item-NaN = the item has no train like; user-NaN = the user has no train like (every candidate is NaN)."""
    valid = cand >= 0
    item_nan = valid & ~item_known_cat[np.maximum(cand, 0)]
    user_nan = ~user_known[:, None] & valid
    lane = np.arange(cand.shape[1]) >= n_main
    out = {}
    for name, cols in [("main_lane", ~lane), ("new_release_lane", lane)]:
        likes = hit[:, cols]
        n = int(likes.sum())
        out[name] = {"likes_inside": n, "on_item_nan": int((likes & item_nan[:, cols]).sum()),
                     "on_user_nan": int((likes & user_nan[:, cols]).sum()),
                     "on_either_nan": int((likes & (item_nan | user_nan)[:, cols]).sum()),
                     "slots": int(valid[:, cols].sum()), "slots_item_nan": int(item_nan[:, cols].sum())}
    return out


def bucket_recall(ranks: dict, hit: np.ndarray, cand: np.ndarray, item_counts_cat: np.ndarray, k: int = 20, rows=None):
    """Share of in-list liked items found in each method's top-k, by how popular the liked item is in train.
    Buckets: no train likes (NCF cannot score), then tertiles of train-like counts among in-list liked items."""
    rows = np.arange(len(cand)) if rows is None else rows
    cnt = np.where(cand >= 0, item_counts_cat[np.maximum(cand, 0)], -1)
    h = hit[rows]
    c = cnt[rows]
    pos_counts = c[h & (c > 0)]
    e1, e2 = np.quantile(pos_counts, [1 / 3, 2 / 3])
    bucket = np.select([c == 0, c <= e1, c <= e2], [0, 1, 2], default=3)
    names = ["no train likes", f"tail (<= {e1:.0f} likes)", f"mid (<= {e2:.0f})", "head"]
    out = {}
    for m, rk in ranks.items():
        top = rk[rows] < k
        out[m] = [float((h & top & (bucket == b)).sum() / max((h & (bucket == b)).sum(), 1)) for b in range(4)]
    out["_n_positives"] = [int((h & (bucket == b)).sum()) for b in range(4)]
    return names, out
