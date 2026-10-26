"""Preparation for the one-shot test evaluation (Module 8). Everything here uses TRAIN and VALIDATION data only; the test
portion is never read (tests/test_module2.py scans it). Outputs are frozen artifacts whose hashes are recorded in a manifest.

history   = what a deployed system would know at the start of the evaluation window: train + validation interactions
            (rehearsal: train only, so the validation period can play the role of a fake test period)."""
import numpy as np
import pandas as pd

from .eval import rank_top
from .ncf import score_candidates
from .rerank import build_candidate_lists

N_MAIN, N_LANE = 200, 30


def history_frame(train: pd.DataFrame, val: pd.DataFrame, rehearsal: bool) -> pd.DataFrame:
    cols = ["user_id", "item_id", "label", "timestamp"]
    return (train[cols] if rehearsal else pd.concat([train[cols], val[cols]], ignore_index=True)).reset_index(drop=True)


def seen_map(history_df: pd.DataFrame, catalog) -> dict:
    cat = pd.Index(catalog)
    return {u: cat.get_indexer(g["item_id"].to_numpy()) for u, g in history_df.groupby("user_id")}


def candidate_lists(users, seen, V, tt_ids, tt_U, content_ids, profiles, emb_c, lane_pool_idx, pop_counts, chunk: int = 500):
    """230-slot lists for every user: main lane = frozen ID-only two-tower with user vectors from the history, already-seen items
    removed; new-release lane = centered-content search restricted to the lane pool, appended (never replacing a main slot).
    Users with no history vector fall back to popularity (main) and get no lane."""
    tt_idx, ct_idx = {u: i for i, u in enumerate(tt_ids)}, {u: i for i, u in enumerate(content_ids)}
    pop = pop_counts / max(pop_counts.max(), 1.0)
    outside = np.setdiff1d(np.arange(V.shape[0]), lane_pool_idx)
    tiebreak = 1e-6 * pop
    mains, lanes = [], []
    for a in range(0, len(users), chunk):
        us = users[a:a + chunk]
        rows = np.array([tt_idx.get(u, -1) for u in us])
        s = np.where((rows >= 0)[:, None], tt_U[np.maximum(rows, 0)] @ V.T, pop[None, :])
        sn = [seen.get(u, np.empty(0, dtype=np.int64)) for u in us]
        mains.append(rank_top(s, sn, N_MAIN))
        crow = np.array([ct_idx.get(u, -1) for u in us])
        c = profiles[np.maximum(crow, 0)] @ emb_c.T + tiebreak
        c[crow < 0] = -np.inf  # no profile -> no lane
        lanes.append(rank_top(c, [np.union1d(x, outside) for x in sn], N_LANE))
    main_top, lane_top = np.vstack(mains), np.vstack(lanes)
    return build_candidate_lists(main_top, lane_top, N_MAIN, N_LANE)[0]


def two_tower_scores(users, cand, V, tt_ids, tt_U, known_items):
    """ID-only two-tower score for main-lane candidates the frozen model knows; NaN for the lane and for unknown items."""
    idx = {u: i for i, u in enumerate(tt_ids)}
    out = np.full(cand.shape, np.nan)
    main = np.arange(cand.shape[1]) < N_MAIN
    for a in range(0, len(users), 2000):
        b = min(a + 2000, len(users))
        rows = np.array([idx.get(u, -1) for u in users[a:b]])
        c = cand[a:b]
        s = np.einsum("cwd,cd->cw", V[np.maximum(c, 0)], tt_U[np.maximum(rows, 0)])
        ok = (c >= 0) & main[None, :] & known_items[np.maximum(c, 0)] & (rows >= 0)[:, None]
        out[a:b] = np.where(ok, s, np.nan)
    return out


def ncf_matrix(scorer, users, cand, catalog):
    return score_candidates(scorer, users, cand, catalog).astype(np.float32), np.isin(users, scorer.user_ids)
