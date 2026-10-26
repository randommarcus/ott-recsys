"""Ranker features for (user, candidate) rows.

build_features() receives TRAIN interactions and the request times only. Validation labels and validation items
are never inputs, and every user-side quantity (liked-item profile, genre shares, session, as-of popularity)
uses train interactions with timestamp STRICTLY BEFORE that user's request time. The frozen retriever / NCF
scores arrive as ready-made matrices (they were trained on train only).
Never touches the test portion (tests/test_module2.py scans it)."""
import numpy as np
import pandas as pd

from .context import compact, pop_asof, pop_keys, session_state

BASE = ["tt_score", "tt_rank", "ncf_logit", "ncf_user_known", "ncf_item_known", "pop_asof", "content_sim", "genre_aff",
        "item_age_days", "lane"]
CTX_TIME = ["hour", "dow"]
CTX_SESSION = ["mins_since_last", "in_session", "session_len", "session_like_rate", "sess_genre_match", "sess_content_sim"]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.full_like(v, np.nan)


def build_features(train: pd.DataFrame, users, t_req, cand, n_main: int, catalog, genre_m, genre_names, emb_c, rel_ts,
                   tt_score, ncf_logit, ncf_user_known, ncf_item_known_cat, max_session_items: int = 50,
                   chunk: int = 500) -> dict:
    users, t_req, cand = np.asarray(users), np.asarray(t_req, dtype=np.int64), np.asarray(cand, dtype=np.int64)
    U, W = cand.shape
    n_items, G = len(catalog), genre_m.shape[1]
    # ---- release-by-request-time filter (every method sees the same filtered lists) and compaction
    valid = cand >= 0
    rel = np.where(valid, rel_ts[np.maximum(cand, 0)], np.nan)
    keep = valid & ~(rel > t_req[:, None])  # unknown release date (NaN) is kept
    lane_full = np.broadcast_to(np.arange(W) >= n_main, (U, W))
    cand_f = compact(cand, keep, -1)
    lane_f = compact(np.ascontiguousarray(lane_full), keep, False)
    tt_f, ncf_f, rel_f = (compact(a, keep, np.nan) for a in (tt_score, ncf_logit, rel))
    valid_f = cand_f >= 0
    main_kept = valid_f & ~lane_f
    tt_rank = np.where(main_kept, np.cumsum(main_kept, axis=1) - 1, np.nan).astype(np.float32)

    # ---- per-user, STRICTLY-BEFORE quantities from train
    order = np.lexsort((train["item_id"].to_numpy(), train["timestamp"].to_numpy(), train["user_id"].to_numpy()))
    uid_all = train["user_id"].to_numpy()[order]
    ts_all = train["timestamp"].to_numpy()[order].astype(np.int64)
    item_all = pd.Index(catalog).get_indexer(train["item_id"].to_numpy()[order])
    lab_all = train["label"].to_numpy()[order]
    lo, hi = np.searchsorted(uid_all, users, "left"), np.searchsorted(uid_all, users, "right")
    D = emb_c.shape[1]
    prof, share, sess_vec, sess_share = (np.full((U, D), np.nan, np.float32), np.full((U, G), np.nan, np.float32),
                                         np.full((U, D), np.nan, np.float32), np.full((U, G), np.nan, np.float32))
    sc = {k: np.full(U, np.nan, np.float32) for k in ("mins_since_last", "in_session", "session_len", "session_like_rate")}
    for r in range(U):
        ts, items, lab = ts_all[lo[r]:hi[r]], item_all[lo[r]:hi[r]], lab_all[lo[r]:hi[r]]
        st = session_state(ts, lab, int(t_req[r]))
        liked = items[:st["n"]][lab[:st["n"]] == 1]
        if len(liked):
            prof[r] = _unit(emb_c[liked].mean(0))
            share[r] = genre_m[liked].sum(0) / genre_m[liked].sum()
        sc["mins_since_last"][r], sc["in_session"][r], sc["session_len"][r] = st["mins_since_last"], st["in_session"], st["length"]
        sc["session_like_rate"][r] = st["like_rate"]
        if st["in_session"]:
            s_items = items[max(st["start"], st["n"] - max_session_items):st["n"]]
            sess_vec[r] = _unit(emb_c[s_items].mean(0))
            tot = genre_m[s_items].sum()
            if tot > 0:
                sess_share[r] = genre_m[s_items].sum(0) / tot

    # ---- slot-level features
    keys, start = pop_keys(item_all[lab_all == 1], ts_all[lab_all == 1], n_items)
    safe = np.maximum(cand_f, 0)
    pop = pop_asof(keys, start, safe, np.broadcast_to(t_req[:, None], (U, W))).astype(np.float32)
    content_sim = np.full((U, W), np.nan, np.float32)
    sess_content = np.full((U, W), np.nan, np.float32)
    gaff = np.full((U, W), np.nan, np.float32)
    sess_gaff = np.full((U, W), np.nan, np.float32)
    for a in range(0, U, chunk):
        b = min(a + chunk, U)
        E, Gm = emb_c[safe[a:b]], genre_m[safe[a:b]]
        content_sim[a:b] = np.einsum("cwd,cd->cw", E, prof[a:b])
        sess_content[a:b] = np.einsum("cwd,cd->cw", E, sess_vec[a:b])
        gaff[a:b] = genre_affinity_batch(share[a:b], Gm)
        sess_gaff[a:b] = genre_affinity_batch(sess_share[a:b], Gm)
    age = ((t_req[:, None] - rel_f) / 86400.0).astype(np.float32)
    hour = pd.to_datetime(t_req, unit="s").hour.to_numpy().astype(np.float32)
    dow = pd.to_datetime(t_req, unit="s").day_of_week.to_numpy().astype(np.float32)

    # ---- flatten rows (user-major, slot order preserved)
    ru, rs = np.nonzero(valid_f)
    cols = BASE + [f"g_{n}" for n in genre_names] + CTX_TIME + CTX_SESSION
    X = np.empty((len(ru), len(cols)), dtype=np.float32)
    items_r = cand_f[ru, rs]
    base = [tt_f[ru, rs], tt_rank[ru, rs], ncf_f[ru, rs], np.asarray(ncf_user_known, np.float32)[ru],
            ncf_item_known_cat[items_r].astype(np.float32), pop[ru, rs], content_sim[ru, rs], gaff[ru, rs], age[ru, rs],
            lane_f[ru, rs].astype(np.float32)]
    for i, col in enumerate(base):
        X[:, i] = col
    X[:, len(BASE):len(BASE) + G] = genre_m[items_r]
    j = len(BASE) + G
    X[:, j], X[:, j + 1] = hour[ru], dow[ru]
    sess_cols = {"mins_since_last": sc["mins_since_last"][ru], "in_session": sc["in_session"][ru],
                 "session_len": sc["session_len"][ru], "session_like_rate": sc["session_like_rate"][ru],
                 "sess_genre_match": sess_gaff[ru, rs], "sess_content_sim": sess_content[ru, rs]}
    for i, name in enumerate(CTX_SESSION):
        X[:, j + 2 + i] = sess_cols[name]
    return {"X": X, "cols": cols, "row_user": ru.astype(np.int32), "row_slot": rs.astype(np.int16), "cand_f": cand_f.astype(np.int32),
            "lane_f": lane_f, "ncf_f": ncf_f.astype(np.float32), "removed_future": int((valid & ~keep).sum()),
            "session_len": sc["session_len"], "in_session": sc["in_session"], "hour": hour, "dow": dow}


def genre_affinity_batch(share: np.ndarray, item_multihot: np.ndarray) -> np.ndarray:
    """share [c, G], item_multihot [c, W, G] -> [c, W]; NaN where the share is undefined."""
    out = np.einsum("cwg,cg->cw", item_multihot, np.nan_to_num(share)) / np.maximum(item_multihot.sum(-1), 1.0)
    out[np.isnan(share).any(1)] = np.nan
    return out.astype(np.float32)
