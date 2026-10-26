"""Ranker features for ONE request, in plain numpy. This mirrors ott_recsys.rankfeat.build_features line by line (same formulas, same NaN rules);
tests/test_serving_features.py compares the two on real users.

The request clock is SIMULATED: t_req = the user's last recorded activity + a gap chosen by the caller. Because t_req is after every recorded
activity of the user, all of the user's history counts, and the session state follows from the gap alone."""
import numpy as np

SESSION_GAP_S = 1800  # 30 minutes, as in Module 1
BASE = ["tt_score", "tt_rank", "ncf_logit", "ncf_user_known", "ncf_item_known", "pop_asof", "content_sim", "genre_aff", "item_age_days", "lane"]
CTX_TIME = ["hour", "dow"]
CTX_SESSION = ["mins_since_last", "in_session", "session_len", "session_like_rate", "sess_genre_match", "sess_content_sim"]


def _unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 0 else np.full_like(v, np.nan)


def _genre_aff(share, item_multihot):
    """Mean of the user's genre share over the item's genres; NaN where the share is undefined."""
    if share is None or np.isnan(share).any():
        return np.full(len(item_multihot), np.nan, np.float32)
    return (np.einsum("wg,g->w", item_multihot, share) / np.maximum(item_multihot.sum(-1), 1.0)).astype(np.float32)


def hour_dow(t_req: int):
    """UTC hour and day of week (Monday = 0), identical to pandas' .hour / .day_of_week."""
    return float((t_req // 3600) % 24), float((t_req // 86400 + 3) % 7)


def build_request_features(b, rec: dict, cand: np.ndarray, t_req: int, n_main: int = 200, device: str | None = None) -> dict:
    """cand: int array [W] of catalog indices (-1 = empty), first n_main slots main lane, the rest new-release lane.
    Returns a dict of named float32 feature columns plus '_cand', '_lane', '_rel' for the kept (released-by-t_req) candidates."""
    cand = np.asarray(cand, dtype=np.int64)
    W, G = len(cand), b.genre_m.shape[1]
    valid = cand >= 0
    rel = np.where(valid, b.rel_ts[np.maximum(cand, 0)], np.nan)
    keep = valid & ~(rel > t_req)  # unknown release date (NaN) is kept
    lane_full = np.arange(W) >= n_main
    c, lane, rel_k = cand[keep], lane_full[keep], rel[keep]
    m = len(c)
    main_kept = ~lane
    tt_rank = np.where(main_kept, np.cumsum(main_kept) - 1, np.nan).astype(np.float32)

    # ---- model scores
    ttvec = rec.get("ttvec")
    if ttvec is not None and m:
        s = np.einsum("wd,d->w", b.V[c], ttvec)
        tt = np.where(main_kept & b.known[c], s, np.nan).astype(np.float32)
    else:
        tt = np.full(m, np.nan, np.float32)
    ncf = b.ncf.score(rec.get("ncf_ug"), rec.get("ncf_um"), c)

    # ---- user-side quantities that depend on the clock (session) or not (profile)
    prof, share = rec.get("prof"), rec.get("share")
    since = t_req - int(rec["last_ts"])
    assert since > 0, "the simulated clock must be after the user's last recorded activity"
    in_session = since <= SESSION_GAP_S
    sess_len, like_rate = 0, np.nan
    sess_vec = sess_share = None
    if in_session:
        sess_len = int(rec["sess_len"])
        like_rate = int(rec["sess_likes"]) / sess_len
        s_items = rec["sess_items"]
        sess_vec = _unit(b.emb[s_items].astype(np.float32).mean(0))
        tot = b.genre_m[s_items].sum()
        if tot > 0:
            sess_share = b.genre_m[s_items].sum(0) / tot

    E = b.emb[c].astype(np.float32) if m else np.zeros((0, b.emb.shape[1]), np.float32)
    nan_m = np.full(m, np.nan, np.float32)
    content_sim = np.einsum("wd,d->w", E, prof) if prof is not None and m else nan_m
    sess_content = np.einsum("wd,d->w", E, sess_vec) if sess_vec is not None and m else nan_m
    Gm = b.genre_m[c]
    gaff, sess_gaff = _genre_aff(share, Gm), _genre_aff(sess_share, Gm)

    # ---- popularity strictly before the request time (all users' likes in the history)
    pop = np.array([np.searchsorted(b.pop_ts[b.pop_start[i]:b.pop_start[i + 1]], t_req, side="left") for i in c], dtype=np.float32)
    hour, dow = hour_dow(int(t_req))
    f = {"tt_score": tt, "tt_rank": tt_rank, "ncf_logit": ncf, "ncf_user_known": np.full(m, float(rec.get("ncf_known", 0)), np.float32),
         "ncf_item_known": b.known[c].astype(np.float32), "pop_asof": pop, "content_sim": np.asarray(content_sim, np.float32),
         "genre_aff": gaff, "item_age_days": ((t_req - rel_k) / 86400.0).astype(np.float32), "lane": lane.astype(np.float32)}
    for i, name in enumerate(b.genre_names):
        f[f"g_{name}"] = Gm[:, i]
    f["hour"], f["dow"] = np.full(m, hour, np.float32), np.full(m, dow, np.float32)
    f["mins_since_last"] = np.full(m, since / 60.0, np.float32)
    f["in_session"] = np.full(m, 1.0 if in_session else 0.0, np.float32)
    f["session_len"] = np.full(m, float(sess_len), np.float32)
    f["session_like_rate"] = np.full(m, like_rate, np.float32)
    f["sess_genre_match"], f["sess_content_sim"] = sess_gaff, np.asarray(sess_content, np.float32)
    if device is not None:  # SIMULATED device, one-hot in the order of DEVICES (only the M3b demo model uses it)
        from .bundle import DEVICES
        for d in DEVICES:
            f[f"dev_{d}"] = np.full(m, 1.0 if d == device else 0.0, np.float32)
    f["_cand"], f["_lane"], f["_rel"] = c, lane, rel_k  # bookkeeping (underscore: not features)
    return f


def matrix(f: dict, names) -> np.ndarray:
    """Feature matrix [rows, len(names)] in the column order a model was trained with."""
    return np.column_stack([f[n] for n in names]).astype(np.float32) if len(f["_cand"]) else np.zeros((0, len(names)), np.float32)
