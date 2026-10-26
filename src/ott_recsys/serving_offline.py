"""OFFLINE side of Module 9: turn the frozen Module 8 artifacts into (1) the bundle shipped inside the Lambda zip and (2) one record per user for DynamoDB.

Everything here uses TRAIN + VALIDATION history only (the 'real' Module 8 artifacts); the test portion is never read (tests/test_module2.py scans this file).
pandas / scipy are fine here: this code never runs in Lambda."""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .context import SESSION_GAP_S, session_state
from .rankfeat import _unit
from .serving.trees import TreeEnsemble

MAX_SESSION_ITEMS = 50  # same cap as rankfeat.build_features


def item_tables(history: pd.DataFrame, catalog: np.ndarray):
    """Sorted like timestamps per item (popularity 'as of' any time), for the bundle."""
    likes = history[history["label"] == 1]
    idx = pd.Index(catalog).get_indexer(likes["item_id"].to_numpy())
    ts = likes["timestamp"].to_numpy().astype(np.int64)
    order = np.lexsort((ts, idx))
    pop_ts = ts[order].astype(np.uint32)
    pop_start = np.searchsorted(idx[order], np.arange(len(catalog) + 1)).astype(np.int64)
    counts = np.bincount(idx, minlength=len(catalog)).astype(float)
    return pop_ts, pop_start, np.argsort(-counts, kind="stable").astype(np.int32)


def write_bundle(out: Path, *, catalog, V, emb_c, genre_m, genre_names, rel_ts, known, history, ncf_arrays, ncf_item_ids, titles, rankers: dict,
                 emb_dtype=np.float32) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    pop_ts, pop_start, pop_order = item_tables(history, catalog)
    np.savez_compressed(out / "items.npz", catalog=catalog.astype(np.int64), V=V.astype(np.float32), emb_c=emb_c.astype(emb_dtype),
                        genre_m=genre_m.astype(np.uint8), genre_names=np.array(genre_names), rel_ts=rel_ts.astype(np.float64), known=known,
                        pop_ts=pop_ts, pop_start=pop_start, pop_order=pop_order)
    np.savez_compressed(out / "ncf.npz", item_ids=np.asarray(ncf_item_ids), **ncf_arrays)
    (out / "titles.json").write_text(json.dumps(list(titles), ensure_ascii=False))
    for name, path in rankers.items():
        TreeEnsemble.from_xgboost_json(path).save_npz(out / f"ranker_{name}.npz")
    return {p.name: p.stat().st_size for p in sorted(out.glob("*"))}


def build_user_arrays(history: pd.DataFrame, users, catalog, emb_c, genre_m, cand, tt_ids, tt_U, fold) -> dict:
    """Column-wise records for `users` (aligned with `cand` rows). Missing pieces are NaN rows plus a boolean mask."""
    users = np.asarray(users)
    U, D, G = len(users), emb_c.shape[1], genre_m.shape[1]
    cat = pd.Index(catalog)
    h = history.sort_values(["user_id", "timestamp", "item_id"], kind="mergesort")
    uid, ts = h["user_id"].to_numpy(), h["timestamp"].to_numpy().astype(np.int64)
    items, lab = cat.get_indexer(h["item_id"].to_numpy()), h["label"].to_numpy()
    lo, hi = np.searchsorted(uid, users, "left"), np.searchsorted(uid, users, "right")
    tt_pos = pd.Index(tt_ids).get_indexer(users)
    f_pos = fold._u.get_indexer(users)
    a = {"user_id": users.astype(np.int64), "cand": cand.astype(np.int32), "last_ts": np.zeros(U, np.int64), "sess_len": np.zeros(U, np.int32),
         "sess_likes": np.zeros(U, np.int32), "ncf_known": (f_pos >= 0).astype(np.int8),
         "ttvec": np.full((U, tt_U.shape[1]), np.nan, np.float32), "has_ttvec": tt_pos >= 0,
         "ncf_ug": np.full((U, fold.a["ug"].shape[1]), np.nan, np.float32), "ncf_um": np.full((U, fold.a["um"].shape[1]), np.nan, np.float32),
         "prof": np.full((U, D), np.nan, np.float32), "has_prof": np.zeros(U, bool), "share": np.full((U, G), np.nan, np.float32),
         "seen_flat": [], "seen_off": np.zeros(U + 1, np.int64), "sess_flat": [], "sess_off": np.zeros(U + 1, np.int64)}
    a["ttvec"][tt_pos >= 0] = tt_U[tt_pos[tt_pos >= 0]]
    a["ncf_ug"][f_pos >= 0], a["ncf_um"][f_pos >= 0] = fold.a["ug"][f_pos[f_pos >= 0]], fold.a["um"][f_pos[f_pos >= 0]]
    for r in range(U):
        t, it, lb = ts[lo[r]:hi[r]], items[lo[r]:hi[r]], lab[lo[r]:hi[r]]
        assert len(t) > 0, "every user in the lists has history"
        a["last_ts"][r] = t[-1]
        st = session_state(t, lb, int(t[-1]) + 1)  # a clock one second later: the user is mid-session, which gives the last spree
        assert st["in_session"] == 1.0 and st["n"] == len(t)
        a["sess_len"][r], a["sess_likes"][r] = st["length"], int(lb[st["start"]:st["n"]].sum())
        a["sess_flat"].append(it[max(st["start"], st["n"] - MAX_SESSION_ITEMS):st["n"]].astype(np.int32))
        a["sess_off"][r + 1] = a["sess_off"][r] + len(a["sess_flat"][-1])
        a["seen_flat"].append(np.unique(it).astype(np.int32))
        a["seen_off"][r + 1] = a["seen_off"][r] + len(a["seen_flat"][-1])
        liked = it[lb == 1]
        if len(liked):
            a["prof"][r] = _unit(emb_c[liked].mean(0))
            a["share"][r] = genre_m[liked].sum(0) / genre_m[liked].sum()
            a["has_prof"][r] = True
    a["seen_flat"], a["sess_flat"] = np.concatenate(a["seen_flat"]), np.concatenate(a["sess_flat"])
    return a


def record_at(a: dict, i: int) -> dict:
    """One user's record dict (what the DynamoDB item unpacks to)."""
    ok = lambda mask, arr: arr[i] if mask else None
    return {"user_id": int(a["user_id"][i]), "cand": a["cand"][i], "last_ts": int(a["last_ts"][i]), "sess_len": int(a["sess_len"][i]),
            "sess_likes": int(a["sess_likes"][i]), "ncf_known": int(a["ncf_known"][i]), "ttvec": ok(a["has_ttvec"][i], a["ttvec"]),
            "ncf_ug": ok(a["ncf_known"][i], a["ncf_ug"]), "ncf_um": ok(a["ncf_known"][i], a["ncf_um"]), "prof": ok(a["has_prof"][i], a["prof"]),
            "share": ok(a["has_prof"][i], a["share"]), "seen": a["seen_flat"][a["seen_off"][i]:a["seen_off"][i + 1]],
            "sess_items": a["sess_flat"][a["sess_off"][i]:a["sess_off"][i + 1]]}
