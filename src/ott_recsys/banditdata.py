"""Inputs for the bandit simulation: arms, hidden-rate data, content prior and the request stream.
Uses global-cutoff train and validation only; never the test portion (tests/test_module2.py scans it)."""
import numpy as np
import pandas as pd

from . import content
from .bandit import fit_prior, session_starts
from .newrelease import release_sets


def load_bandit_inputs(cfg, root, catalog, movies, links, emb_c) -> dict:
    cat_idx = pd.Index(catalog)
    train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
    tmdb = content.load_tmdb_cache(root / "data" / "raw" / "tmdb", links["tmdbId"])
    rel = content.release_dates(movies, links, tmdb)
    sets = release_sets(train_g, val_g, rel, months=6)
    arm_ids = np.asarray(sets["release_cold"])
    rel_ts = np.array([rel.get(i, pd.NaT).timestamp() if pd.notna(rel.get(i, pd.NaT)) else np.nan for i in arm_ids])
    vr = val_g[val_g["item_id"].isin(set(arm_ids))]
    n_arm = vr.groupby("item_id").size().reindex(arm_ids).fillna(0).to_numpy()
    likes_arm = vr.groupby("item_id")["label"].sum().reindex(arm_ids).fillna(0).to_numpy()
    g = train_g.groupby("item_id").agg(n=("label", "size"), likes=("label", "sum"))
    g = g[g["n"] >= 20]  # PRE-cutoff movies only: the content prior never sees arm data
    model, w_prior, corr = fit_prior(emb_c, cat_idx.get_indexer(g.index), g["n"].to_numpy(), g["likes"].to_numpy())
    mu_arm = np.clip(model.predict(emb_c[cat_idx.get_indexer(arm_ids)]), 0.05, 0.95)
    return {"arm_ids": arm_ids, "rel_ts": rel_ts, "n_arm": n_arm, "likes_arm": likes_arm, "global_rate": float(vr["label"].mean()),
            "mu_arm": mu_arm, "w_prior": w_prior, "prior_corr": corr, "n_prior_movies": len(g), "req_t": session_starts(val_g),
            "cutoff": sets["cutoff"], "n_users": int(val_g["user_id"].nunique())}
