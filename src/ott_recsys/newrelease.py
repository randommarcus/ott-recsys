"""New-release evaluation on the GLOBAL time split (train = everything up to the
cutoff, val = the window after it). Uses only global train and global val; the
test portion is never touched."""
import numpy as np
import pandas as pd

from .eval import evaluate, summarize

DEFINITIONS = ("release_date", "release_cold", "release_warm", "first_rated")
# primary = release_date; release_cold / release_warm split it by "no ratings / some ratings in train"; secondary = first_rated
MODES = ("full", "pool")  # compete against the whole catalog / only against the new items


def release_sets(train_g: pd.DataFrame, val_g: pd.DataFrame, rel: pd.Series, months: int = 6) -> dict:
    """PRIMARY: TMDB release date on/after (cutoff - months) and not after the end of the
    validation window. SECONDARY: movies first rated after the cutoff (in val, never in train)."""
    cutoff = pd.to_datetime(int(train_g["timestamp"].max()), unit="s")
    val_end = pd.to_datetime(int(val_g["timestamp"].max()), unit="s")
    lo = cutoff - pd.DateOffset(months=months)
    primary = rel.index[(rel >= lo) & (rel <= val_end)].to_numpy()
    secondary = np.setdiff1d(val_g["item_id"].unique(), train_g["item_id"].unique())
    in_train = set(train_g["item_id"])
    cold = np.array([i for i in primary if i not in in_train], dtype=primary.dtype)
    warm = np.array([i for i in primary if i in in_train], dtype=primary.dtype)
    return {"release_date": primary, "release_cold": cold, "release_warm": warm, "first_rated": secondary, "cutoff": cutoff, "val_end": val_end, "window_start": lo,
            "primary_with_no_train_ratings": int(sum(i not in in_train for i in primary)),
            "overlap": int(len(np.intersect1d(primary, secondary)))}


def evaluate_new_release(models: dict, train_g, val_g, catalog, sets: dict, ks=(10, 50, 200)) -> dict:
    """models: name -> FITTED model (fit on train_g). Eval users must have >= 1 liked item in
    train_g (otherwise there is no profile to personalise from)."""
    profile_users = set(train_g.loc[train_g["label"] == 1, "user_id"])
    out = {}
    for d in [d for d in DEFINITIONS if d in sets]:
        items = sets[d]
        ev = val_g[val_g["item_id"].isin(set(items)) & val_g["user_id"].isin(profile_users)]
        n_users = ev.loc[ev["label"] == 1, "user_id"].nunique()
        for mode in MODES:
            cand = items if mode == "pool" else None
            tables = {}
            if n_users:
                for name, m in models.items():
                    res = evaluate(m, train_g, ev, catalog, ks=ks, candidate_items=cand)
                    tables[name] = summarize(res.per_user, res.coverage)
            out[(d, mode)] = {"tables": tables, "n_users": int(n_users), "n_items": int(len(items)),
                              "n_likes": int(((ev["label"] == 1)).sum())}
    return out


def format_table(tables: dict, ks=(10, 50, 200)) -> pd.DataFrame:
    cols = [f"all_recall@{k}" for k in ks] + ["all_ndcg@10"]
    rows = {n: {c.split("_", 1)[1]: f"{t.loc[c, 'mean']:.4f} [{t.loc[c, 'lo']:.3f}-{t.loc[c, 'hi']:.3f}]" for c in cols}
            for n, t in tables.items()}
    return pd.DataFrame(rows).T


def new_release_tops(models: dict, train_g, val_g, catalog, sets: dict, definitions=("release_cold", "release_date", "first_rated"),
                     ks=(10, 50, 200)) -> dict:
    """Ranked top-200 lists per model for each (definition, mode), plus the evaluation frame, so
    candidate mixes can be built from them. Same user filter as evaluate_new_release."""
    profile_users = set(train_g.loc[train_g["label"] == 1, "user_id"])
    out = {}
    for d in definitions:
        items = sets[d]
        ev = val_g[val_g["item_id"].isin(set(items)) & val_g["user_id"].isin(profile_users)]
        for mode in MODES:
            cand = items if mode == "pool" else None
            res = {n: evaluate(m, train_g, ev, catalog, ks=ks, candidate_items=cand) for n, m in models.items()}
            out[(d, mode)] = {"ev": ev, "res": res, "n_items": int(len(items))}
    return out
