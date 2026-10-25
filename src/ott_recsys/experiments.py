"""Tuning and validation-only evaluation of the baselines.

THIS MODULE MUST NEVER TOUCH THE TEST PORTION (split.test / unlock_test).
tests/test_module2.py enforces that both at runtime and by scanning this file."""
from itertools import product

import numpy as np
import pandas as pd

from .baselines import GlobalPop, HourPop, ItemKNN, RandomRec
from .eval import cold_report, evaluate, first_hour, summarize

PRIMARY = "all_ndcg@10"  # metric used to pick hyperparameters (on validation)

DEFAULT_GRIDS = {
    "hour_pop": {"window": [0, 1, 3], "lam": [1.0, 50.0, 500.0, 5000.0]},
    "item_knn": {"k": [20, 50, 100, 200], "shrink": [0.0, 10.0, 50.0]},
}


def _score_of(res) -> float:
    return float(res.per_user[PRIMARY].mean())


def _tune(make, grid, fit_df, val_df, catalog, hours):
    best = (-1.0, None)
    for vals in product(*grid.values()):
        params = dict(zip(grid, vals))
        m = make(**params)
        m.fit(fit_df, catalog)
        s = _score_of(evaluate(m, fit_df, val_df, catalog, hours))
        if s > best[0]:
            best = (s, params)
    return best[1], best[0]


def run_validation(split, movies: pd.DataFrame, grids=None, n_shuffle: int = 5, n_boot: int = 1000) -> dict:
    """Fit on train, tune on val, report on val. Never reads split.test."""
    grids = grids or DEFAULT_GRIDS
    train, val = split.train, split.val
    catalog = np.sort(movies["movieId"].to_numpy())
    hours = first_hour(val)

    best = {}
    for name, make in [("hour_pop", HourPop), ("item_knn", ItemKNN)]:
        best[name] = _tune(make, grids[name], train, val, catalog, hours)

    runs = {"random": RandomRec(), "global_pop": GlobalPop(),
            "hour_pop": HourPop(**best["hour_pop"][0]), "item_knn": ItemKNN(**best["item_knn"][0])}
    results = {}
    for name, m in runs.items():
        m.fit(train, catalog)
        results[name] = evaluate(m, train, val, catalog, hours)

    # control: same hour-pop settings, hour labels shuffled; average over seeds
    ctrl = []
    for seed in range(n_shuffle):
        m = HourPop(**best["hour_pop"][0], shuffle_hours=True, seed=seed)
        m.fit(train, catalog)
        ctrl.append(evaluate(m, train, val, catalog, hours))
    avg = sum(c.per_user for c in ctrl) / len(ctrl)
    results["hour_pop_shuffled"] = ctrl[0].__class__(avg, float(np.mean([c.coverage for c in ctrl])),
                                                    ctrl[0].top, ctrl[0].users, catalog)

    tables = {n: summarize(r.per_user, r.coverage, n_boot) for n, r in results.items()}
    return {"best": {k: v[0] for k, v in best.items()}, "results": results, "tables": tables,
            "cold": cold_report(train, val), "catalog": catalog, "hours": hours, "models": runs}


def results_table(tables: dict, view: str = "all") -> pd.DataFrame:
    metrics = [f"{view}_recall@10", f"{view}_recall@50", f"{view}_recall@200", f"{view}_ndcg@10"]
    rows = {}
    for name, t in tables.items():
        row = {m.split("_", 1)[1]: f"{t.loc[m, 'mean']:.4f} [{t.loc[m, 'lo']:.3f}-{t.loc[m, 'hi']:.3f}]"
               for m in metrics}
        row["users"] = int(t.loc[metrics[0], "n_users"])
        if view == "all":
            row["coverage@10"] = f"{t.loc['coverage@10', 'mean']:.4f}"
        rows[name] = row
    return pd.DataFrame(rows).T


def sample_users(out: dict, split, movies: pd.DataFrame, n: int = 5) -> str:
    """Plain-text view of n users spread from light to heavy, with titles.
    Shows VALIDATION likes (test is locked)."""
    titles = movies.set_index("movieId")["title"].to_dict()
    train, val = split.train, split.val
    users = out["results"]["global_pop"].users
    n_train = train.groupby("user_id").size().reindex(users)
    order = users[np.argsort(n_train.to_numpy(), kind="stable")]
    picks = [order[int(q * (len(order) - 1))] for q in np.linspace(0.1, 0.9, n)]
    cat = out["catalog"]
    seen_items = set(train["item_id"])
    lines = ["(held-out likes shown are from VALIDATION; the test portion is locked)\n"]
    for u in picks:
        tr = train[train["user_id"] == u].sort_values(["timestamp", "item_id"])
        liked_tr = tr[tr["label"] == 1]
        vl = val[(val["user_id"] == u) & (val["label"] == 1)]
        v_ids = set(vl["item_id"])
        lines.append(f"=== user {u}: {len(tr)} train ratings ({len(liked_tr)} likes), "
                     f"{len(vl)} validation likes ===")
        lines.append("  recent train likes: " + "; ".join(titles[i] for i in liked_tr["item_id"].tail(8)))
        lines.append("  validation likes:   " + "; ".join(
            titles[i] + ("  [cold]" if i not in seen_items else "") for i in list(vl["item_id"])[:10]))
        row = int(np.where(out["results"]["global_pop"].users == u)[0][0])
        for name in ["global_pop", "hour_pop", "item_knn"]:
            top = out["results"][name].top[row, :10]
            hits = sum(cat[i] in v_ids for i in top)
            lines.append(f"  top-10 {name} ({hits} hit{'s' if hits != 1 else ''}):")
            for rank, i in enumerate(top, 1):
                mark = "✓" if cat[i] in v_ids else " "
                lines.append(f"     {rank:2d}. {mark} {titles[cat[i]]}")
        lines.append("")
    return "\n".join(lines)
