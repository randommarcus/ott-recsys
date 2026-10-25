"""Tuning and validation-only evaluation of the baselines.

THIS MODULE MUST NEVER TOUCH THE TEST PORTION (split.test / unlock_test).
tests/test_module2.py enforces that both at runtime and by scanning this file."""
from itertools import product

import numpy as np
import pandas as pd

from .baselines import GlobalPop, HourPop, ItemKNN, RandomRec, knn_neighbors
from .eval import cold_report, evaluate, first_hour, paired_diff, summarize

PRIMARY = "all_ndcg@10"  # metric used to pick hyperparameters (on validation)

# Widened once after the small-dataset run hit the edge; item_knn widened a 2nd
# time after the 25M run hit k=500 and shrink=0 (0 is the natural lower limit).
DEFAULT_GRIDS = {
    "hour_pop": {"window": [0, 1, 3, 6], "lam": [50.0, 500.0, 5_000.0, 50_000.0, 500_000.0, 5_000_000.0]},
    "item_knn": {"k": [100, 200, 500, 1000, 1500], "shrink": [0.0, 1.0, 3.0, 10.0, 50.0]},  # 2nd widening
}


def _score_of(res) -> float:
    return float(res.per_user[PRIMARY].mean())


def _edges(best: dict, grid: dict) -> list[str]:
    """Names of tuned parameters whose best value sits at the edge of its grid."""
    return [k for k, v in best.items() if len(grid[k]) > 1 and v in (min(grid[k]), max(grid[k]))]


def _tune_hour_pop(grid, train, val_t, catalog, hours):
    best = (-1.0, None)
    for vals in product(*grid.values()):
        params = dict(zip(grid, vals))
        m = HourPop(**params)
        m.fit(train, catalog)
        sc = _score_of(evaluate(m, train, val_t, catalog, hours))
        if sc > best[0]:
            best = (sc, params)
    return best[1]


def _tune_knn(grid, train, val_t, catalog, hours):
    # one expensive pass over all shrink values; every k is a prefix of the sorted neighbours
    state = knn_neighbors(train, catalog, grid["shrink"], max(grid["k"]))
    best = (-1.0, None)
    for sh, k in product(grid["shrink"], grid["k"]):
        m = ItemKNN().use(state, sh, k)
        sc = _score_of(evaluate(m, train, val_t, catalog, hours))
        if sc > best[0]:
            best = (sc, {"k": k, "shrink": sh})
    return best[1], state


def run_validation(split, movies: pd.DataFrame, grids=None, n_shuffle: int = 5, n_boot: int = 1000,
                   tune_users: int | None = None, seed: int = 0, log=print) -> dict:
    """Fit on train, tune on (a subsample of) val, report on all of val.
    Never reads split.test."""
    grids = grids or DEFAULT_GRIDS
    train, val = split.train, split.val
    catalog = np.sort(movies["movieId"].to_numpy())
    hours = first_hour(val)

    liked_users = np.sort(val.loc[val["label"] == 1, "user_id"].unique())
    if tune_users and tune_users < len(liked_users):
        pick = np.random.default_rng(seed).choice(liked_users, tune_users, replace=False)
        val_t = val[val["user_id"].isin(pick)]
    else:
        val_t = val
    log(f"tuning on {val_t.loc[val_t['label'] == 1, 'user_id'].nunique()} of {len(liked_users)} validation users")

    best_hour = _tune_hour_pop(grids["hour_pop"], train, val_t, catalog, hours)
    log(f"hour_pop tuned: {best_hour}")
    best_knn, state = _tune_knn(grids["item_knn"], train, val_t, catalog, hours)
    log(f"item_knn tuned: {best_knn}")
    best = {"hour_pop": best_hour, "item_knn": best_knn}
    edges = {"hour_pop": _edges(best_hour, grids["hour_pop"]), "item_knn": _edges(best_knn, grids["item_knn"])}

    runs = {"random": RandomRec(), "global_pop": GlobalPop(),
            "hour_pop": HourPop(**best_hour), "item_knn": ItemKNN().use(state, best_knn["shrink"], best_knn["k"])}
    results = {}
    for name, m in runs.items():
        if name != "item_knn":
            m.fit(train, catalog)
        results[name] = evaluate(m, train, val, catalog, hours)
        log(f"evaluated {name} on all validation users")

    # control: same hour-pop settings, hour labels shuffled; average over seeds
    ctrl = []
    for sd in range(n_shuffle):
        m = HourPop(**best_hour, shuffle_hours=True, seed=sd)
        m.fit(train, catalog)
        ctrl.append(evaluate(m, train, val, catalog, hours))
    avg = sum(c.per_user for c in ctrl) / len(ctrl)
    results["hour_pop_shuffled"] = ctrl[0].__class__(avg, float(np.mean([c.coverage for c in ctrl])),
                                                    ctrl[0].top, ctrl[0].users, catalog)
    log("evaluated shuffled-hour control")

    tables = {n: summarize(r.per_user, r.coverage, n_boot) for n, r in results.items()}
    pu = {n: r.per_user for n, r in results.items()}
    paired = {}
    for label, (x, y) in {"hour_pop - global_pop": ("hour_pop", "global_pop"),
                          "hour_pop - hour_pop_shuffled": ("hour_pop", "hour_pop_shuffled"),
                          "item_knn - global_pop": ("item_knn", "global_pop")}.items():
        for col in ["all_ndcg@10", "all_recall@10", "all_recall@50"]:
            paired[f"{label} | {col}"] = paired_diff(pu[x][col], pu[y][col])
    return {"paired": paired, "best": best, "edges": edges, "results": results, "tables": tables,
            "cold": cold_report(train, val), "catalog": catalog, "hours": hours, "models": runs,
            "tuned_on_users": int(val_t.loc[val_t["label"] == 1, "user_id"].nunique())}


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
