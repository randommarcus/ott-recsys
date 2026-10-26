"""XGBRanker (rank:ndcg) training with grouped-by-user folds. Groups = users; rows must be user-major.
Never touches the test portion (tests/test_module2.py scans it)."""
import numpy as np

PARAMS = dict(n_estimators=800, learning_rate=0.08, max_depth=6, subsample=0.8, colsample_bytree=0.8, min_child_weight=1.0,
              max_bin=128, early_stopping_rounds=30, lambdarank_pair_method="topk", lambdarank_num_pair_per_sample=20)


def user_folds(n_users: int, k: int = 5, seed: int = 0) -> np.ndarray:
    """Fold id per user: every user is scored by exactly one model that never saw them."""
    f = np.arange(n_users) % k
    return f[np.random.default_rng(seed).permutation(n_users)]


def fit_ranker(Xtr, ytr, qtr, Xva, yva, qva, seed: int = 0, n_jobs: int = 16, params: dict | None = None):
    import xgboost as xgb
    m = xgb.XGBRanker(objective="rank:ndcg", tree_method="hist", eval_metric="ndcg@10", n_jobs=n_jobs, random_state=seed,
                      **{**PARAMS, **(params or {})})
    m.fit(Xtr, ytr, qid=qtr, eval_set=[(Xva, yva)], eval_qid=[qva], verbose=False)
    return m


def oof_scores(X, y, row_user, n_users: int, names, k: int = 5, seed: int = 0, inner_frac: float = 0.1, log=print, params=None):
    """Out-of-fold ranker scores for every row, per-fold best iteration and normalised gain importance."""
    folds = user_folds(n_users, k, seed)
    rng = np.random.default_rng(seed + 1)
    scores = np.full(len(y), np.nan, dtype=np.float32)
    best_iter, imp = [], []
    fold_of_row = folds[row_user]
    for f in range(k):
        tr_users = np.flatnonzero(folds != f)
        inner = rng.choice(tr_users, max(1, int(inner_frac * len(tr_users))), replace=False)
        is_inner = np.zeros(n_users, bool)
        is_inner[inner] = True
        tr = (fold_of_row != f) & ~is_inner[row_user]
        va = is_inner[row_user] & (fold_of_row != f)
        te = fold_of_row == f
        m = fit_ranker(X[tr], y[tr], row_user[tr], X[va], y[va], row_user[va], seed + f, params=params)
        scores[te] = m.predict(X[te])
        best_iter.append(int(m.best_iteration) + 1)
        b = m.get_booster()
        b.feature_names = list(names)
        g = b.get_score(importance_type="gain")
        tot = sum(g.values()) or 1.0
        imp.append({n: g.get(n, 0.0) / tot for n in names})
        log(f"  fold {f}: trees {best_iter[-1]}, train rows {int(tr.sum())}, scored rows {int(te.sum())}")
        del m
    return scores, best_iter, {n: float(np.mean([d[n] for d in imp])) for n in names}


GROUPS = {"two-tower": ["tt_score", "tt_rank"], "NCF": ["ncf_logit", "ncf_user_known", "ncf_item_known"],
          "popularity (as of request)": ["pop_asof"], "content similarity": ["content_sim"], "genre match": ["genre_aff"],
          "item age": ["item_age_days"], "lane flag": ["lane"], "hour / day-of-week": ["hour", "dow"],
          "session": ["mins_since_last", "in_session", "session_len", "session_like_rate", "sess_genre_match", "sess_content_sim"],
          "device (SIMULATED)": ["dev_tv", "dev_mobile", "dev_web", "dev_tablet"]}


def group_importance(imp: dict) -> dict:
    """Sum normalised gain by feature group; item genre one-hots (g_*) form their own group."""
    out = {g: sum(imp.get(n, 0.0) for n in names) for g, names in GROUPS.items()}
    out["item genre one-hots"] = sum(v for n, v in imp.items() if n.startswith("g_"))
    return {g: v for g, v in out.items() if v > 0 or g in GROUPS}
