"""Module 7 prerequisite: robustness check of the Module 6 context conclusion. Reruns ranker M0, M1, M2 with a
lower learning rate (0.03, was 0.08), a larger inner early-stopping set (25% of each training fold's users, was
10%), patience 100 (was 30) and up to 2000 trees. Same 5 user-grouped folds, so paired with the old runs.
Out-of-fold only. Test stays sealed. Resumable."""
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import paired_diff, summarize
from ott_recsys.memguard import Stages
from ott_recsys.rankfeat import BASE, CTX_SESSION, CTX_TIME
from ott_recsys.ranker import PARAMS, oof_scores
from ott_recsys.rerank import all_orders, hit_matrix, metrics_from_ranks, rank_positions

NEW = dict(learning_rate=0.03, n_estimators=2000, early_stopping_rounds=100)
INNER = 0.25
cfg = Config()
stages = Stages()
RK = cfg.out_dir / "ranker"
z = np.load(RK / "features.npz")
X, cols = z["X"], [str(c) for c in z["cols"]]
row_user, row_slot, cand_f, ncf_f, users, y = z["row_user"].astype(np.int64), z["row_slot"].astype(np.int64), z["cand_f"].astype(np.int64), z["ncf_f"], z["users"], z["y"].astype(int)
U, W = cand_f.shape
valid = cand_f >= 0
catalog = np.sort(load_movies(cfg)["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
split = load_user_split(cfg)
counts = np.bincount(cat_idx.get_indexer(split.train.loc[split.train["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in split.val[split.val["label"] == 1].groupby("user_id")}
t_list = [truth[u] for u in users]
n_truth = np.array([len(t) for t in t_list])
hit = hit_matrix(cand_f, t_list)
hold = ~np.isin(users, np.random.default_rng(0).choice(users, cfg.tune_users, replace=False))
CI = {c: i for i, c in enumerate(cols)}
f_base = [CI[c] for c in BASE + [c for c in cols if c.startswith("g_")]]
VARS = {"M0": f_base, "M1": f_base + [CI[c] for c in CTX_TIME], "M2": f_base + [CI[c] for c in CTX_TIME] + [CI[c] for c in CTX_SESSION]}
new_oof, new_meta = {}, {}
for name, idx in VARS.items():
    ps, pm = RK / f"oof_R_{name}.npy", RK / f"oof_R_{name}.json"
    if ps.exists() and pm.exists():
        new_oof[name], new_meta[name] = np.load(ps), json.loads(pm.read_text())
        print(f"{name}: loaded", flush=True)
        continue
    t0 = time.time()
    print(f"R_{name}: {len(idx)} features; 5 folds, lr {NEW['learning_rate']}, inner set {INNER:.0%}", flush=True)
    s, best, imp = oof_scores(X[:, idx], y, row_user, U, [cols[i] for i in idx], k=5, seed=0, inner_frac=INNER, log=lambda m: print(m, flush=True), params=NEW)
    new_oof[name], new_meta[name] = s, {"best_iter": best, "seconds": round(time.time() - t0)}
    np.save(ps, s)
    pm.write_text(json.dumps(new_meta[name]))
    stages.done(f"R_{name}")

rand = np.random.default_rng(0).random((U, W))
ranks = all_orders(cand_f, hit, counts, {"main": ncf_f}, seed=0)
for tag, src in [("old", {n: np.load(RK / f"oof_{n}.npy") for n in VARS}), ("new", new_oof)]:
    for n, s in src.items():
        S = np.full((U, W), np.nan)
        S[row_user, row_slot] = s
        ranks[f"{n}_{tag}"] = rank_positions(valid, S, rand)
M = {m: metrics_from_ranks(r, hit, n_truth, users) for m, r in ranks.items()}
old_meta = {n: json.loads((RK / f"oof_{n}.json").read_text()) for n in VARS}
C = ["e2e_ndcg10", "e2e_recall20", "rr_ndcg10", "rr_recall20"]


def fmt(df):
    t = summarize(df[C], float("nan"), n_boot=1000)
    return {c: f"{t.loc[c, 'mean']:.4f} [{t.loc[c, 'lo']:.3f}-{t.loc[c, 'hi']:.3f}]" for c in C}


def diff(a, b, cols=C):
    out = []
    for c in cols:
        m, lo, hi = paired_diff(M[a][hold][c], M[b][hold][c])
        out.append(f"{c} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("*" if lo > 0 or hi < 0 else ""))
    return "  ".join(out)


lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "",
         f"ROBUSTNESS CHECK of the Module 6 context conclusion. Old: {PARAMS} with inner set 10%.",
         f"New: learning_rate 0.03, up to 2000 trees, early stopping patience 100, inner early-stopping set {INNER:.0%} of each training fold's users. Same 5 user-grouped folds; out-of-fold; held-out 15,530 users; test locked.", "",
         "== HELD-OUT USERS, end-to-end and reachable-only: mean [95% CI] =="]
tab = {"tt (two-tower order)": fmt(M["tt"][hold])}
for n in VARS:
    tab[f"{n} old (lr 0.08)"], tab[f"{n} new (lr 0.03)"] = fmt(M[f"{n}_old"][hold]), fmt(M[f"{n}_new"][hold])
lines += [pd.DataFrame(tab).T.to_string(), "", "== PAIRED DIFFERENCES (* = interval excludes 0) =="]
for tag in ("old", "new"):
    lines += [f"-- {tag} --", f"  M1 - M0 (hour/day):  {diff(f'M1_{tag}', f'M0_{tag}')}", f"  M2 - M1 (session):   {diff(f'M2_{tag}', f'M1_{tag}')}",
              f"  M2 - M0 (all context): {diff(f'M2_{tag}', f'M0_{tag}')}"]
lines += ["-- new minus old, same variant --"] + [f"  {n}: {diff(f'{n}_new', f'{n}_old')}" for n in VARS]
lines += [f"  M0 new - tt: {diff('M0_new', 'tt')}", "", "== TREES PER FOLD (early stopping) =="]
lines += [f"{n}: old {old_meta[n]['best_iter']}   new {new_meta[n]['best_iter']}  ({new_meta[n]['seconds']} s)" for n in VARS]
d_old = paired_diff(M["M2_old"][hold]["e2e_ndcg10"], M["M0_old"][hold]["e2e_ndcg10"])
d_new = paired_diff(M["M2_new"][hold]["e2e_ndcg10"], M["M0_new"][hold]["e2e_ndcg10"])
lines += ["", f"SUMMARY: all-context gain (M2 - M0, e2e NDCG@10) old {d_old[0]:+.4f} [{d_old[1]:+.4f}, {d_old[2]:+.4f}]; new {d_new[0]:+.4f} [{d_new[1]:+.4f}, {d_new[2]:+.4f}]."]
text = "\n".join(lines)
print(text)
(ROOT / "reports" / f"module7_ranker_robustness_{cfg.run_name}.txt").write_text(text)
stages.done("report")
