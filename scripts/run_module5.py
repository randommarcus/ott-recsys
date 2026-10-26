"""Module 5 report: NCF alone as a re-ranker of the 230-slot candidate lists (validation, per-user split),
checks, global-cutoff robustness and the offline-feature export. Test stays sealed."""
import dataclasses
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import paired_diff, summarize
from ott_recsys.memguard import Stages
from ott_recsys.ncf import NCFScorer, score_candidates
from ott_recsys.rerank import all_orders, bucket_recall, hit_matrix, metrics_from_ranks, nan_share

pd.set_option("display.width", 230)
cfg = Config()
stages = Stages()
NC = cfg.out_dir / "ncf"
catalog = np.sort(load_movies(cfg)["movieId"].to_numpy())
cat_idx = pd.Index(catalog)
split = load_user_split(cfg)
train, val = split.train, split.val
counts = np.bincount(cat_idx.get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog))
pop = counts.astype(float)
lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), ""]
COLS = ["e2e_ndcg10", "e2e_recall20", "rr_ndcg10", "rr_recall20"]


def fmt(df, cols=COLS):
    t = summarize(df[cols], float("nan"), n_boot=1000)
    return {c: f"{t.loc[c, 'mean']:.4f} [{t.loc[c, 'lo']:.3f}-{t.loc[c, 'hi']:.3f}]" for c in cols} | {"users": int(t.loc[cols[0], "n_users"])}


def method_table(M, mask, order):
    rows = {m: fmt(M[m][mask]) for m in order}
    rnd, orc = M["random"][mask]["rr_ndcg10"].mean(), M["oracle"][mask]["rr_ndcg10"].mean()
    for m in order:
        rows[m]["headroom captured (rr NDCG)"] = f"{(M[m][mask]['rr_ndcg10'].mean() - rnd) / (orc - rnd):.1%}"
    return pd.DataFrame(rows).T


def paired_lines(M, mask, pairs, cols=("e2e_ndcg10", "e2e_recall20", "rr_ndcg10", "rr_recall20")):
    out = []
    for a, b in pairs:
        for c in cols:
            m, lo, hi = paired_diff(M[a][mask][c], M[b][mask][c])
            out.append(f"  {a + ' - ' + b:30s} {c:14s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
    return out


# ---------------------------------------------------------------- validation
z = np.load(cfg.out_dir / "candidates" / "candidates_val.npz")
users, cand = z["users"], z["cand"].astype(np.int64)
valid = cand >= 0
truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in val[val["label"] == 1].groupby("user_id")}
t_list = [truth[u] for u in users]
n_truth = np.array([len(t) for t in t_list])
hit = hit_matrix(cand, t_list)
sel = np.isin(users, np.random.default_rng(0).choice(users, cfg.tune_users, replace=False))  # the 3,000 selection users
hold, allu = ~sel, np.ones(len(users), bool)

runs = {"main": "ncf_main", "uniform": "ncf_uniform", "k8": "ncf_k8"}
scores = {k: np.load(NC / r / "scores.npy") for k, r in runs.items()}
main_scorer = NCFScorer.load(NC / "ncf_main" / "weights.npz")
scores["shuffled"] = score_candidates(main_scorer.shuffled_users(0), users, cand, catalog)
ranks = all_orders(cand, hit, pop, scores, seed=0)
M = {m: metrics_from_ranks(r, hit, n_truth, users) for m, r in ranks.items()}
stages.done("orders + metrics")

metas = {k: json.loads((NC / r / "meta.json").read_text()) for k, r in runs.items()}
lines += [f"== VALIDATION: {len(users)} users with >= 1 validation like; candidate lists = ID-only two-tower top 200 + up to 30 appended "
          f"new-release-lane picks ({int(valid.sum(1).mean())} slots on average); test locked ==",
          f"selection users (early stopping): {int(sel.sum())}; HELD-OUT users (never used for selection): {int(hold.sum())}",
          f"reachable users (>= 1 validation like inside the list): {int((hit.sum(1) > 0).sum())} ({(hit.sum(1) > 0).mean():.1%}); "
          f"mean likes {n_truth.mean():.1f}, inside list {hit.sum(1).mean():.1f}",
          "e2e = all users, truth = ALL validation likes (capped by retrieval). rr = reachable users, truth restricted to the list "
          "(pure ordering quality). Ties broken at random.", ""]
order = ["random", "pop", "tt", "ncf_main", "blend_main", "ncf_uniform", "ncf_k8", "ncf_shuffled", "oracle"]
NAMES = {"tt": "tt (two-tower's own order)", "pop": "pop (popularity order)", "blend_main": "blend (untuned 50/50 tt+ncf)",
         "ncf_shuffled": "ncf, user embeddings shuffled"}
for label, mask in [("HELD-OUT users (primary)", hold), ("ALL users", allu), ("SELECTION users (used for early stopping)", sel)]:
    t = method_table(M, mask, order)
    t.index = [NAMES.get(i, i) for i in t.index]
    lines += [f"--- {label}: mean [95% CI over users] ---", t.to_string(), ""]
lines += ["--- paired differences, HELD-OUT users (* = interval excludes 0) ---"]
lines += paired_lines(M, hold, [("ncf_main", "tt"), ("ncf_main", "pop"), ("ncf_main", "random"), ("blend_main", "tt"), ("blend_main", "ncf_main"),
                                ("ncf_uniform", "ncf_main"), ("ncf_k8", "ncf_main"), ("ncf_shuffled", "ncf_main"), ("ncf_shuffled", "pop")])
lines += ["", f"selection optimism (ncf_main e2e NDCG@10): selection users {M['ncf_main'][sel]['e2e_ndcg10'].mean():.4f} vs held-out "
          f"{M['ncf_main'][hold]['e2e_ndcg10'].mean():.4f}", ""]

lines += ["--- training runs ---"]
for k, m in metas.items():
    lines.append(f"{m['name']:12s} neg_k={m['neg_k']} pop_frac={m['pop_frac']} best_epoch={m['best_epoch']} epochs={m['epochs_trained']} "
                 f"minutes={m['train_minutes']} known users/items={m['known_users']}/{m['known_items']}")
lines += ["ncf_main learning curve (selection users):"] + [f"  epoch {h['epoch']:2d} loss {h['loss']:.4f}  NDCG@10 {h['ndcg@10']:.4f}  Recall@20 {h['recall@20']:.4f}  ({h['train_s']:.0f}s)"
                                                       for h in metas["main"]["history_log"]]
lines.append("")

# ---------------------------------------------------------------- NaN share (items / users NCF cannot score)
item_known = counts > 0
user_known = np.isin(users, main_scorer.user_ids)
ns = nan_share(cand, hit, int(z["n_main"]), item_known, user_known)
lines += ["== VALIDATION LIKES INSIDE THE 230 THAT NCF CANNOT SCORE (all users) ==",
          f"evaluated users NCF cannot score at all (no train likes): {int((~user_known).sum())} of {len(users)}"]
for lane, d in ns.items():
    lines.append(f"{lane:17s} slots {d['slots']:>8d} (of which item has no train like: {d['slots_item_nan']:>7d} = {d['slots_item_nan'] / max(d['slots'], 1):.1%}) | "
                 f"likes inside {d['likes_inside']:>6d}: on NaN item {d['on_item_nan']:>5d} = {d['on_item_nan'] / max(d['likes_inside'], 1):.2%}, "
                 f"user NaN {d['on_user_nan']:>3d}, either {d['on_either_nan']:>5d} = {d['on_either_nan'] / max(d['likes_inside'], 1):.2%}")
tot = sum(d["likes_inside"] for d in ns.values())
lines += [f"combined: likes inside the 230 {tot}; on NaN item {sum(d['on_item_nan'] for d in ns.values())} = "
          f"{sum(d['on_item_nan'] for d in ns.values()) / tot:.2%}; either {sum(d['on_either_nan'] for d in ns.values())} = "
          f"{sum(d['on_either_nan'] for d in ns.values()) / tot:.2%}", ""]

# ---------------------------------------------------------------- popularity buckets
names, br = bucket_recall({m: ranks[m] for m in ["random", "pop", "tt", "ncf_main", "blend_main"]}, hit, cand, counts, 20, np.flatnonzero(hold))
lines += ["== SHARE OF IN-LIST LIKED ITEMS FOUND IN THE TOP 20, BY HOW POPULAR THE LIKED ITEM IS (held-out users) ==",
          pd.DataFrame(br, index=names).T.round(3).to_string(), ""]

# ---------------------------------------------------------------- global-cutoff robustness
zg = np.load(cfg.out_dir / "candidates" / "candidates_global.npz")
ug, cg = zg["users"], zg["cand"].astype(np.int64)
train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
pop_g = np.bincount(cat_idx.get_indexer(train_g.loc[train_g["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
tg = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in val_g[val_g["label"] == 1].groupby("user_id")}
tg_list = [tg[u] for u in ug]
hit_g = hit_matrix(cg, tg_list)
sc_g = np.load(NC / "ncf_global" / "scores.npy")
ranks_g = all_orders(cg, hit_g, pop_g, {"global": sc_g}, seed=0)
Mg = {m: metrics_from_ranks(r, hit_g, np.array([len(t) for t in tg_list]), ug) for m, r in ranks_g.items()}
meta_g = json.loads((NC / "ncf_global" / "meta.json").read_text())
lines += [f"== GLOBAL-CUTOFF ROBUSTNESS (fit on ratings <= 2015-02-03, fixed {meta_g['epochs_trained']} epochs from the main run; users with a "
          f"profile and >= 1 validation-window like: {len(ug)}; few users -> wide intervals) =="]
tg_tab = pd.DataFrame({m: fmt(Mg[m]) for m in ["random", "pop", "tt", "ncf_global", "blend_global", "oracle"]}).T
lines += [tg_tab.to_string(), "", "paired differences (* = interval excludes 0):"]
lines += paired_lines(Mg, np.ones(len(ug), bool), [("ncf_global", "tt"), ("ncf_global", "pop"), ("ncf_global", "random"), ("blend_global", "tt")])
lines.append(f"NCF cannot score {float(np.isnan(sc_g[cg >= 0]).mean()):.1%} of the global candidate slots (no train likes at the cutoff)\n")
stages.done("report")

# ---------------------------------------------------------------- export: scorer + offline feature table
ex = NC / "export"
ex.mkdir(exist_ok=True)
shutil.copy(NC / "ncf_main" / "weights.npz", ex / "ncf_weights.npz")
U, W = cand.shape
flat_ok = valid.ravel()
item_ids = np.where(valid, catalog[np.maximum(cand, 0)], -1).ravel()[flat_ok]
user_ids = np.repeat(users, W)[flat_ok]
lane_col = np.tile((np.arange(W) >= int(z["n_main"])).astype(np.int8), U)[flat_ok]
pos_col = np.tile(np.arange(W, dtype=np.int16), U)[flat_ok]
ncf_logit = scores["main"].ravel()[flat_ok]
uk, ik = main_scorer.known(user_ids, item_ids)
pq.write_table(pa.table({"user_id": pa.array(user_ids.astype(np.int64)), "item_id": pa.array(item_ids.astype(np.int64)),
                         "lane": pa.array(lane_col), "cand_pos": pa.array(pos_col), "ncf_logit": pa.array(ncf_logit),
                         "ncf_user_known": pa.array(uk), "ncf_item_known": pa.array(ik)}), ex / "ncf_features_val_candidates.parquet")
chk = np.random.default_rng(1).choice(len(user_ids), 5000, replace=False)  # the exported numpy scorer reproduces the table
re = NCFScorer.load(ex / "ncf_weights.npz").score(user_ids[chk], item_ids[chk])
assert np.allclose(re, ncf_logit[chk], equal_nan=True, atol=1e-5), "exported scorer disagrees with the stored features"
(ex / "schema.json").write_text(json.dumps({
    "features_file": "ncf_features_val_candidates.parquet", "rows": int(len(user_ids)),
    "columns": {"user_id": "INT64", "item_id": "INT64", "lane": "0 = main lane (ID-only two-tower top 200), 1 = new-release lane",
                "cand_pos": "position in the 230-slot list (two-tower order)", "ncf_logit": "NeuMF logit; NaN when user or item is unknown",
                "ncf_user_known": "user had >= 1 train like", "ncf_item_known": "item had >= 1 train like"},
    "out_of_sample": "Every row is a validation-period candidate scored by a model trained on TRAIN only (never on these pairs). "
                     "Do NOT train a ranker on NCF scores of train interactions: they are in-sample and inflated.",
    "scorer": "NCFScorer (src/ott_recsys/ncf.py) loads ncf_weights.npz; numpy only, no TensorFlow. score(user_ids, item_ids) -> logits, NaN if unknown.",
    "missing_values": "NaN is meaningful ('NCF cannot score this pair'); XGBoost handles it natively."}, indent=2))
lines += ["== EXPORT ==", f"{ex}: ncf_weights.npz ({(ex / 'ncf_weights.npz').stat().st_size / 1e6:.1f} MB), ncf_features_val_candidates.parquet "
          f"({len(user_ids)} rows, {(ex / 'ncf_features_val_candidates.parquet').stat().st_size / 1e6:.0f} MB), schema.json",
          "numpy scorer verified against the stored features on 5,000 random pairs."]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module5_ncf_{cfg.run_name}.txt").write_text(text)
stages.done("export + write")
