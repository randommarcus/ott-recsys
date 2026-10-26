"""Train one NeuMF run on TRAIN positives only.

  --split user   : early-stop on NDCG@10 (tie-break Recall@20) over the 230-slot candidate lists of a seeded
                   3,000-user validation subsample; then score the lists of every validation user.
  --split global : fixed epoch count (--epochs), global-cutoff train; no validation data used for selection.
Never touches the test portion."""
import argparse
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
from ott_recsys.memguard import Stages
from ott_recsys.ncf import build_training_data, score_candidates, train
from ott_recsys.rerank import apply_user_fallback, hit_matrix, metrics_from_ranks, rank_positions

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--neg-k", type=int, default=4)
ap.add_argument("--pop-frac", type=float, default=0.5)
ap.add_argument("--split", choices=["user", "global"], default="user")
ap.add_argument("--epochs", type=int, default=None)
args = ap.parse_args()
if args.split == "global" and not args.epochs:
    sys.exit("--split global needs --epochs")
cfg = Config()
stages = Stages()
out = cfg.out_dir / "ncf" / args.name
out.mkdir(parents=True, exist_ok=True)
catalog = np.sort(load_movies(cfg)["movieId"].to_numpy())
if args.split == "user":
    split = load_user_split(cfg)
    train_df, val = split.train, split.val
    z = np.load(cfg.out_dir / "candidates" / "candidates_val.npz")
else:
    train_df, val = pd.read_parquet(cfg.out_dir / "train.parquet"), None
    z = np.load(cfg.out_dir / "candidates" / "candidates_global.npz")
users, cand = z["users"], z["cand"].astype(np.int64)
data = build_training_data(train_df)
pop = np.bincount(pd.Index(catalog).get_indexer(train_df.loc[train_df["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
print(f"run {args.name}: neg_k={args.neg_k} pop_frac={args.pop_frac} split={args.split}; {len(data['pos_u'])} positives, "
      f"{len(data['users'])} known users, {len(data['items'])} known items", flush=True)

eval_fn = None
if args.split == "user":
    sel_users = np.random.default_rng(0).choice(users, min(cfg.tune_users, len(users)), replace=False)  # same subsample as Modules 2/4
    sel = np.isin(users, sel_users)
    cat_idx = pd.Index(catalog)
    truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in val[val["label"] == 1].groupby("user_id")}
    t_sel = [truth[u] for u in users[sel]]
    n_truth = np.array([len(t) for t in t_sel])
    c_sel, valid_sel = cand[sel], cand[sel] >= 0
    hit_sel = hit_matrix(c_sel, t_sel)
    pop_sel = np.where(valid_sel, pop[np.maximum(c_sel, 0)], np.nan)
    rand_sel = np.random.default_rng(0).random(c_sel.shape)

    def eval_fn(scorer):
        sc = apply_user_fallback(score_candidates(scorer, users[sel], c_sel, catalog), valid_sel, pop_sel)
        m = metrics_from_ranks(rank_positions(valid_sel, sc, rand_sel), hit_sel, n_truth, users[sel])
        return {"ndcg@10": float(m["e2e_ndcg10"].mean()), "recall@20": float(m["e2e_recall20"].mean())}

t0 = time.time()
scorer, history, best_epoch = train(data, neg_k=args.neg_k, pop_frac=args.pop_frac, fixed_epochs=args.epochs if args.split == "global" else None,
                                    eval_fn=eval_fn, log=lambda m: print(m, flush=True))
minutes = (time.time() - t0) / 60
stages.done("training")
scorer.save(out / "weights.npz")
scores = score_candidates(scorer, users, cand, catalog).astype(np.float32)  # NaN = cannot score
np.save(out / "scores.npy", scores)
meta = {"name": args.name, "neg_k": args.neg_k, "pop_frac": args.pop_frac, "split": args.split, "best_epoch": int(best_epoch),
        "epochs_trained": len(history), "train_minutes": round(minutes, 2), "history_log": history,
        "known_users": int(len(data["users"])), "known_items": int(len(data["items"])), "positives": int(len(data["pos_u"])),
        "share_candidate_slots_nan": float(np.isnan(scores[cand >= 0]).mean())}
(out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
stages.done("score candidates + save")
print(f"DONE {args.name}: best_epoch={best_epoch} minutes={minutes:.1f}", flush=True)
