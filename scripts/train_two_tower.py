"""Train one two-tower run.

  --split user   : fit on per-user train, early-stop/select on a validation subsample
                   (Recall@200, tie-break Recall@50), evaluate on ALL validation users.
  --split global : fit on the global-cutoff train for a FIXED number of epochs (--epochs);
                   used for the new-release evaluation. No validation data is touched.
Never touches the test portion."""
import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import evaluate
from ott_recsys.memguard import Stages
from ott_recsys.twotower import TwoTowerRetrieval, build_pairs, export_inference, train, user_histories

ap = argparse.ArgumentParser()
ap.add_argument("--name", required=True)
ap.add_argument("--variant", choices=["id", "content", "both"], required=True)
ap.add_argument("--history", type=int, default=None)
ap.add_argument("--no-logq", action="store_true")
ap.add_argument("--split", choices=["user", "global"], default="user")
ap.add_argument("--epochs", type=int, default=None, help="fixed epochs (required for --split global)")
args = ap.parse_args()
cfg = Config()
N = args.history or cfg.history_len
if args.split == "global" and not args.epochs:
    sys.exit("--split global needs --epochs (taken from the matching per-user run)")

stages = Stages()
out = cfg.out_dir / "two_tower" / args.name
out.mkdir(parents=True, exist_ok=True)
movies = load_movies(cfg)
catalog = np.sort(movies["movieId"].to_numpy())
emb = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
print(f"run {args.name}: variant={args.variant} history={N} logq={not args.no_logq} split={args.split}", flush=True)
print("config:", {k: v for k, v in dataclasses.asdict(cfg).items() if k in
                  ("embed_dim", "temperature", "batch_size", "learning_rate", "id_dropout", "max_epochs", "patience", "model_seed")}, flush=True)

if args.split == "user":
    split = load_user_split(cfg)
    fit_df, val = split.train, split.val
else:
    fit_df, val = pd.read_parquet(cfg.out_dir / "train.parquet"), None  # global train only (<= cutoff)
pairs = build_pairs(fit_df, catalog, N)
likes_cnt = np.bincount(pd.Index(catalog).get_indexer(fit_df.loc[fit_df["label"] == 1, "item_id"]), minlength=len(catalog))
known = likes_cnt > 0
print(f"{len(pairs['pos'])} training pairs; {int((~known).sum())} catalog items have no train likes", flush=True)
stages.done("pairs")

eval_fn = None
if args.split == "user":
    liked_users = np.sort(val.loc[val["label"] == 1, "user_id"].unique())
    pick = np.random.default_rng(0).choice(liked_users, min(cfg.tune_users, len(liked_users)), replace=False)  # same subsample as Module 2 tuning
    val_t = val[val["user_id"].isin(pick)]
    h_uid, h_hist = user_histories(fit_df, catalog, N, users=pick)
    pop = likes_cnt.astype(np.float32)

    def eval_fn(model):
        art = export_inference(model, known)
        r = TwoTowerRetrieval(art["item_vecs"], art["W"], art["b"], N)
        r.prepare(h_uid, h_hist, pop)
        pu = evaluate(r, fit_df, val_t, catalog).per_user
        return {"recall@200": float(pu["all_recall@200"].mean()), "recall@50": float(pu["all_recall@50"].mean()),
                "ndcg@10": float(pu["all_ndcg@10"].mean())}

t0 = time.time()
model, history, best_epoch = train(
    pairs, len(catalog), emb if args.variant != "id" else None, args.variant, use_logq=not args.no_logq,
    dim=cfg.embed_dim, tau=cfg.temperature, batch_size=cfg.batch_size, lr=cfg.learning_rate, id_dropout=cfg.id_dropout,
    max_epochs=cfg.max_epochs, patience=cfg.patience, fixed_epochs=args.epochs if args.split == "global" else None,
    eval_fn=eval_fn, known_mask=known, seed=cfg.model_seed, log=lambda m: print(m, flush=True))
train_min = (time.time() - t0) / 60
stages.done("training")

art = export_inference(model, known)
np.save(out / "item_vecs.npy", art["item_vecs"])
np.savez(out / "user_tower.npz", W=art["W"], b=art["b"])
np.save(out / "known.npy", known)
meta = {"name": args.name, "variant": args.variant, "history": N, "logq": not args.no_logq, "split": args.split,
        "best_epoch": int(best_epoch), "epochs_trained": len(history), "train_minutes": round(train_min, 2),
        "pairs": int(len(pairs["pos"])), "history_log": history, "cold_items_in_catalog": int((~known).sum())}

if args.split == "user":
    retr = TwoTowerRetrieval(art["item_vecs"], art["W"], art["b"], N)
    retr.fit(fit_df, catalog)
    res = evaluate(retr, fit_df, val, catalog)
    res.per_user.to_parquet(out / "per_user.parquet")
    np.save(out / "top200.npy", res.top.astype(np.int32))
    np.save(out / "eval_users.npy", res.users)
    all_users = fit_df["user_id"].unique()
    meta.update({"train_users": int(len(all_users)), "train_users_no_likes": int(retr.no_likes(all_users).sum()),
                 "val_eval_users": int(len(res.users)), "val_eval_users_no_train_likes": int(retr.no_likes(res.users).sum()),
                 "coverage@10": res.coverage})
    print({k: meta[k] for k in ("train_users_no_likes", "val_eval_users_no_train_likes")}, flush=True)
(out / "meta.json").write_text(json.dumps(meta, default=float, indent=1))
stages.done("evaluate + save")
print(f"DONE {args.name}: best_epoch={best_epoch} train_min={train_min:.1f}", flush=True)
