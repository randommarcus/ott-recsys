"""Materialise the 230-slot candidate lists from Module 4b: ID-only two-tower top 200 + up to 30 appended
new-release-lane picks (centered-SBERT content search restricted to movies with no train likes; in the global
cutoff version, to movies released since cutoff-6mo). Saved for NCF (Module 5) and the later ranker.
Uses train / global-train data and validation USERS only for list membership; never the test portion."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.baselines import ContentRetrieval, RandomTies
from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import evaluate, evaluate_lists
from ott_recsys.memguard import Stages
from ott_recsys.newrelease import release_sets
from ott_recsys.rerank import build_candidate_lists
from ott_recsys.twotower import TwoTowerRetrieval

cfg = Config()
stages = Stages()
TT = cfg.out_dir / "two_tower"
out_dir = cfg.out_dir / "candidates"
out_dir.mkdir(exist_ok=True)
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
split = load_user_split(cfg)
train, val = split.train, split.val
counts = np.bincount(pd.Index(catalog).get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog))

# ---- per-user validation lists
content_model = ContentRetrieval(emb_c)
content_model.fit(train, catalog)
lane = evaluate(content_model, train, val, catalog, candidate_items=catalog[counts == 0])
main_top = np.load(TT / "a_id_n20" / "top200.npy").astype(np.int64)
users = np.load(TT / "a_id_n20" / "eval_users.npy")
assert (lane.users == users).all()
cand, added = build_candidate_lists(main_top, lane.top, 200, 30)
# reproduce the Module 4b numbers (Recall at the full 230 width: all 0.6501, cold 0.0131)
pu = evaluate_lists(cand, train, val, catalog, ks=(230,))
all_r, cold_r = pu["all_recall@230"].mean(), pu["cold_recall@230"].mean()
print(f"val lists: {cand.shape}, lane picks per user mean {added.mean():.1f}; Recall@230 all {all_r:.4f} cold {cold_r:.4f}")
assert abs(all_r - 0.6501) < 5e-4 and abs(cold_r - 0.0131) < 5e-4, "lists do not reproduce Module 4b"
np.savez(out_dir / "candidates_val.npz", users=users, cand=cand.astype(np.int32), n_main=200)
stages.done("validation candidates")

# ---- global-cutoff lists (robustness check): fit on ratings <= cutoff, users with a profile and >= 1 val-window like
train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
sets = release_sets(train_g, val_g, content.release_dates(movies, links, tmdb), months=6)
profile = set(train_g.loc[train_g["label"] == 1, "user_id"])
ev = val_g[val_g["user_id"].isin(profile)]
w = np.load(TT / "g_id_n20" / "user_tower.npz")
tt_g = TwoTowerRetrieval(np.load(TT / "g_id_n20" / "item_vecs.npy"), w["W"], w["b"], 20)
tt_g.fit(train_g, catalog)
cg = ContentRetrieval(emb_c)
cg.fit(train_g, catalog)
main_g = evaluate(RandomTies(tt_g, seed=1), train_g, ev, catalog)
lane_g = evaluate(RandomTies(cg, seed=1), train_g, ev, catalog, candidate_items=sets["release_date"])
assert (main_g.users == lane_g.users).all()
cand_g, added_g = build_candidate_lists(main_g.top, lane_g.top, 200, 30)
print(f"global lists: {cand_g.shape}, lane picks per user mean {added_g.mean():.1f}")
np.savez(out_dir / "candidates_global.npz", users=main_g.users, cand=cand_g.astype(np.int32), n_main=200)
stages.done("global candidates")
