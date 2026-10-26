"""Module 9, step 1 (local only): build the Lambda bundle and the per-user DynamoDB records from the frozen Module 8 'real' artifacts.

Reads train + validation history and artifacts that were frozen BEFORE the test was unlocked. Never reads the test portion, never contacts AWS or Zilliz.
Outputs under data/processed/<run>/serving/: bundle/ (items.npz, ncf.npz, ranker_*.npz, titles.json), users.npz (records), build_facts.json.
Run:  python scripts/guarded_run.py scripts/build_serving_artifacts.py [--emb-dtype float32|float16]"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content, testprep
from ott_recsys.config import Config
from ott_recsys.context import genre_matrix
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.memguard import Stages
from ott_recsys.ncf import NCFScorer
from ott_recsys.serving_offline import build_user_arrays, write_bundle

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dtype", default="float32", choices=["float32", "float16"])
args = ap.parse_args()
cfg = Config()
stages = Stages()
SV = cfg.out_dir / "serving"
TP = cfg.out_dir / "test_prep"
REAL, FZ = TP / "real", TP / "frozen"
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
split = load_user_split(cfg)
train, val = split.train, split.val  # the test portion is never touched
history = testprep.history_frame(train, val, rehearsal=False)
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
gm, gnames = genre_matrix(movies, catalog)
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
rel = content.release_dates(movies, links, tmdb).reindex(catalog)
rel_ts = np.array([x.timestamp() if pd.notna(x) else np.nan for x in rel])
train_counts = np.bincount(pd.Index(catalog).get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog)).astype(float)
known = train_counts > 0
V = np.load(cfg.out_dir / "two_tower" / "a_id_n20" / "item_vecs.npy")
stale = NCFScorer.load(cfg.out_dir / "ncf" / "ncf_main" / "weights.npz")
fold = NCFScorer.load(REAL / "ncf_foldin.npz")
assert all(np.array_equal(stale.a[k], fold.a[k]) for k in ("ig", "im", "Wo", "bo", "W0", "W1", "W2")), "fold-in must leave items and the MLP untouched"
ncf_arrays = {k: stale.a[k] for k in ("ig", "im", "Wo", "bo", "W0", "b0", "W1", "b1", "W2", "b2")}
rankers = {"M2": FZ / "final_M2.json"}
if (SV / "final_M3b.json").exists():
    rankers["M3b"] = SV / "final_M3b.json"  # synthetic-device DEMO model (scripts/refit_m3b.py)
stages.done("inputs")

sizes = write_bundle(SV / "bundle", catalog=catalog, V=V, emb_c=emb_c, genre_m=gm, genre_names=gnames, rel_ts=rel_ts, known=known, history=history,
                     ncf_arrays=ncf_arrays, ncf_item_ids=stale.item_ids, titles=movies.set_index("movieId").reindex(catalog)["title"].fillna("").tolist(),
                     rankers=rankers, emb_dtype=np.dtype(args.emb_dtype))
stages.done("bundle")

lists = np.load(REAL / "lists.npz")
hv = np.load(REAL / "history_vectors.npz")
users = lists["users"]
arrays = build_user_arrays(history, users, catalog, emb_c, gm, lists["cand"], hv["tt_ids"], hv["tt_U"], fold)
np.savez_compressed(SV / "users.npz", **arrays)
stages.done("user records")

facts = {"emb_dtype": args.emb_dtype, "users": int(len(users)), "users_with_ttvec": int(arrays["has_ttvec"].sum()), "users_with_profile": int(arrays["has_prof"].sum()),
         "users_ncf_known": int(arrays["ncf_known"].sum()), "mean_seen": float(np.diff(arrays["seen_off"]).mean()), "max_seen": int(np.diff(arrays["seen_off"]).max()),
         "bundle_bytes": sizes, "bundle_total_mb": round(sum(sizes.values()) / 1e6, 1), "rankers": sorted(rankers)}
(SV / "build_facts.json").write_text(json.dumps(facts, indent=1))
print(json.dumps(facts, indent=1), flush=True)
