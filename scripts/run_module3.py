"""Module 3 evaluation: content-only retrieval (per-user validation) and the
new-release evaluation (global cutoff, train + val only). Test stays sealed."""
import dataclasses
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd

from ott_recsys import content
from ott_recsys.baselines import ContentRetrieval, GlobalPop, ItemKNN, RandomRec, RandomTies, knn_neighbors
from ott_recsys.config import Config
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import evaluate, paired_diff, summarize
from ott_recsys.experiments import results_table
from ott_recsys.memguard import Stages
from ott_recsys.newrelease import DEFINITIONS, MODES, evaluate_new_release, format_table, release_sets

pd.set_option("display.width", 220)
cfg = Config()
stages = Stages()
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
emb = {d: content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, w) for d, w in [("384", "emb384"), ("64", "emb64")]}
# diagnostic: 384-d with the mean vector removed (what PCA does implicitly), re-normalised
_c = emb["384"] - emb["384"].mean(axis=0, keepdims=True)
emb["384c"] = (_c / np.maximum(np.linalg.norm(_c, axis=1, keepdims=True), 1e-12)).astype(np.float32)
best = json.loads((ROOT / "reports" / f"module2_validation_{cfg.run_name}.json").read_text())["best"]["item_knn"]
knn_k, knn_shrink = int(best["k"]), float(best["shrink"])
lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "",
         f"item_knn settings (tuned in Module 2): {best}", ""]


def make_models(fit_df, include_knn=True):
    ms = {"random": RandomRec(), "global_pop": GlobalPop(), "content_384": ContentRetrieval(emb["384"]),
          "content_384c": ContentRetrieval(emb["384c"]), "content_64": ContentRetrieval(emb["64"])}
    if include_knn:
        ms["item_knn"] = None  # filled below (shared neighbour pass)
    for n, m in ms.items():
        if m is not None:
            m.fit(fit_df, catalog)
    if include_knn:
        state = knn_neighbors(fit_df, catalog, [knn_shrink], knn_k)
        ms["item_knn"] = ItemKNN().use(state, knn_shrink, knn_k)
    return {k: ms[k] for k in ["random", "global_pop", "item_knn", "content_384", "content_384c", "content_64"]}


# ---------------------------------------------------------------- A. per-user validation
split = load_user_split(cfg)
train, val = split.train, split.val
models = make_models(train)
stages.done("fit models (per-user split)")
res = {n: evaluate(m, train, val, catalog) for n, m in models.items()}
tables = {n: summarize(r.per_user, r.coverage) for n, r in res.items()}
stages.done("evaluate on per-user validation")
pu = {n: r.per_user for n, r in res.items()}
paired = {}
for lab, (x, y) in {"content_384 - global_pop": ("content_384", "global_pop"), "content_384 - item_knn": ("content_384", "item_knn"),
                    "content_384 - content_64": ("content_384", "content_64"),
                    "content_384c - content_384": ("content_384c", "content_384"), "content_64 - content_384c": ("content_64", "content_384c"), "item_knn - global_pop": ("item_knn", "global_pop")}.items():
    for col in ["all_ndcg@10", "all_recall@50", "warm_recall@50"]:
        paired[f"{lab} | {col}"] = paired_diff(pu[x][col], pu[y][col])
lines += ["== A. PER-USER VALIDATION (fit on train; test locked) ==", ""]
for view, label in [("all", "ALL validation likes"), ("warm", "WARM likes"), ("cold", "COLD likes (item never seen in train)")]:
    lines += [f"--- {label} (mean [95% CI over users]) ---", results_table(tables, view).to_string(), ""]
lines += ["--- paired differences (same users) ---"] + [
    f"{k:48s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else "  (not distinguishable from 0)")
    for k, (m, lo, hi) in paired.items()] + [""]

# sample cold likes: what does content retrieval surface?
titles = movies.set_index("movieId")["title"].to_dict()
cold_users = pu["content_384"]["cold_recall@50"].dropna()
hit_users = cold_users[cold_users > 0].index[:3]
lines.append("--- examples: users whose COLD (never-seen-in-train) likes appear in content_384's top 50 ---")
for u in hit_users:
    row = int(np.where(res["content_384"].users == u)[0][0])
    top50 = {catalog[i] for i in res["content_384"].top[row, :50]}
    seen = set(train.loc[train.user_id == u, "item_id"])
    cold_likes = val[(val.user_id == u) & (val.label == 1) & ~val.item_id.isin(set(train.item_id))]["item_id"]
    liked = train[(train.user_id == u) & (train.label == 1)].tail(5)["item_id"]
    lines.append(f"user {u}: recent train likes: " + "; ".join(titles[i] for i in liked))
    lines.append("     cold likes found in top 50: " + "; ".join(titles[i] for i in cold_likes if i in top50))
lines.append("")
stages.done("report A")

# ---------------------------------------------------------------- B. new releases (global cutoff)
train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
rel = content.release_dates(movies, links, tmdb)
sets = release_sets(train_g, val_g, rel, months=6)
gm = {n: RandomTies(m, seed=1) for n, m in make_models(train_g).items()}
stages.done("fit models (global cutoff)")
nr = evaluate_new_release(gm, train_g, val_g, catalog, sets)
stages.done("evaluate new releases")
lines += ["== B. NEW-RELEASE EVALUATION (global cutoff; fit on everything up to the cutoff; val window only) ==",
          f"cutoff (end of global train): {sets['cutoff']}   validation window ends: {sets['val_end']}",
          f"PRIMARY: TMDB release_date between {sets['window_start']:%Y-%m-%d} (cutoff - 6 months) and the window end: "
          f"{len(sets['release_date'])} movies, of which {sets['primary_with_no_train_ratings']} have no ratings in train",
          f"SECONDARY: first rated after the cutoff: {len(sets['first_rated'])} movies; overlap with primary: {sets['overlap']}",
          "Ties between equal scores are broken at random (seeded). Users need >= 1 liked item before the cutoff.", ""]
LABEL = {"release_date": "PRIMARY: released since cutoff-6mo (all)", "release_cold": "  primary, COLD part: no train ratings",
         "release_warm": "  primary, WARM part: already rated in train", "first_rated": "SECONDARY: first rated after cutoff"}
for d in DEFINITIONS:
    for mode in MODES:
        r = nr[(d, mode)]
        lines += [f"--- {LABEL[d]} | {'full catalog' if mode == 'full' else 'new-items pool only'} | users={r['n_users']}, "
                  f"pool/new items={r['n_items']}, likes={r['n_likes']} ---"]
        lines += [format_table(r["tables"]).to_string() if r["tables"] else "(no evaluable users)", ""]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module3_content_{cfg.run_name}.txt").write_text(text)
stages.done("write report")
