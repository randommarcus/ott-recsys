"""Module 4b: candidate-source mixes (two-tower + centered-SBERT content), sample users, exports.
Validation (per-user split) and new-release evaluation (global cutoff, train + val only).
Test stays sealed."""
import dataclasses
import json
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
from ott_recsys.eval import append_top, evaluate, evaluate_lists, mix_top, paired_diff, summarize
from ott_recsys.export import export_collection, export_user_vectors
from ott_recsys.memguard import Stages
from ott_recsys.newrelease import new_release_tops, release_sets
from ott_recsys.twotower import TwoTowerRetrieval

pd.set_option("display.width", 230)
cfg = Config()
stages = Stages()
TT = cfg.out_dir / "two_tower"
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
emb384 = content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384")
emb_c = content.center_embeddings(emb384)
emb64 = content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb64")
split = load_user_split(cfg)
train, val = split.train, split.val

# name -> (two-tower source, slots from two-tower, slots from content source, content source)
# content source: "content" = centered-SBERT search over the WHOLE catalog; "lane" = the same search restricted to
# the new-release lane (validation: movies with no train likes; new-release eval: movies released since cutoff-6mo)
MIXES = {"a_id200": ("id", 200, 0, "content"), "b_id170+c30": ("id", 170, 30, "content"),
         "c_id150+c50": ("id", 150, 50, "content"), "d_both200": ("both", 200, 0, "content"),
         "e_both150+c50": ("both", 150, 50, "content"),
         # extra (not in the original list): budget controls and the restricted lane
         "g_id170_only": ("id", 170, 0, "content"), "i_id150_only": ("id", 150, 0, "content"),
         "f_id170+lane30": ("id", 170, 30, "lane"), "h_id150+lane50": ("id", 150, 50, "lane"),
         # keep the full 200 ID-only slots and APPEND lane picks (230 / 250 candidates in total)
         "j_id200+lane30(230)": ("id", 200, 30, "lane+"), "k_id200+lane50(250)": ("id", 200, 50, "lane+"),
         # equal-budget controls: the same total number of slots spent only on ID-only items
         "l_id230_only": ("id", 230, 0, "content"), "m_id250_only": ("id", 250, 0, "content")}
PAIRS = [("b_id170+c30", "a_id200"), ("c_id150+c50", "a_id200"), ("c_id150+c50", "b_id170+c30"),
         ("e_both150+c50", "d_both200"), ("d_both200", "a_id200"),
         ("g_id170_only", "a_id200"), ("i_id150_only", "a_id200"),
         ("f_id170+lane30", "a_id200"), ("f_id170+lane30", "g_id170_only"), ("f_id170+lane30", "b_id170+c30"),
         ("h_id150+lane50", "a_id200"), ("h_id150+lane50", "i_id150_only"), ("h_id150+lane50", "c_id150+c50"),
         ("j_id200+lane30(230)", "a_id200"), ("k_id200+lane50(250)", "a_id200"), ("j_id200+lane30(230)", "f_id170+lane30"),
         ("j_id200+lane30(230)", "l_id230_only"), ("k_id200+lane50(250)", "m_id250_only"), ("l_id230_only", "a_id200"), ("m_id250_only", "a_id200")]
lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "",
         "Mix rule: first n of the two-tower list + the first m content picks not already there, then fill to 200 "
         "from the two-tower list (duplicates skipped). 'content slots' = content picks that really took a slot.", ""]


def ci(t, col):
    return f"{t.loc[col, 'mean']:.4f} [{t.loc[col, 'lo']:.3f}-{t.loc[col, 'hi']:.3f}]"


def mix_report(title, pool, evaluate_mix):
    """pool: dict source -> (top lists); evaluate_mix(lists) -> per-user metrics DataFrame."""
    per, slots, width = {}, {}, {}
    for name, (src, n_a, n_c, csrc) in MIXES.items():
        if csrc.endswith("+"):
            lists, sl = append_top(pool[src][:, :n_a], pool[csrc[:-1]], n_c)
        elif n_c:
            lists, sl = mix_top(pool[src], pool[csrc], n_a, n_c)
        else:
            lists, sl = pool[src][:, :n_a], np.zeros(len(pool[src]), int)
        W = lists.shape[1]  # recall is measured at the full list width (200, or 230/250 for the append variants)
        p_ = evaluate_mix(lists, W)
        per[name] = p_[[f"{v}_recall@{W}" for v in ("all", "warm", "cold")]].rename(columns=lambda c: c.split("_")[0] + "_R")
        slots[name], width[name] = sl, W
    tabs = {n: summarize(p, float("nan"), n_boot=1000) for n, p in per.items()}
    no_cold = per["a_id200"]["cold_R"].isna().to_numpy()  # users with no cold like ("warm users")
    rows = {}
    for n, t in tabs.items():
        sl = slots[n]
        rows[n] = {"slots": width[n], "all R": ci(t, "all_R"), "warm R": ci(t, "warm_R"), "cold R": ci(t, "cold_R"),
                   "n_cold_users": int(t.loc["cold_R", "n_users"]),
                   "slots mean": f"{sl.mean():.1f}", "slots median": f"{np.median(sl):.0f}", "slots p90": f"{np.percentile(sl, 90):.0f}",
                   "slots (users w/o cold likes)": f"{sl[no_cold].mean():.1f}"}
    out = [f"--- {title} --- (R = Recall at the full list width)", pd.DataFrame(rows).T.to_string(), "", "paired differences (same users; * = interval excludes 0):"]
    for x, y in PAIRS:
        for col in ("all_R", "warm_R", "cold_R"):
            m, lo, hi = paired_diff(per[x][col], per[y][col])
            if not np.isnan(m):
                out.append(f"  {x + ' - ' + y:34s} {col:16s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
    return out + [""]


# ---------------------------------------------------------------- 1a. per-user validation
content_model = ContentRetrieval(emb_c)
content_model.fit(train, catalog)
res_c = evaluate(content_model, train, val, catalog)
lane_items = catalog[np.bincount(pd.Index(catalog).get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog)) == 0]
res_lane = evaluate(content_model, train, val, catalog, candidate_items=lane_items)
assert (res_lane.users == res_c.users).all()
pool = {"content": res_c.top, "lane": res_lane.top}
for key, run in [("id", "a_id_n20"), ("both", "a_both_n20")]:
    pool[key] = np.load(TT / run / "top200.npy").astype(np.int64)
    assert (np.load(TT / run / "eval_users.npy") == res_c.users).all()
w_ = np.load(TT / "a_id_n20" / "user_tower.npz")  # ID-only lists to 250 deep, for the equal-budget controls
retr_id = TwoTowerRetrieval(np.load(TT / "a_id_n20" / "item_vecs.npy"), w_["W"], w_["b"], 20)
retr_id.fit(train, catalog)
pool["id"] = evaluate(retr_id, train, val, catalog, ks=(10, 50, 200, 250)).top
assert (pool["id"][:, :200] == np.load(TT / "a_id_n20" / "top200.npy")).all(), "250-deep list disagrees with the saved 200-deep list"
lines += ["== 1a. PER-USER VALIDATION (fit on train; 18,530 users with >= 1 validation like; test locked) ==",
          f"lane = content search restricted to the {len(lane_items)} movies with no train likes"]
lines += mix_report("Recall@200 by candidate mix", pool, lambda L, W: evaluate_lists(L, train, val, catalog, ks=(W,)))
stages.done("validation mixes")

# ---------------------------------------------------------------- 1b. new releases
train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
sets = release_sets(train_g, val_g, content.release_dates(movies, links, tmdb), months=6)
gm = {"content": ContentRetrieval(emb_c)}
gm["content"].fit(train_g, catalog)
for key, run in [("id", "g_id_n20"), ("both", "g_both_n20")]:
    w = np.load(TT / run / "user_tower.npz")
    gm[key] = TwoTowerRetrieval(np.load(TT / run / "item_vecs.npy"), w["W"], w["b"], 20)
    gm[key].fit(train_g, catalog)
tops = new_release_tops({k: RandomTies(m, seed=1) for k, m in gm.items()}, train_g, val_g, catalog, sets, ks=(10, 50, 200, 250))
LAB = {"release_cold": "PRIMARY, COLD part (TMDB release since cutoff-6mo, no train ratings)", "release_date": "PRIMARY, all",
       "first_rated": "SECONDARY (first rated after cutoff)"}
lines += ["== 1b. NEW-RELEASE EVALUATION (global cutoff 2015-02-03; models fit on ratings <= cutoff; the 152-movie warm part is skipped "
          "because its pool is smaller than 200 slots) =="]
lane_pool = sets["release_date"]  # new-release lane: movies released since cutoff-6mo (known from TMDB metadata)
for (d, mode), v in tops.items():
    v["res"]["lane"] = evaluate(RandomTies(gm["content"], seed=1), train_g, v["ev"], catalog, candidate_items=lane_pool)
    pool = {k: r.top for k, r in v["res"].items()}
    title = f"{LAB[d]} | {'full catalog' if mode == 'full' else 'new-items pool only'} | users={len(v['res']['id'].users)}, items={v['n_items']}"
    lines += mix_report(title, pool, lambda L, W, v=v: evaluate_lists(L, train_g, v["ev"], catalog, ks=(W,)))
stages.done("new-release mixes")

# ---------------------------------------------------------------- 2. sample users
titles = movies.set_index("movieId")["title"].to_dict()
res_users = res_c.users
n_train = train.groupby("user_id").size().reindex(res_users)
order = res_users[np.argsort(n_train.to_numpy(), kind="stable")]
picks = [order[int(q * (len(order) - 1))] for q in np.linspace(0.1, 0.9, 5)]
seen_train_items = set(train["item_id"])
id_top = np.load(TT / "a_id_n20" / "top200.npy")
lines += ["== 2. SAMPLE USERS: ID-only two-tower, top 10 (validation likes shown; test is locked; same 5 users as Module 2) =="]
for u in picks:
    tr = train[train["user_id"] == u].sort_values(["timestamp", "item_id"])
    liked = tr[tr["label"] == 1]
    vl = val[(val["user_id"] == u) & (val["label"] == 1)]
    v_ids, row = set(vl["item_id"]), int(np.where(res_users == u)[0][0])
    top10 = catalog[id_top[row, :10]]
    hits = sum(i in v_ids for i in top10)
    lines += [f"=== user {u}: {len(tr)} train ratings ({len(liked)} likes), {len(vl)} validation likes; top-10 hits: {hits} ===",
              "  last liked in train: " + "; ".join(titles[i] for i in liked["item_id"].tail(8)),
              "  validation likes:    " + "; ".join(titles[i] + ("  [cold]" if i not in seen_train_items else "") for i in list(vl["item_id"])[:10]),
              "  ID-only two-tower top 10:"]
    lines += [f"     {r:2d}. {'✓' if i in v_ids else ' '} {titles[i]}" for r, i in enumerate(top10, 1)]
    lines.append("")
stages.done("samples")

# ---------------------------------------------------------------- 3. exports
counts = np.bincount(pd.Index(catalog).get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog))
known = counts > 0
idd = TT / "a_id_n20"
V_id, w_id = np.load(idd / "item_vecs.npy"), np.load(idd / "user_tower.npz")
exp1 = TT / "export_id_only"
st1 = export_collection(exp1, "movie_items_twotower_id", catalog, {"vector": V_id}, movies, counts, include=known,
                        notes="ID-only two-tower item vectors (64-d, unit length, IP == cosine). Only items with >= 1 train like are included: "
                              "the other items share one untrained 'unknown' vector and carry no information. Serve cold/new items from the "
                              "content collection. Query vector: numpy-only user tower (user_tower.npz).")
retr = TwoTowerRetrieval(V_id, w_id["W"], w_id["b"], 20)
retr.fit(train, catalog)
uid = np.array(sorted(retr.u_index))
export_user_vectors(exp1 / "user_vectors.parquet", uid, {"vector": retr.U[[retr.u_index[u] for u in uid]]})
np.savez(exp1 / "user_tower.npz", W=w_id["W"], b=w_id["b"], history_len=20)

exp2 = cfg.out_dir / "content" / "export_content"
st2 = export_collection(exp2, "movie_items_content", catalog, {"vec384": emb_c, "vec64": emb64}, movies, counts, include=None,
                        notes="Centered SBERT (all-MiniLM-L6-v2) vectors for EVERY movie: vec384 = mean-removed, unit length; vec64 = PCA-64 "
                              "(mean-removed by PCA), unit length. Query = normalised mean of the user's liked items' vectors from the same field "
                              "(no weights to learn). Text uses TMDB overviews + MovieLens tags dated before the 2015-02-03 cutoff.")
up = {}
for field, E in [("vec384", emb_c), ("vec64", emb64)]:
    m = ContentRetrieval(E)
    m.fit(train, catalog)
    up[field] = m
uid2 = np.array([u for u in sorted(up["vec384"].u_index) if np.linalg.norm(up["vec384"].profiles[up["vec384"].u_index[u]]) > 0])
export_user_vectors(exp2 / "user_vectors.parquet", uid2, {f: up[f].profiles[[up[f].u_index[u] for u in uid2]] for f in up})
RECIPE = {"id_only": "u = l2norm(mean(V[last 20 liked item ids]) @ W + b); search 'vector' with IP; drop items the user has seen.",
          "content": "u = l2norm(mean(vec[all liked item ids])); search vec384 (or vec64) with IP; drop seen items.",
          "mix": "dedupe: take first n from the two-tower search, then content hits not already there, fill from the two-tower list to the target size."}
for d, name in [(exp1, "id_only"), (exp2, "content")]:
    (d / "serving_recipe.json").write_text(json.dumps({"recipe": RECIPE[name], "mix": RECIPE["mix"]}, indent=2))
lines += ["== 3. EXPORTS ==", f"{exp1}: {st1}", f"{exp2}: {st2}",
          f"user vectors: ID-only {len(uid)} users; content {len(uid2)} users (users without any liked train item have no profile and are omitted; they get popularity)", ""]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module4b_candidate_mix_{cfg.run_name}.txt").write_text(text)
stages.done("exports + report")
