"""Module 4 report: two-tower runs vs baselines on VALIDATION (per-user split), candidate-union
evaluation, new-release evaluation (global cutoff), and the Milvus-ready export.
Reads the artefacts written by scripts/train_two_tower.py. Test stays sealed."""
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
from ott_recsys.eval import evaluate, evaluate_lists, paired_diff, summarize, union_top
from ott_recsys.export import export_milvus
from ott_recsys.memguard import Stages
from ott_recsys.newrelease import DEFINITIONS, MODES, evaluate_new_release, format_table, release_sets
from ott_recsys.twotower import TwoTowerRetrieval, better

pd.set_option("display.width", 230)
cfg = Config()
stages = Stages()
TT = cfg.out_dir / "two_tower"
movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
catalog = np.sort(movies["movieId"].to_numpy())
emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
knn_best = json.loads((ROOT / "reports" / f"module2_validation_{cfg.run_name}.json").read_text())["best"]["item_knn"]
K, SH = int(knn_best["k"]), float(knn_best["shrink"])
split = load_user_split(cfg)
train, val = split.train, split.val
lines = ["== CONFIG ==", *(f"{k:26s} {v}" for k, v in dataclasses.asdict(cfg).items()), "",
         f"item_knn settings (Module 2): {knn_best}", ""]


def load_run(name):
    d = TT / name
    if not (d / "meta.json").exists():
        return None
    meta = json.loads((d / "meta.json").read_text())
    return {"meta": meta, "dir": d}


def load_val_run(name):
    r = load_run(name)
    if r is None:
        return None
    pu = pd.read_parquet(r["dir"] / "per_user.parquet")
    return {**r, "per_user": pu, "top": np.load(r["dir"] / "top200.npy").astype(np.int64),
            "users": np.load(r["dir"] / "eval_users.npy"), "coverage": r["meta"]["coverage@10"]}


# ---------------------------------------------------------------- A. validation
base = {"random": RandomRec(), "global_pop": GlobalPop(), "content_384c": ContentRetrieval(emb_c)}
for m in base.values():
    m.fit(train, catalog)
knn_state = knn_neighbors(train, catalog, [SH], K)
base["item_knn"] = ItemKNN().use(knn_state, SH, K)
res = {n: evaluate(m, train, val, catalog) for n, m in base.items()}
stages.done("baselines on validation")
users = res["global_pop"].users
entries = {n: {"per_user": r.per_user, "coverage": r.coverage, "top": r.top} for n, r in res.items()}
LABEL = {"a_id_n20": "tt_id_n20", "a_content_n20": "tt_content_n20", "a_both_n20": "tt_both_n20",
         "a_both_n50": "tt_both_n50", "a_both_n20_nologq": "tt_both_n20_nologq"}
for run, lab in LABEL.items():
    r = load_val_run(run)
    if r is not None:
        assert (r["users"] == users).all(), "evaluation users differ between runs"
        entries[lab] = {"per_user": r["per_user"], "coverage": r["coverage"], "top": r["top"]}
tables = {n: summarize(e["per_user"], e["coverage"]) for n, e in entries.items()}


def fmt(view):
    cols = [f"{view}_recall@200", f"{view}_recall@50", f"{view}_ndcg@10", f"{view}_recall@10"]
    rows = {n: {c.split("_", 1)[1]: f"{t.loc[c, 'mean']:.4f} [{t.loc[c, 'lo']:.3f}-{t.loc[c, 'hi']:.3f}]" for c in cols}
            for n, t in tables.items()}
    df = pd.DataFrame(rows).T
    df["users"] = [int(t.loc[cols[0], "n_users"]) for t in tables.values()]
    if view == "all":
        df["coverage@10"] = [f"{t.loc['coverage@10', 'mean']:.4f}" for t in tables.values()]
    return df


lines += ["== A. PER-USER VALIDATION (fit on train; test locked); selection metric Recall@200, tie-break Recall@50 ==", ""]
for view, label in [("all", "ALL validation likes"), ("warm", "WARM likes (item seen in train)"),
                    ("cold", "COLD likes (item never seen in train)")]:
    lines += [f"--- {label}: mean [95% CI over users] ---", fmt(view).to_string(), ""]

pu = {n: e["per_user"] for n, e in entries.items()}
pairs = [("tt_both_n20", "item_knn"), ("tt_both_n20", "global_pop"), ("tt_both_n20", "content_384c"),
         ("tt_both_n20", "tt_id_n20"), ("tt_both_n20", "tt_content_n20"), ("tt_both_n50", "tt_both_n20"),
         ("tt_both_n20", "tt_both_n20_nologq"), ("item_knn", "global_pop")]
lines += ["--- paired differences over the same users: mean [95% CI]; * = interval excludes 0 ---"]
for x, y in pairs:
    if x in pu and y in pu:
        for col in ["all_recall@200", "all_recall@50", "all_ndcg@10", "warm_recall@200", "cold_recall@200"]:
            m, lo, hi = paired_diff(pu[x][col], pu[y][col])
            lines.append(f"{x + ' - ' + y:42s} {col:16s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
lines.append("")

lines += ["--- training runs ---"]
for run in list(LABEL) + ["g_id_n20", "g_content_n20", "g_both_n20", "g_both_n20_nologq"]:
    r = load_run(run)
    if r:
        m = r["meta"]
        lines.append(f"{run:20s} variant={m['variant']:8s} N={m['history']:2d} logq={m['logq']!s:5s} split={m['split']:6s} "
                     f"best_epoch={m['best_epoch']:2d} epochs={m['epochs_trained']:2d} minutes={m['train_minutes']:.1f} pairs={m['pairs']}")
r = load_run("a_both_n20")
if r:
    lines += ["", "--- a_both_n20 learning curve (validation subsample) ---"] + [
        f"epoch {h['epoch']:2d} loss {h['loss']:.3f}  Recall@200 {h.get('recall@200', float('nan')):.4f}  "
        f"Recall@50 {h.get('recall@50', float('nan')):.4f}  NDCG@10 {h.get('ndcg@10', float('nan')):.4f}" for h in r["meta"]["history_log"]]
    mm = r["meta"]
    lines += ["", f"--- users without liked items in train (popularity fallback) ---",
              f"all {mm['train_users']} users: {mm['train_users_no_likes']} have none; "
              f"of the {mm['val_eval_users']} validation users with >= 1 validation like: {mm['val_eval_users_no_train_likes']}"]
lines.append("")
stages.done("report A")

# ---------------------------------------------------------------- B. candidate union
both = [(n, load_run(a)) for n, a in [("tt_both_n20", "a_both_n20"), ("tt_both_n50", "a_both_n50")] if load_run(a)]
if both:
    sel = lambda m: max(m["history_log"], key=lambda h: (round(h.get("recall@200", 0), 4), h.get("recall@50", 0)))
    pick = both[0]
    for cand in both[1:]:
        if better(sel(cand[1]["meta"]), sel(pick[1]["meta"])):
            pick = cand
    tt_name = pick[0]
    tt_top, knn_top = entries[tt_name]["top"], entries["item_knn"]["top"]
    fill = tt_top if pu[tt_name]["all_recall@200"].mean() >= pu["item_knn"]["all_recall@200"].mean() else knn_top
    uni = union_top(tt_top, knn_top, 100, 200, fill)
    pu_union = evaluate_lists(uni, train, val, catalog)
    pu["union"] = pu_union
    entries["union"] = {"per_user": pu_union, "coverage": float("nan")}
    lines += [f"== B. CANDIDATE UNION: Recall@200 of {tt_name} alone, item_knn alone, and 100 from each (deduplicated, "
              f"filled to 200 from the better list = {'two-tower' if fill is tt_top else 'item_knn'}) ==", ""]
    rows = {}
    for lab, d in [(tt_name, pu[tt_name]), ("item_knn", pu["item_knn"]), ("union_100+100", pu_union)]:
        t = summarize(d[[f"{v}_recall@200" for v in ("all", "warm", "cold")]], float("nan"), n_boot=1000)
        rows[lab] = {v: f"{t.loc[f'{v}_recall@200', 'mean']:.4f} [{t.loc[f'{v}_recall@200', 'lo']:.3f}-{t.loc[f'{v}_recall@200', 'hi']:.3f}] (n={int(t.loc[f'{v}_recall@200', 'n_users'])})"
                     for v in ("all", "warm", "cold")}
    lines += [pd.DataFrame(rows).T.to_string(), "", "--- paired: union vs each list alone ---"]
    for ref in (tt_name, "item_knn"):
        for v in ("all", "warm", "cold"):
            m, lo, hi = paired_diff(pu_union[f"{v}_recall@200"], pu[ref][f"{v}_recall@200"])
            lines.append(f"union - {ref:12s} {v}_recall@200 {m:+.4f} [{lo:+.4f}, {hi:+.4f}]" + ("  *" if lo > 0 or hi < 0 else ""))
    lines.append("")
stages.done("report B")

# ---------------------------------------------------------------- C. new releases (global cutoff)
GL = {"tt_id": "g_id_n20", "tt_content": "g_content_n20", "tt_both": "g_both_n20", "tt_both_nologq": "g_both_n20_nologq"}
if all(load_run(r) for r in GL.values()):
    train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
    tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
    sets = release_sets(train_g, val_g, content.release_dates(movies, links, tmdb), months=6)
    gm = {"random": RandomRec(), "global_pop": GlobalPop(), "content_384c": ContentRetrieval(emb_c)}
    for m in gm.values():
        m.fit(train_g, catalog)
    gm["item_knn"] = ItemKNN().use(knn_neighbors(train_g, catalog, [SH], K), SH, K)
    for lab, run in GL.items():
        d = load_run(run)["dir"]
        w = np.load(d / "user_tower.npz")
        m = TwoTowerRetrieval(np.load(d / "item_vecs.npy"), w["W"], w["b"], load_run(run)["meta"]["history"])
        m.fit(train_g, catalog)
        gm[lab] = m
    order = ["random", "global_pop", "item_knn", "content_384c", "tt_id", "tt_content", "tt_both", "tt_both_nologq"]
    nr = evaluate_new_release({n: RandomTies(gm[n], seed=1) for n in order}, train_g, val_g, catalog, sets)
    lines += ["== C. NEW-RELEASE EVALUATION (global cutoff; fit only on ratings <= cutoff; two-tower models retrained on that data "
              "with the epochs chosen by the per-user runs) ==",
              f"cutoff {sets['cutoff']}, window ends {sets['val_end']}; primary = TMDB release since cutoff-6mo "
              f"({len(sets['release_date'])} movies: {len(sets['release_cold'])} cold, {len(sets['release_warm'])} warm); "
              f"secondary = first rated after cutoff ({len(sets['first_rated'])})", ""]
    LAB = {"release_date": "PRIMARY (all)", "release_cold": "PRIMARY, COLD part (no train ratings)",
           "release_warm": "PRIMARY, WARM part (already rated in train)", "first_rated": "SECONDARY (first rated after cutoff)"}
    for d in DEFINITIONS:
        for mode in MODES:
            r = nr[(d, mode)]
            lines += [f"--- {LAB[d]} | {'full catalog' if mode == 'full' else 'new-items pool only'} | users={r['n_users']}, "
                      f"items={r['n_items']}, likes={r['n_likes']} ---",
                      format_table(r["tables"]).to_string() if r["tables"] else "(none)", ""]
stages.done("report C")

# ---------------------------------------------------------------- D. export best ID+content run
if both:
    bdir = load_run({"tt_both_n20": "a_both_n20", "tt_both_n50": "a_both_n50"}[tt_name])
    art_dir, w = bdir["dir"], np.load(bdir["dir"] / "user_tower.npz")
    V, N = np.load(art_dir / "item_vecs.npy"), bdir["meta"]["history"]
    r = TwoTowerRetrieval(V, w["W"], w["b"], N)
    r.fit(train, catalog)
    counts = np.bincount(pd.Index(catalog).get_indexer(train.loc[train["label"] == 1, "item_id"]), minlength=len(catalog))
    uid = np.array(sorted(r.u_index))
    stats = export_milvus(TT / "export", catalog, V, movies, counts, w["W"], w["b"], N, uid, r.U[[r.u_index[u] for u in uid]])
    lines += [f"== D. MILVUS-READY EXPORT ({tt_name}) -> {TT / 'export'} ==", str(stats),
              "files: item_vectors.parquet (item_id INT64 pk, vector FLOAT_VECTOR 64 + metadata), item_vectors.npy, "
              "milvus_schema.json, user_tower.npz, user_vectors.parquet", ""]
text = "\n".join(lines)
print(text)
(ROOT / "reports").mkdir(exist_ok=True)
(ROOT / "reports" / f"module4_two_tower_{cfg.run_name}.txt").write_text(text)
stages.done("write report")
