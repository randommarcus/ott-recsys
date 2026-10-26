"""THE ONE-SHOT TEST EVALUATION (Module 8).

  real run   : python scripts/unlock_test.py --confirm UNLOCK-TEST-ONCE         (runs once, ever)
  rehearsal  : python scripts/unlock_test.py --rehearsal                         (fake test = validation data; the real
               test.parquet is made fatal to read; separate log and report; meaningless numbers - plumbing only)

This script TRAINS NOTHING and TUNES NOTHING. It loads the frozen artifacts written by scripts/prepare_test.py (checked against
their sha256 manifest), builds test-time features from train+validation histories and the first test timestamp of each user,
scores, evaluates and reports. The bandit simulation learns online inside the simulation - that is the thing being evaluated.
Guards (src/ott_recsys/unlockguard.py): the log is written BEFORE any test data is read; a second run is refused (also if the
log was deleted but is in git history); a crash rerun needs --after-crash and is allowed only if no test metric had been written
or printed; metrics are emitted only after everything has been computed. Nothing else in the repository may import this file
(tests/test_module8_static.py enforces it)."""
import sys

if __name__ != "__main__":
    raise ImportError("scripts/unlock_test.py is a one-shot script and must never be imported")

import argparse
import hashlib
import json
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np
import pandas as pd
import xgboost as xgb
from scipy import sparse

from ott_recsys import content
from ott_recsys import unlockguard as G
from ott_recsys.bandit import display_with_reserved, displayed_metrics, draw_coins, draw_truth, make_arms, mean_ci, oracle_expected, run_metrics, session_starts, simulate
from ott_recsys.baselines import ItemKNN
from ott_recsys.config import Config
from ott_recsys.context import genre_matrix, request_times
from ott_recsys.data import load_movies, load_user_split
from ott_recsys.eval import evaluate, paired_diff, summarize
from ott_recsys.frozen import FrozenContent, FrozenPop, FrozenTwoTower
from ott_recsys.newrelease import release_sets
from ott_recsys.rankfeat import BASE, CTX_SESSION, CTX_TIME, build_features
from ott_recsys.rerank import all_orders, hit_matrix, metrics_from_ranks, rank_positions
from ott_recsys.testprep import history_frame

CONFIRM = "UNLOCK-TEST-ONCE"
BANNER_SIM = "*** SIMULATION, NOT AN A/B TEST. Reward = 'liked given rated' (ignores click probability). ***"
UCB_C = 0.2                         # pre-registered primary exploration constant
EPS = 0.05                          # pre-registered epsilon (used as a comparison row only)
SEEDS, BANDIT_SEED = 30, 303
ARM_SETS = [("239 arms (>=20 ratings)", 20, 0.35), ("629 arms (>=5 ratings)", 5, 0.2), ("2031 arms (>=1 rating)", 1, 0.1)]  # (label, min ratings, Module 7b c*)
PRIMARY_ARM_SET = "629 arms (>=5 ratings)"
INTERPRETATION = [
    "INTERPRETATION RULE (pre-registered in reports/module8_protocol.md):",
    "  * Headline claims are P1-P5 only. Everything else is secondary or descriptive and is not a finding even if an interval excludes 0.",
    "  * A failed prediction is reported as it is. No re-running after seeing results; this script runs once.",
    "  * A later bug fix, if any, is shown as a labelled addition NEXT TO the original numbers, never replacing them.",
]
MISLEADING = [
    "WHAT COULD MAKE THESE TEST NUMBERS MISLEADING:",
    "  1. The test window is twice as long as validation (last 20% of each user's history vs 10%): users have about twice as many likes, so Recall@20's denominator roughly doubles",
    "     and absolute Recall falls while NDCG@10 may rise. ABSOLUTE test numbers are not comparable with validation numbers; only method-to-method comparisons within test are.",
    "  2. Frozen ID models (two-tower item vectors, NCF) are staler than they were on validation; this favours history-based features over ID features.",
    "  3. Feature shift: the rankers learned on features built from train-only histories; at test the histories include validation, so score distributions differ.",
    "  4. Movies first liked during validation are unknown to the frozen ID models (the lane pool covers them; the main lane cannot rank them).",
    "  5. Rating bursts continue across the validation/test boundary, which favours recency-aware methods and session features.",
    "  6. The per-user split still lets models learn from other users' later ratings (calendar leakage); the global-cutoff bandit window is cleaner.",
    "  7. The same users appear in validation and test (the rankers were fitted on their validation labels): this measures the next period for KNOWN users, not new users.",
    "  8. Many methods x views x metrics were computed in one shot: only P1-P5 are pre-registered; other 'significant' differences may be chance.",
    "  9. MovieLens ratings are not viewing; hours are UTC; device is simulated and not tested; TMDB release dates drive the release filter.",
    " 10. The bandit part is a simulation on thin test-window data with 'liked given rated' rewards and the exploration constant fixed in advance (c=0.2).",
]


def git_commit() -> str:
    return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()


def install_rehearsal_read_guard() -> None:
    """Rehearsal only: any attempt to read a real test.parquet is fatal."""
    orig = pd.read_parquet

    def guarded(path, *a, **k):
        if "test.parquet" in str(path):
            raise SystemExit(f"REHEARSAL BUG: tried to read {path}")
        return orig(path, *a, **k)
    pd.read_parquet = guarded


def fmt_ci(x, scale=1.0, nd=1):
    m, lo, hi = mean_ci(np.asarray(x) * scale)
    return f"{m:.{nd}f} [{lo:.{nd}f}-{hi:.{nd}f}]"


def pdiff(a, b):
    m, lo, hi = paired_diff(pd.Series(np.asarray(a)), pd.Series(np.asarray(b)))
    return m, lo, hi


def dstr(t, nd=4):
    return f"{t[0]:+.{nd}f} [{t[1]:+.{nd}f}, {t[2]:+.{nd}f}]" + ("  *" if t[1] > 0 or t[2] < 0 else "")


# ---------------------------------------------------------------------------------------------- per-user evaluation
def per_user_evaluation(cfg, rehearsal, history, test_df, catalog, movies, links, MD, FZ):
    cat_idx = pd.Index(catalog)
    hv, lists = np.load(MD / "history_vectors.npz"), np.load(MD / "lists.npz")
    emb_c = content.center_embeddings(content.load_embeddings(cfg.out_dir / "item_embeddings.npz", catalog, "emb384"))
    gm, gnames = genre_matrix(movies, catalog)
    tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
    rel = content.release_dates(movies, links, tmdb).reindex(catalog)
    rel_ts = np.array([x.timestamp() if pd.notna(x) else np.nan for x in rel])
    train_counts, hist_counts = hv["train_counts"], hv["hist_counts"]
    known_items = lists["ncf_item_known_cat"]

    truth_df = test_df[test_df["label"] == 1]
    users_t = np.sort(truth_df["user_id"].unique())
    truth = {u: cat_idx.get_indexer(g["item_id"].to_numpy()) for u, g in truth_df.groupby("user_id")}
    n_truth = np.array([len(truth[u]) for u in users_t])
    t_req = request_times(test_df, users_t)                      # first test interaction: its TIMESTAMP only
    rows = np.searchsorted(lists["users"], users_t)
    assert (lists["users"][rows] == users_t).all(), "a test user has no prepared candidate list"

    args = dict(n_main=200, catalog=catalog, genre_m=gm, genre_names=gnames, emb_c=emb_c, rel_ts=rel_ts, ncf_item_known_cat=known_items)
    out_f = build_features(history, users_t, t_req, cand=lists["cand"][rows].astype(np.int64), tt_score=lists["tt_score"][rows].astype(float),
                           ncf_logit=lists["ncf_fold"][rows].astype(float), ncf_user_known=lists["ncf_fold_user_known"][rows], **args)
    out_s = build_features(history, users_t, t_req, cand=lists["cand"][rows].astype(np.int64), tt_score=lists["tt_score"][rows].astype(float),
                           ncf_logit=lists["ncf_stale"][rows].astype(float), ncf_user_known=lists["ncf_stale_user_known"][rows], **args)
    assert (out_f["cand_f"] == out_s["cand_f"]).all() and (out_f["row_user"] == out_s["row_user"]).all()
    cand_f = out_f["cand_f"].astype(np.int64)
    U, W = cand_f.shape
    valid = cand_f >= 0
    hit = hit_matrix(cand_f, [truth[u] for u in users_t])

    cols = out_f["cols"]
    CI = {c: i for i, c in enumerate(cols)}
    f_base = [CI[c] for c in BASE + [c for c in cols if c.startswith("g_")]]
    VAR = {"M0": f_base, "M1": f_base + [CI[c] for c in CTX_TIME], "M2": f_base + [CI[c] for c in CTX_TIME] + [CI[c] for c in CTX_SESSION]}
    scores = {}
    for name, idx in VAR.items():
        bst = xgb.Booster()
        bst.load_model(str(FZ / f"final_{name}.json"))
        names = [cols[i] for i in idx]
        for tag, out in (("", out_f), ("_stale", out_s)):
            if tag == "_stale" and name != "M2":
                continue
            S = np.full((U, W), np.nan)
            S[out["row_user"].astype(np.int64), out["row_slot"].astype(np.int64)] = bst.predict(xgb.DMatrix(out["X"][:, idx], feature_names=names))
            scores[name + tag] = S

    ranks = all_orders(cand_f, hit, hist_counts, {"foldin": out_f["ncf_f"], "stale": out_s["ncf_f"]}, seed=0)
    rand = np.random.default_rng(0).random((U, W))
    for name, S in scores.items():
        ranks[name] = rank_positions(valid, S, rand)
    M = {m: metrics_from_ranks(r, hit, n_truth, users_t) for m, r in ranks.items()}

    # ---- retrieval over the whole catalog (frozen two-tower, frozen kNN neighbours, content, popularity)
    V = np.load(cfg.out_dir / "two_tower" / "a_id_n20" / "item_vecs.npy")
    nbr = np.load(FZ / "knn_neighbors.npz")
    Xh = sparse.load_npz(MD / "knn_history.npz")
    g = train_counts
    state = {"X": Xh, "u_index": {u: i for i, u in enumerate(hv["users_all"])}, "neighbors": {0.0: (nbr["idx"], nbr["sim"])},
             "k_max": nbr["idx"].shape[1], "tiebreak": 1e-6 * g / max(g.max(), 1.0)}
    retrievers = {"two-tower (ID-only)": FrozenTwoTower(V, hv["tt_ids"], hv["tt_U"], hist_counts), "item-kNN": ItemKNN().use(state, 0.0, 500),
                  "content (centered SBERT)": FrozenContent(emb_c, hv["ct_ids"], hv["ct_P"], hist_counts), "popularity": FrozenPop(hist_counts)}
    R = {n: evaluate(m, history, test_df, catalog, ks=(10, 50, 200)).per_user for n, m in retrievers.items()}

    # ---- cost of reserving slots (main lane ranking by the M2 ranker with fold-in NCF)
    S2 = np.where(out_f["lane_f"], -np.inf, np.where(np.isnan(scores["M2"]), -np.inf, scores["M2"]))
    main_hits = np.take_along_axis(hit, np.argsort(-S2, axis=1, kind="stable")[:, :25], axis=1)
    base_nd, base_rc = displayed_metrics(main_hits[:, :20], n_truth)
    slot = {}
    for lab, pos in [("2 reserved at the END (19, 20)", [19, 20]), ("3 reserved at the END (18, 19, 20)", [18, 19, 20]), ("2 reserved INSIDE the top 10 (5, 10)", [5, 10])]:
        nd, rc = displayed_metrics(display_with_reserved(main_hits, pos)[0], n_truth)
        slot[lab] = (pdiff(nd, base_nd), pdiff(rc, base_rc))
    return {"users": users_t, "n_truth": n_truth, "M": M, "R": R, "slot": slot, "out_f": out_f, "out_s": out_s, "hit": hit, "valid": valid,
            "lane_f": out_f["lane_f"], "cand_f": cand_f, "t_req": t_req, "scores": scores, "ncf_stale_share_nan": float(np.isnan(out_s["ncf_f"][valid]).mean()),
            "ncf_fold_share_nan": float(np.isnan(out_f["ncf_f"][valid]).mean()), "likes_per_user": float(n_truth.mean())}


# ---------------------------------------------------------------------------------------------- bandit simulation
def bandit_evaluation(cfg, history_g, test_g, catalog, movies, links):
    tmdb = content.load_tmdb_cache(ROOT / "data" / "raw" / "tmdb", links["tmdbId"])
    rel = content.release_dates(movies, links, tmdb)
    sets = release_sets(history_g, test_g, rel, months=6)
    arm_ids = np.asarray(sets["release_cold"])
    rel_ts = np.array([rel.get(i, pd.NaT).timestamp() if pd.notna(rel.get(i, pd.NaT)) else np.nan for i in arm_ids])
    tw = test_g[test_g["item_id"].isin(set(arm_ids))]
    n_arm = tw.groupby("item_id").size().reindex(arm_ids).fillna(0).to_numpy()
    likes = tw.groupby("item_id")["label"].sum().reindex(arm_ids).fillna(0).to_numpy()
    grate = float(tw["label"].mean())
    req_t = session_starts(test_g)
    out = {"n_arms_total": int(len(arm_ids)), "requests": int(len(req_t)), "global_rate": grate, "cutoff": sets["cutoff"], "val_end": sets["val_end"], "sets": {}}
    for label, minr, c_star in ARM_SETS:
        arms = make_arms(arm_ids, rel_ts, n_arm, likes, np.zeros(len(arm_ids)), grate, 5.0, minr)
        strategies = [("random", "random", {}), ("greedy", "greedy", {}), (f"eps-greedy {EPS}", "eps_greedy", {"eps": EPS}), ("UCB1 (c = sqrt 2)", "ucb1", {}),
                      (f"UCB c = {UCB_C} (pre-registered)", "ucb_c", {"c": UCB_C}), ("UCB1-Tuned", "ucb_tuned", {}), ("Thompson (Beta)", "thompson", {})]
        if c_star != UCB_C:
            strategies.append((f"UCB c* = {c_star} (Module 7b choice for this set; sensitivity)", "ucb_c", {"c": c_star}))
        p, coins = draw_truth(arms, SEEDS, BANDIT_SEED), draw_coins(len(req_t), 2, SEEDS, BANDIT_SEED)
        orc = oracle_expected(arms, req_t, 2, 365, p)
        res = {}
        for name, strat, kw in strategies:
            r = simulate(arms, req_t, strat, 2, 365, SEEDS, BANDIT_SEED, p=p, coins=coins, **kw)
            res[name] = (run_metrics(r, orc), r)
        out["sets"][label] = {"arms": len(arms["a"]), "oracle": 1000 * orc.sum(0) / res["random"][0]["impressions"], "res": res}
    return out


# ---------------------------------------------------------------------------------------------- report
def render(rehearsal, info, pu, bd, facts):
    L = []
    M, R = pu["M"], pu["R"]
    hdr = "REHEARSAL REPORT - fake test set = VALIDATION data, history = train only; the ranker was fitted on these very labels, so ALL NUMBERS BELOW ARE IN-SAMPLE AND MEANINGLESS. Plumbing check only." if rehearsal \
        else "TEST RESULTS - single, pre-registered run"
    L += [hdr, f"run info: {json.dumps(info, default=str)}", ""] + INTERPRETATION + [""]
    p1, p2, p3 = (pdiff(M["M2"]["e2e_ndcg10"], M["tt"]["e2e_ndcg10"]), pdiff(M["M0"]["e2e_ndcg10"], M["tt"]["e2e_ndcg10"]), pdiff(M["M2"]["e2e_ndcg10"], M["M0"]["e2e_ndcg10"]))
    p4 = pdiff(R["two-tower (ID-only)"]["all_recall@200"], R["item-kNN"]["all_recall@200"])
    prim = bd["sets"][PRIMARY_ARM_SET]["res"]
    ucb = prim[f"UCB c = {UCB_C} (pre-registered)"][0]
    def lk(m): return 1000 * m["likes"] / m["impressions"]
    p5a, p5b = mean_ci(lk(ucb) - lk(prim["UCB1 (c = sqrt 2)"][0])), mean_ci(lk(ucb) - lk(prim["Thompson (Beta)"][0]))
    star = lambda t: "  *" if t[1] > 0 or t[2] < 0 else ""
    L += ["=" * 110, "HEADLINE RESULTS (P1-P5, pre-registered)", "=" * 110,
          f"P1  M2 ranker (fold-in NCF) - two-tower order, end-to-end NDCG@10:   {dstr(p1)}",
          f"P2  M0 ranker (fold-in NCF) - two-tower order, end-to-end NDCG@10:   {dstr(p2)}",
          f"P3  M2 - M0 (context gain), end-to-end NDCG@10:                      {dstr(p3)}",
          f"P4  two-tower - item-kNN, Recall@200 (full catalog):                 {dstr(p4)}",
          f"P5a SIMULATION, {PRIMARY_ARM_SET}: UCB c={UCB_C} - UCB1, likes per 1,000 impressions: {p5a[0]:+.1f} [{p5a[1]:+.1f}, {p5a[2]:+.1f}]" + ("  *" if p5a[1] > 0 or p5a[2] < 0 else ""),
          f"P5b SIMULATION, {PRIMARY_ARM_SET}: UCB c={UCB_C} - Thompson, likes per 1,000 impressions: {p5b[0]:+.1f} [{p5b[1]:+.1f}, {p5b[2]:+.1f}]" + ("  *" if p5b[1] > 0 or p5b[2] < 0 else ""),
          "(* = 95% interval excludes 0.)", ""]
    L += [f"test users with >= 1 liked item: {len(pu['users'])}; mean likes per user {pu['likes_per_user']:.1f}; candidate slots with NCF = NaN: fold-in {pu['ncf_fold_share_nan']:.1%}, stale {pu['ncf_stale_share_nan']:.1%}", ""]
    C = ["e2e_ndcg10", "e2e_recall20", "rr_ndcg10", "rr_recall20"]
    order = ["random", "pop", "tt", "ncf_foldin", "ncf_stale", "blend_foldin", "blend_stale", "M0", "M1", "M2", "M2_stale", "oracle"]
    names = {"tt": "tt (two-tower's own order)", "pop": "pop (popularity order, train+val)", "ncf_foldin": "NCF order, fold-in (primary)", "ncf_stale": "NCF order, stale (sensitivity)",
             "blend_foldin": "50/50 blend tt+NCF fold-in", "blend_stale": "50/50 blend tt+NCF stale", "M0": "M0 ranker, no context (fold-in NCF)", "M1": "M1 + hour/day (fold-in)",
             "M2": "M2 + session (fold-in NCF) [primary]", "M2_stale": "M2 with STALE NCF features (sensitivity)"}
    rows = {}
    for m in order:
        t = summarize(M[m][C], float("nan"), n_boot=1000)
        rows[names.get(m, m)] = {c: f"{t.loc[c, 'mean']:.4f} [{t.loc[c, 'lo']:.3f}-{t.loc[c, 'hi']:.3f}]" for c in C}
    L += ["== RE-RANKING THE 230-SLOT LISTS (all test users; SECONDARY): mean [95% CI] ==", pd.DataFrame(rows).T.to_string(), ""]
    L += ["== SECONDARY paired differences (end-to-end NDCG@10 / Recall@20) =="]
    for a, b in [("M2", "M2_stale"), ("ncf_foldin", "ncf_stale"), ("M2", "blend_foldin"), ("M2", "pop"), ("M1", "M0"), ("M2", "M1")]:
        L.append(f"  {a + ' - ' + b:26s} NDCG@10 {dstr(pdiff(M[a]['e2e_ndcg10'], M[b]['e2e_ndcg10']))}   Recall@20 {dstr(pdiff(M[a]['e2e_recall20'], M[b]['e2e_recall20']))}")
    L += ["", "== RETRIEVAL OVER THE WHOLE CATALOG (SECONDARY except P4): mean [95% CI] =="]
    cols = ["all_recall@10", "all_recall@50", "all_recall@200", "all_ndcg@10", "warm_recall@200", "cold_recall@200"]
    L += [pd.DataFrame({n: {c: f"{summarize(r[[c]], float('nan'), n_boot=1000).loc[c, 'mean']:.4f}" for c in cols} | {"users": len(r)} for n, r in R.items()}).T.to_string(), ""]
    L += ["== COST OF RESERVING SLOTS (M2 main-lane ranking; paired difference vs no reservation) =="]
    for lab, (nd, rc) in pu["slot"].items():
        L.append(f"  {lab:40s} NDCG@10 {dstr(nd)}   Recall@20 {dstr(rc)}")
    L += ["", BANNER_SIM, f"== BANDIT SIMULATION on the {'FAKE (validation)' if rehearsal else 'global-cutoff TEST'} window: {bd['n_arms_total']} new-release arms, {bd['requests']} real session starts "
          f"({pd.to_datetime(bd['cutoff'])} to {pd.to_datetime(bd['val_end'])}); k=2 slots, 365-day shelf, {SEEDS} paired seeds; global like rate {bd['global_rate']:.3f}; strategies frozen in advance =="]
    for label, d in bd["sets"].items():
        res = d["res"]
        rnd = res["random"][0]["regret"]
        rows = {}
        for n, (m, _) in res.items():
            rows[n] = {"likes per 1,000 impressions": fmt_ci(lk(m)), "regret as % of random's": fmt_ci(100 * m["regret"] / rnd), "exposure Gini": fmt_ci(m["gini"], nd=2),
                       "impressions on true top-10% arms": fmt_ci(100 * m["top10pct_share"]) + "%", "arms never shown": fmt_ci(100 * m["never_shown"]) + "%"}
        rows["oracle (expected)"] = {"likes per 1,000 impressions": fmt_ci(d["oracle"])}
        L += ["", BANNER_SIM, f"-- {label} ({d['arms']} arms) --", pd.DataFrame(rows).T.to_string()]
    L += [""] + MISLEADING
    if rehearsal:
        L += ["", "REHEARSAL CHECKS:"] + facts
    return "\n".join(L)


# ---------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rehearsal", action="store_true")
    ap.add_argument("--rehearsal-reset", action="store_true", help="rehearsal only: delete the rehearsal log")
    ap.add_argument("--confirm", default="")
    ap.add_argument("--after-crash", action="store_true")
    ap.add_argument("--rehearsal-crash-at", choices=["start", "after-flag"], default=None, help="rehearsal only: simulate a crash to exercise the crash rule")
    args = ap.parse_args()
    rehearsal = args.rehearsal
    log_rel = G.REHEARSAL_LOG if rehearsal else G.LOG_NAME
    log_path = ROOT / log_rel
    if args.rehearsal_reset:
        if not rehearsal:
            sys.exit("--rehearsal-reset only works together with --rehearsal")
        log_path.unlink(missing_ok=True)
        print("rehearsal log removed")
        return
    if args.rehearsal_crash_at and not rehearsal:
        sys.exit("REFUSED: --rehearsal-crash-at only works together with --rehearsal")
    if not rehearsal and args.confirm != CONFIRM:
        sys.exit(f"REFUSED: the real run needs --confirm {CONFIRM}")
    try:
        prev = G.check_not_spent(ROOT, log_rel, args.after_crash, check_history=not rehearsal)
        warnings = []
        try:
            prereg = G.preregistration(ROOT)
            G.clean_tree(ROOT)
        except G.UnlockRefused as e:
            if not rehearsal:
                raise
            prereg, warnings = {}, [f"(rehearsal only) pre-flight would refuse a real run: {e}"]
        cfg = Config()
        MD = cfg.out_dir / "test_prep" / ("rehearsal" if rehearsal else "real")
        FZ = cfg.out_dir / "test_prep" / "frozen"
        manifest = G.verify_manifest(MD / "manifest.json")
    except G.UnlockRefused as e:
        sys.exit(f"REFUSED: {e}")

    src_hashes = G.hash_files(sorted((ROOT / "src" / "ott_recsys").glob("*.py")) + [ROOT / "scripts" / "unlock_test.py", ROOT / "scripts" / "prepare_test.py"])
    info = {"mode": "rehearsal" if rehearsal else "real", "git_commit": git_commit(), "preregistration_commits": prereg, "manifest_files": len(manifest["artifacts"]),
            "source_sha256": hashlib.sha256(json.dumps(src_hashes, sort_keys=True).encode()).hexdigest()}
    started = G.write_start_log(log_path, info, prev)          # <<< the log exists BEFORE any test data is read
    t0 = time.time()
    if args.rehearsal_crash_at == "start":
        raise RuntimeError("simulated crash right after the log was written (rehearsal)")

    # ---------------- test access begins
    movies, links = load_movies(cfg), pd.read_parquet(cfg.out_dir / "links.parquet")
    catalog = np.sort(movies["movieId"].to_numpy())
    split = load_user_split(cfg)
    train_g, val_g = pd.read_parquet(cfg.out_dir / "train.parquet"), pd.read_parquet(cfg.out_dir / "val.parquet")
    if rehearsal:
        install_rehearsal_read_guard()
        test_df, history_g, test_g = split.val, train_g, val_g
    else:
        split.unlock_test()
        test_df = split.test
        history_g, test_g = pd.concat([train_g, val_g], ignore_index=True), pd.read_parquet(cfg.out_dir / "test.parquet")
    history = history_frame(split.train, split.val, rehearsal)
    pu = per_user_evaluation(cfg, rehearsal, history, test_df, catalog, movies, links, MD, FZ)
    bd = bandit_evaluation(cfg, history_g, test_g, catalog, movies, links)

    facts = list(warnings)
    if rehearsal:
        z = np.load(cfg.out_dir / "ranker" / "features.npz")        # Module 6 validation features (stale NCF, train-only histories)
        same_users = np.array_equal(z["users"], pu["users"])
        Xo, Xn = z["X"], pu["out_s"]["X"]
        if same_users and Xo.shape == Xn.shape:
            exact_rows = float((~(~((Xo == Xn) | (np.isnan(Xo) & np.isnan(Xn)))).any(1)).mean())
            bad_share = float((~np.isclose(Xo, Xn, equal_nan=True, rtol=1e-4, atol=1e-5)).any(1).mean())
            facts.append(f"rebuilt stale-NCF feature matrix vs Module 6 features.npz: same users {same_users}, shape {Xo.shape}; rows bit-identical {exact_rows:.2%} "
                         f"(the NCF logit differs in its last float32 bits because the two runs scored different sets of pairs, so BLAS chunking differs); "
                         f"rows differing beyond rtol 1e-4 / atol 1e-5: {bad_share:.4%} (budget 0.1%; these are users whose tied items are ordered differently)")
            assert bad_share < 0.001, f"rebuilt features differ from Module 6 features in {bad_share:.3%} of rows"
        else:
            facts.append(f"rebuilt features vs Module 6 features.npz: same users {same_users}, shapes {Xn.shape} vs {Xo.shape}")
            assert False, "rehearsal feature matrix does not have the Module 6 shape"
        facts.append("exercised: log written before data access, manifest verification, request times from fake-test timestamps only, build_features (fold-in and stale), "
                     "frozen ranker loading/prediction, rank orders, retrieval scorers (two-tower, kNN, content, popularity), slot-cost, bandit simulation, report rendering, "
                     "metrics-emitted flag, done flag; the real test.parquet read guard stayed silent.")
        facts.append(f"elapsed {time.time() - t0:.0f} s")
    text = render(rehearsal, {**info, "attempts": started["attempts"]}, pu, bd, facts)
    out_path = ROOT / "reports" / ("rehearsal/module8_rehearsal_report.txt" if rehearsal else "module8_test_results.txt")

    G.mark_metrics_emitted(log_path)                           # <<< from here on a crash means the run is spent
    if args.rehearsal_crash_at == "after-flag":
        raise RuntimeError("simulated crash after the metrics-emitted flag (rehearsal)")
    emit_report(text, out_path)
    G.mark_done(log_path, str(out_path.relative_to(ROOT)), hashlib.sha256(text.encode()).hexdigest())


def emit_report(text: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(text)
    print(text)


main()
