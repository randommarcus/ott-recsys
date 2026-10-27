"""Checkpoint 3 verification, with the READ-ONLY key only (ZILLIZ_TOKEN_READONLY). Writes nothing on purpose; one write is ATTEMPTED and must be refused.

    python scripts/verify_zilliz.py             # A1 + A2 + B
    python scripts/verify_zilliz.py --recall    # also C: the level study (levels 1, 3, 5, 7, 10) on 100 users
    python scripts/verify_zilliz.py --level 7   # run A1/A2 at an explicit search level (default: none sent = server default)
    python scripts/verify_zilliz.py --skip-write-test --recall   # the key is known to be write-capable (accepted 2026-10-27): skip B, run the study

A1  search through pymilvus: two-tower top-10 (the query item must come back first), content lane top-30 with filter is_cold == true (all must be cold),
    content vec64 top-5.
A2  the same searches through the exact REST client the Lambda function uses (serving/search.py::ZillizREST), and how many results agree with A1.
B   a harmless write: insert one dummy row into the scratch collection ott_write_probe, once through pymilvus and once through REST. Each must be
    REFUSED for a permission reason. ALLOWED means the key can write (wrong role); INCONCLUSIVE means some other error (proves nothing). Afterwards the
    probe must still hold only its one original row.
C   (--recall) the level study. For the same 100 sampled users and each search level (Zilliz AUTOINDEX `level`, 1-10): (i) recall of the live candidate
    lists (built through REST exactly as the Lambda does) against the exact precomputed lists, main lane and new-release lane; (ii) median and 90th
    percentile time of one REST search measured FROM THIS LAPTOP (network included; NOT Lambda latency); (iii) what the user sees: the request handler's
    top-18 ranked items in live mode vs precomputed mode for the same user, clock and seed. Informational; it does not change the verdict.
Refuses to run if ZILLIZ_TOKEN_READONLY is blank or equal to the full-access ZILLIZ_TOKEN. Prints counts, yes/no and error TYPE names only: never the URI,
a token or an error message. Exit code 0 only if A1, A2 and B all pass."""
import argparse
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ott_recsys.config import Config
from ott_recsys.serving.search import SearchError, ZillizREST
from ott_recsys.serving.secrets import ZillizConfig
from ott_recsys.zilliz_collections import (CONTENT, PROBE, PROBE_ROW, TT, WRITE_TEST_ROW, default_client_factory, error_code, load_env,
                                           looks_like_permission_error, quiet, read_export)

OUT = Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42).out_dir


def hit_ids(result) -> list:
    """pymilvus search result for ONE query -> list of primary keys."""
    out = []
    for h in result[0]:
        ent = h.get("entity", {}) if hasattr(h, "get") else {}
        out.append(int(h.get("id", ent.get("item_id"))))
    return out


def search_params_pm(level=None) -> dict:
    """pymilvus search_params; `level` per the Zilliz "Tune recall rate" docs: search_params={"params": {"level": N}}."""
    sp = {"metric_type": "IP"}
    if level is not None:
        sp["params"] = {"level": int(level)}
    return sp


def search_pm(client, name, field, vec, limit, flt="", level=None):
    with quiet():
        res = client.search(collection_name=name, data=[np.asarray(vec, np.float32).tolist()], anns_field=field, limit=limit, filter=flt,
                            search_params=search_params_pm(level), output_fields=["item_id"])
    return hit_ids(res)


def error_words(problem) -> str:
    """The text of an error, used ONLY to look for permission words. It is never printed."""
    return " ".join(map(str, getattr(problem, "args", ()))) + " " + str(getattr(problem, "message", ""))


def write_attempt_pymilvus(client) -> str:
    try:
        with quiet():
            client.insert(collection_name=PROBE, data=[WRITE_TEST_ROW])
    except Exception as exc:
        if looks_like_permission_error(error_words(exc)):
            return "REFUSED (permission)"
        shown = error_code(exc)
        return f"INCONCLUSIVE (error type: {shown})"
    return "ALLOWED (UNEXPECTED: this key can write)"


def write_attempt_rest(endpoint: str, key: str, timeout: float = 10.0) -> str:
    body = json.dumps({"collectionName": PROBE, "data": [WRITE_TEST_ROW]}).encode()
    req = urllib.request.Request(endpoint.rstrip("/") + "/v2/vectordb/entities/insert", data=body, method="POST",
                                 headers={"Authorization": "Bearer " + key, "Content-Type": "application/json", "Accept": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read())
    except urllib.error.HTTPError as http:
        return "REFUSED (permission, HTTP 403)" if http.code == 403 else f"INCONCLUSIVE (HTTP {int(http.code)})"
    except Exception as exc:
        shown = type(exc).__name__
        return f"INCONCLUSIVE (error type: {shown})"
    code = payload.get("code", 0) if isinstance(payload, dict) else 0
    if code in (0, 200):
        return "ALLOWED (UNEXPECTED: this key can write)"
    if looks_like_permission_error(payload.get("message", "")):
        return f"REFUSED (permission, code {int(code)})"
    return f"INCONCLUSIVE (code {int(code)})"


class TimedCache:
    """Wraps the Lambda REST client: times every real search, and answers an identical repeated query from memory (the handler's live mode repeats
    exactly the queries of the recall step, so each user costs 2 searches per level, not 4)."""

    def __init__(self, inner):
        self.inner, self.cache, self.times = inner, {}, []

    def search(self, collection, field, vec, limit, flt=None):
        key = (collection, field, np.asarray(vec, np.float32).tobytes(), int(limit), flt)
        if key not in self.cache:
            t0 = time.perf_counter()
            self.cache[key] = self.inner.search(collection, field, vec, limit, flt)
            self.times.append(time.perf_counter() - t0)
        return self.cache[key]


def top18(resp) -> list:
    return [x["movie_id"] for x in resp["items"] if x["source"] == "ranker"][:18]


def level_study(endpoint: str, key: str, levels, n_users: int = 100, seed: int = 0, progress=print, every: int = 20):
    """Yields one result row per level as soon as that level is done; reports progress every `every` users with a rough time left."""
    from ott_recsys.serving import handler as H
    from ott_recsys.serving.bundle import Bundle
    from ott_recsys.serving.search import live_candidates
    from ott_recsys.serving.store import MemoryStore
    from ott_recsys.serving_offline import record_at
    b = Bundle(OUT / "serving" / "bundle")
    a = dict(np.load(OUT / "serving" / "users.npz"))
    sel = np.random.default_rng(seed).choice(np.flatnonzero(a["has_ttvec"]), n_users, replace=False)
    store = MemoryStore()
    recs = [record_at(a, int(i)) for i in sel]
    for rec in recs:
        store.put_user(rec)
    pre_app = H.App(b, store, store)
    pre = {rec["user_id"]: top18(H.handle(pre_app, "POST", "/recommend", {"user_id": rec["user_id"], "seed": 1, "record_impressions": False})[1]) for rec in recs}
    t_study, done_users, total_users = time.perf_counter(), 0, len(levels) * len(recs)
    for level in levels:
        searcher = TimedCache(ZillizREST(ZillizConfig(endpoint, key), timeout=10.0, level=level))
        main_r, lane_r, same_main, same_lane = [], [], 0, 0
        progress(f"    level {level}: searching ({len(recs)} users, about {2 * len(recs)} searches)", flush=True)
        for k, rec in enumerate(recs, 1):
            live, _ = live_candidates(b, rec, searcher)
            done_users += 1
            if k % every == 0 or k == len(recs):
                spent = time.perf_counter() - t_study
                left = spent / done_users * (total_users - done_users)
                progress(f"    level {level}: {k}/{len(recs)} users, {spent:.0f}s so far, about {left:.0f}s left in the study", flush=True)
            off = rec["cand"]
            for part, store_r in ((slice(0, 200), main_r), (slice(200, None), lane_r)):
                p_off, p_live = set(off[part][off[part] >= 0].tolist()), set(live[part][live[part] >= 0].tolist())
                if p_off:
                    store_r.append(len(p_off & p_live) / len(p_off))
            same_main += set(off[:200][off[:200] >= 0].tolist()) == set(live[:200][live[:200] >= 0].tolist())
            same_lane += set(off[200:][off[200:] >= 0].tolist()) == set(live[200:][live[200:] >= 0].tolist())
        live_app = H.App(b, store, store, searcher_factory=lambda s=searcher: s)
        ov, same_set, same_order, fallbacks = [], 0, 0, 0
        for rec in recs:
            status, resp = H.handle(live_app, "POST", "/recommend", {"user_id": rec["user_id"], "mode": "live", "seed": 1, "record_impressions": False})
            fallbacks += status != 200 or resp["diagnostics"]["fallback"] is not None
            lv, pv = top18(resp) if status == 200 else [], pre[rec["user_id"]]
            ov.append(len(set(lv) & set(pv)) / max(len(pv), 1))
            same_set += set(lv) == set(pv)
            same_order += lv == pv
        t = sorted(searcher.times)
        yield ({"level": level, "main_recall": float(np.mean(main_r)), "lane_recall": float(np.mean(lane_r)) if lane_r else float("nan"),
                     "identical_main": same_main, "identical_lane": same_lane, "top18_overlap": float(np.mean(ov)), "top18_same_set": same_set,
                     "top18_same_order": same_order, "fallbacks": fallbacks, "searches": len(t), "median_ms": 1000 * statistics.median(t),
                     "p90_ms": 1000 * t[int(0.9 * (len(t) - 1))]})


def main(argv=None, client_factory=None, out_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recall", action="store_true")
    ap.add_argument("--levels", default="1,3,5,7,10", help="search levels for the --recall study")
    ap.add_argument("--level", type=int, default=None, choices=range(1, 11), metavar="1-10", help="search level for A1/A2 (default: none sent)")
    ap.add_argument("--users", type=int, default=100)
    ap.add_argument("--skip-write-test", action="store_true",
                    help="skip B: the key is KNOWN to be write-capable (decision 2026-10-27, option a); the verdict then covers A1 and A2 only")
    args = ap.parse_args(argv)
    try:
        levels = [int(x) for x in args.levels.split(",")]
        assert all(1 <= x <= 10 for x in levels)
    except (ValueError, AssertionError):
        ap.error("--levels must be integers from 1 to 10, e.g. 1,3,5,7,10")
    load_env()
    endpoint = os.environ.get("ZILLIZ_URI", "").strip()
    ro, full = os.environ.get("ZILLIZ_TOKEN_READONLY", "").strip(), os.environ.get("ZILLIZ_TOKEN", "").strip()
    if not endpoint or not ro:
        print("refused: ZILLIZ_URI and ZILLIZ_TOKEN_READONLY must be set in .env")
        return 1
    if ro == full:
        print("refused: ZILLIZ_TOKEN_READONLY is the same as the full-access ZILLIZ_TOKEN; create a separate Data Viewer key")
        return 1
    out_dir = out_dir or OUT
    tt, ct = read_export(out_dir, TT), read_export(out_dir, CONTENT)
    cold = dict(zip(ct["item_id"].tolist(), ct["is_cold"].tolist()))
    try:
        with quiet():
            client = (client_factory or default_client_factory)(endpoint, ro)
    except Exception as exc:
        shown = error_code(exc)
        print(f"connection with the read-only key: no (error type: {shown})")
        return 1
    print("connection with the read-only key: yes" + (f" (A1/A2 at search level {args.level})" if args.level else " (A1/A2 at the server's default search level)"))
    ok = {}

    # A1 pymilvus searches
    q_tt, self_id, q_384, q_64 = tt["vector"][0], int(tt["item_id"][0]), ct["vec384"][0], ct["vec64"][0]
    try:
        a1_tt = search_pm(client, TT, "vector", q_tt, 10, level=args.level)
        a1_lane = search_pm(client, CONTENT, "vec384", q_384, 30, "is_cold == true", level=args.level)
        a1_64 = search_pm(client, CONTENT, "vec64", q_64, 5, level=args.level)
        ok["A1"] = len(a1_tt) == 10 and a1_tt[0] == self_id and len(a1_lane) == 30 and all(cold.get(i, False) for i in a1_lane) and len(a1_64) == 5
        print(f"A1 search (pymilvus): two-tower {len(a1_tt)} results, query item first: {'yes' if a1_tt and a1_tt[0] == self_id else 'no'}; "
              f"lane {len(a1_lane)} results, all cold: {'yes' if all(cold.get(i, False) for i in a1_lane) else 'no'}; vec64 {len(a1_64)} results  -> {'PASS' if ok['A1'] else 'FAIL'}")
    except Exception as exc:
        shown = error_code(exc)
        hint = "; the key may not have search rights" if looks_like_permission_error(error_words(exc)) else ""
        ok["A1"], a1_tt, a1_lane = False, [], []
        print(f"A1 search (pymilvus): FAIL (error type: {shown}){hint}")

    # A2 the Lambda's REST client
    rest = ZillizREST(ZillizConfig(endpoint, ro), timeout=10.0, level=args.level)
    try:
        r_tt, _ = rest.search(TT, "vector", q_tt, 10)
        r_lane, _ = rest.search(CONTENT, "vec384", q_384, 30, "is_cold == true")
        r_64, _ = rest.search(CONTENT, "vec64", q_64, 5)
        agree_tt, agree_lane = len(set(r_tt.tolist()) & set(a1_tt)), len(set(r_lane.tolist()) & set(a1_lane))
        ok["A2"] = len(r_tt) == 10 and len(r_lane) == 30 and len(r_64) == 5 and all(cold.get(int(i), False) for i in r_lane) and agree_tt >= 8
        print(f"A2 search (Lambda REST client): two-tower {len(r_tt)} results ({agree_tt}/10 same as A1); lane {len(r_lane)} results ({agree_lane}/30 same as A1); "
              f"vec64 {len(r_64)} results  -> {'PASS' if ok['A2'] else 'FAIL'}")
    except SearchError as failure:
        ok["A2"] = False
        print(f"A2 search (Lambda REST client): FAIL ({failure.args[0] if failure.args else 'SearchError'})")

    # B harmless write attempts with the read-only key (skipped when the key is known to be write-capable)
    if args.skip_write_test:
        print("B write test: SKIPPED (--skip-write-test: this key is known to be write-capable; accepted 2026-10-27, option a; nothing written)")
    else:
        w_pm, w_rest = write_attempt_pymilvus(client), write_attempt_rest(endpoint, ro)
        try:
            probe = search_pm(client, PROBE, "vector", WRITE_TEST_ROW["vector"], 10)
            unchanged = probe == [PROBE_ROW["item_id"]]
        except Exception as exc:
            shown = error_code(exc)
            unchanged, probe = None, []
            print(f"probe check: could not search ott_write_probe (error type: {shown})")
        ok["B"] = w_pm.startswith("REFUSED") and w_rest.startswith("REFUSED") and unchanged is True
        print(f"B write with the read-only key: pymilvus {w_pm}; REST {w_rest}; probe unchanged: "
              f"{'yes' if unchanged else 'NO (a test row was written)' if unchanged is False else 'unknown'}  -> {'PASS' if ok['B'] else 'FAIL'}")

    # C the level study (informational)
    if args.recall:
        print(f"C level study ({args.users} users): recall = share of the exact precomputed candidates that live search also returns; "
              "latency = one REST search measured from THIS LAPTOP to Zilliz (network included), NOT Lambda latency; "
              "top-18 = the request handler's 18 ranked items, live vs precomputed mode, same user, clock and seed")
        try:
            for r in level_study(endpoint, ro, levels, args.users):
                print(f"  level {r['level']:>2d}: main recall {r['main_recall']:.1%} ({r['identical_main']} identical), lane recall {r['lane_recall']:.1%} "
                      f"({r['identical_lane']} identical) | top-18 overlap {r['top18_overlap']:.1%}, same set {r['top18_same_set']}/{args.users}, "
                      f"same order {r['top18_same_order']}/{args.users} | laptop search median {r['median_ms']:.0f} ms, p90 {r['p90_ms']:.0f} ms "
                      f"({r['searches']} searches)" + (f" | {r['fallbacks']} fallbacks" if r["fallbacks"] else ""), flush=True)
        except SearchError as failure:
            print(f"C level study: stopped ({failure.args[0] if failure.args else 'SearchError'})")
    judged = ("A1", "A2") if args.skip_write_test else ("A1", "A2", "B")
    passed = all(ok.get(k) for k in judged)
    print(f"verdict: {'PASS' if passed else 'FAIL'} (" + ", ".join(f"{k} {'ok' if ok.get(k) else 'no'}" for k in judged)
          + ("; B skipped" if args.skip_write_test else "") + ")")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
