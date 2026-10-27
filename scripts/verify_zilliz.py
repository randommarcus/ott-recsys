"""Checkpoint 3 verification, with the READ-ONLY key only (ZILLIZ_TOKEN_READONLY). Writes nothing on purpose; one write is ATTEMPTED and must be refused.

    python scripts/verify_zilliz.py             # A1 + A2 + B
    python scripts/verify_zilliz.py --recall    # also C: live search vs the precomputed lists for 100 users

A1  search through pymilvus: two-tower top-10 (the query item must come back first), content lane top-30 with filter is_cold == true (all must be cold),
    content vec64 top-5.
A2  the same searches through the exact REST client the Lambda function uses (serving/search.py::ZillizREST), and how many results agree with A1.
B   a harmless write: insert one dummy row into the scratch collection ott_write_probe, once through pymilvus and once through REST. Each must be
    REFUSED for a permission reason. ALLOWED means the key can write (wrong role); INCONCLUSIVE means some other error (proves nothing). Afterwards the
    probe must still hold only its one original row.
C   (--recall) for 100 sampled users, build the 230-slot candidate lists through REST exactly as the Lambda does and compare them with the precomputed
    lists: average overlap of the main lane and of the new-release lane. Informational (AUTOINDEX search is approximate).
Refuses to run if ZILLIZ_TOKEN_READONLY is blank or equal to the full-access ZILLIZ_TOKEN. Prints counts, yes/no and error TYPE names only: never the URI,
a token or an error message. Exit code 0 only if A1, A2 and B all pass."""
import argparse
import json
import os
import sys
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


def search_pm(client, name, field, vec, limit, flt=""):
    with quiet():
        res = client.search(collection_name=name, data=[np.asarray(vec, np.float32).tolist()], anns_field=field, limit=limit, filter=flt,
                            search_params={"metric_type": "IP"}, output_fields=["item_id"])
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


def recall_check(rest, n_users: int = 100, seed: int = 0) -> dict:
    from ott_recsys.serving.bundle import Bundle
    from ott_recsys.serving.search import live_candidates
    from ott_recsys.serving_offline import record_at
    b = Bundle(OUT / "serving" / "bundle")
    a = dict(np.load(OUT / "serving" / "users.npz"))
    pool = np.flatnonzero(a["has_ttvec"])
    sel = np.random.default_rng(seed).choice(pool, n_users, replace=False)
    main_ov, lane_ov, same_main, same_lane, truncated = [], [], 0, 0, 0
    for i in sel:
        rec = record_at(a, int(i))
        live, info = live_candidates(b, rec, rest)
        off = rec["cand"]
        for part, store in ((slice(0, 200), main_ov), (slice(200, None), lane_ov)):
            p_off, p_live = set(off[part][off[part] >= 0].tolist()), set(live[part][live[part] >= 0].tolist())
            if p_off:
                store.append(len(p_off & p_live) / len(p_off))
        same_main += set(off[:200][off[:200] >= 0].tolist()) == set(live[:200][live[:200] >= 0].tolist())
        same_lane += set(off[200:][off[200:] >= 0].tolist()) == set(live[200:][live[200:] >= 0].tolist())
        truncated += info["truncated_main"]
    return {"users": n_users, "main_overlap": float(np.mean(main_ov)), "lane_overlap": float(np.mean(lane_ov)) if lane_ov else float("nan"),
            "identical_main": same_main, "identical_lane": same_lane, "truncated": truncated}


def main(argv=None, client_factory=None, out_dir=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--recall", action="store_true")
    args = ap.parse_args(argv)
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
    print("connection with the read-only key: yes")
    ok = {}

    # A1 pymilvus searches
    q_tt, self_id, q_384, q_64 = tt["vector"][0], int(tt["item_id"][0]), ct["vec384"][0], ct["vec64"][0]
    try:
        a1_tt = search_pm(client, TT, "vector", q_tt, 10)
        a1_lane = search_pm(client, CONTENT, "vec384", q_384, 30, "is_cold == true")
        a1_64 = search_pm(client, CONTENT, "vec64", q_64, 5)
        ok["A1"] = len(a1_tt) == 10 and a1_tt[0] == self_id and len(a1_lane) == 30 and all(cold.get(i, False) for i in a1_lane) and len(a1_64) == 5
        print(f"A1 search (pymilvus): two-tower {len(a1_tt)} results, query item first: {'yes' if a1_tt and a1_tt[0] == self_id else 'no'}; "
              f"lane {len(a1_lane)} results, all cold: {'yes' if all(cold.get(i, False) for i in a1_lane) else 'no'}; vec64 {len(a1_64)} results  -> {'PASS' if ok['A1'] else 'FAIL'}")
    except Exception as exc:
        shown = error_code(exc)
        hint = "; the key may not have search rights" if looks_like_permission_error(error_words(exc)) else ""
        ok["A1"], a1_tt, a1_lane = False, [], []
        print(f"A1 search (pymilvus): FAIL (error type: {shown}){hint}")

    # A2 the Lambda's REST client
    rest = ZillizREST(ZillizConfig(endpoint, ro), timeout=10.0)
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

    # B harmless write attempts with the read-only key
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

    # C optional recall check
    if args.recall:
        try:
            r = recall_check(rest)
            print(f"C recall (100 users, live REST vs precomputed lists): main overlap {r['main_overlap']:.1%} ({r['identical_main']} identical), "
                  f"lane overlap {r['lane_overlap']:.1%} ({r['identical_lane']} identical), truncated main lanes {r['truncated']}  (informational)")
        except SearchError as failure:
            print(f"C recall: stopped ({failure.args[0] if failure.args else 'SearchError'})")
    passed = all(ok.get(k) for k in ("A1", "A2", "B"))
    print(f"verdict: {'PASS' if passed else 'FAIL'} (A1 {'ok' if ok.get('A1') else 'no'}, A2 {'ok' if ok.get('A2') else 'no'}, B {'ok' if ok.get('B') else 'no'})")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
