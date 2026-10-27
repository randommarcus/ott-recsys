"""Checkpoint 5: latency study of the deployed Lambda (eu-central-1). YOU run it, from the laptop, with the ott-dev profile.

    python scripts/latency_study.py                   # 200 users x 2 modes warm, 10 cold starts, connect baseline (about 5-10 minutes)
    python scripts/latency_study.py --cold 0          # skip the cold starts

Refuses to start unless reports/module9_latency_predictions.md is committed and unchanged (predictions first). Steps, all sequential:
  1. network baseline: 20 TCP+TLS connects from this laptop to the Function URL host (no credentials sent);
  2. warm-up: 5 requests per mode, discarded;
  3. warm run: for each sampled user, one precomputed and one live request through the IAM-signed Function URL, in random order; records client time,
     the server's own timings (server_total and its stages) and live-mode fallbacks. record_impressions is false: the bandit counts do not change;
  4. cold starts: a configuration change forces a new execution environment, then one direct invoke (alternating modes) reads Lambda's REPORT line
     (Init Duration, Duration, Max Memory Used); the temporary variable holds only a counter and is removed at the end.
Writes reports/module9_latency_results.json (numbers only: no URL, no account id) and prints a table plus measured vs predicted. Server time is the
headline; laptop numbers include the India-Frankfurt network and are NOT properties of the serving design. Prints no URL, key or error message."""
import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ott_recsys.config import Config
from ott_recsys.serving_client import ClientError, DirectInvoker, FunctionUrlClient, bootstrap_ci, connect_times_ms, pct

PRED = ROOT / "reports" / "module9_latency_predictions.md"
OUT = Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42).out_dir
STAGES = ("user_read", "search", "features", "ranker", "bandit")


def predictions_committed() -> dict:
    """The predictions block, but only if the file is committed and identical to HEAD (predictions must come before measurements)."""
    head = subprocess.run(["git", "show", "HEAD:reports/module9_latency_predictions.md"], cwd=ROOT, capture_output=True, text=True)
    if head.returncode != 0 or head.stdout != PRED.read_text():
        raise SystemExit("refused: reports/module9_latency_predictions.md must be committed and unchanged before measuring")
    block = PRED.read_text().split("```json", 1)[1].split("```", 1)[0]
    return json.loads(block)


def sample_users(n: int, seed: int) -> list:
    a = np.load(OUT / "serving" / "users.npz")
    pool = np.flatnonzero(a["has_ttvec"])
    return [int(a["user_id"][i]) for i in np.random.default_rng(seed).choice(pool, n, replace=False)]


def summary(xs) -> dict:
    lo50, hi50 = bootstrap_ci(xs, 50)
    lo95, hi95 = bootstrap_ci(xs, 95)
    return {"n": len(xs), "p50": pct(xs, 50), "p50_ci": [lo50, hi50], "p95": pct(xs, 95), "p95_ci": [lo95, hi95], "p99": pct(xs, 99), "max": max(xs) if xs else float("nan")}


def warm_run(client, users, seed: int = 0, progress=print) -> dict:
    rng = np.random.default_rng(seed + 1)
    res = {m: {"client_ms": [], "server_ms": [], "stages": {s: [] for s in STAGES}, "errors": 0, "fallbacks": 0} for m in ("precomputed", "live")}
    for k, uid in enumerate(users, 1):
        for mode in rng.permutation(["precomputed", "live"]):
            try:
                status, body, secs = client.call("POST", "/recommend", {"user_id": uid, "mode": str(mode), "seed": 1, "record_impressions": False})
            except ClientError:
                res[mode]["errors"] += 1
                continue
            if status != 200 or "timings_ms" not in body:
                res[mode]["errors"] += 1
                continue
            r = res[mode]
            r["client_ms"].append(secs * 1000)
            r["server_ms"].append(float(body["timings_ms"]["server_total"]))
            for s in STAGES:
                r["stages"][s].append(float(body["timings_ms"].get(s, 0.0)))
            r["fallbacks"] += body["diagnostics"].get("fallback") is not None
        if k % 25 == 0 or k == len(users):
            progress(f"  warm: {k}/{len(users)} users", flush=True)
    return res


def cold_run(invoker, users, n: int, progress=print, sleep=time.sleep) -> list:
    out = []
    try:
        for k in range(n):
            mode = "precomputed" if k % 2 == 0 else "live"
            invoker.force_cold(str(k + 1), sleep)
            status, body, rep = invoker.invoke("POST", "/recommend", {"user_id": users[k % len(users)], "mode": mode, "seed": 1, "record_impressions": False})
            out.append({"mode": mode, "status": status, "init_ms": rep.get("init_ms"), "duration_ms": rep.get("duration_ms"), "max_memory_mb": rep.get("max_memory_mb"),
                        "server_ms": (body.get("timings_ms") or {}).get("server_total"), "fallback": (body.get("diagnostics") or {}).get("fallback") is not None})
            progress(f"  cold start {k + 1}/{n} ({mode}): status {status}, init {rep.get('init_ms')} ms, duration {rep.get('duration_ms')} ms", flush=True)
    finally:
        invoker.clear_environment()
    return out


def verdicts(pred: dict, m: dict) -> list:
    rows = []
    for key, value, what in (("L1", m.get("warm_precomputed_server_p50"), "warm server, precomputed, median (ms)"),
                             ("L2", m.get("warm_live_server_p50"), "warm server, live, median (ms)"),
                             ("L3", m.get("warm_live_minus_precomputed_p50"), "live minus precomputed, medians (ms)"),
                             ("L4", m.get("warm_precomputed_client_p50"), "laptop end-to-end, precomputed, median (ms)"),
                             ("L5", m.get("cold_init_p50"), "cold start Init Duration, median (ms)"),
                             ("L6", m.get("max_memory_mb"), "max memory used (MB)"),
                             ("L7", m.get("connect_p50"), "laptop TCP+TLS connect, median (ms)"),
                             ("L8", m.get("live_fallbacks"), "live fallbacks during the warm run")):
        lo, hi = pred[key]
        hit = value is not None and not (isinstance(value, float) and np.isnan(value)) and lo <= value <= hi
        rows.append((key, what, lo, hi, value, "inside" if hit else ("not measured" if value is None else "MISS")))
    return rows


def main(argv=None, client=None, invoker=None, connect=connect_times_ms, progress=print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", default="eu-central-1", choices=["eu-central-1", "ap-south-1"])
    ap.add_argument("--profile", default="ott-dev")
    ap.add_argument("--users", type=int, default=200)
    ap.add_argument("--cold", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "module9_latency_results.json"))
    args = ap.parse_args(argv)
    pred = predictions_committed()
    if client is None or invoker is None:
        import boto3
        session = boto3.Session(profile_name=args.profile, region_name=args.region)
        client = client or FunctionUrlClient(session, args.region)
        invoker = invoker or DirectInvoker(session, args.region)
    users = sample_users(args.users, args.seed)
    progress(f"LATENCY STUDY, {args.region}: {len(users)} users x 2 modes warm, {args.cold} cold starts. Predictions: {PRED.name} "
             f"(sha256 {hashlib.sha256(PRED.read_bytes()).hexdigest()[:12]}, committed).", flush=True)

    conn_ms = connect(client.host_for_connect_test, 20)
    progress(f"  network baseline: {len(conn_ms)} TCP+TLS connects from this laptop, median {pct(conn_ms, 50):.0f} ms", flush=True)
    for mode in ("precomputed", "live"):
        for _ in range(5):
            try:
                client.call("POST", "/recommend", {"user_id": users[0], "mode": mode, "record_impressions": False})
            except ClientError:
                pass
    warm = warm_run(client, users, args.seed, progress)
    cold = cold_run(invoker, users, args.cold, progress) if args.cold else []

    res = {"region": args.region, "users": len(users), "connect_ms": summary(conn_ms), "cold": cold, "warm": {}}
    for mode, r in warm.items():
        res["warm"][mode] = {"server_ms": summary(r["server_ms"]), "client_ms": summary(r["client_ms"]), "errors": r["errors"], "fallbacks": r["fallbacks"],
                             "stage_median_ms": {s: pct(v, 50) for s, v in r["stages"].items()}}
    inits = [c["init_ms"] for c in cold if c["init_ms"] is not None]
    mems = [c["max_memory_mb"] for c in cold if c["max_memory_mb"] is not None]
    m = {"warm_precomputed_server_p50": res["warm"]["precomputed"]["server_ms"]["p50"], "warm_live_server_p50": res["warm"]["live"]["server_ms"]["p50"],
         "warm_live_minus_precomputed_p50": res["warm"]["live"]["server_ms"]["p50"] - res["warm"]["precomputed"]["server_ms"]["p50"],
         "warm_precomputed_client_p50": res["warm"]["precomputed"]["client_ms"]["p50"], "cold_init_p50": pct(inits, 50) if inits else None,
         "max_memory_mb": max(mems) if mems else None, "connect_p50": pct(conn_ms, 50), "live_fallbacks": res["warm"]["live"]["fallbacks"]}
    res["headline"] = m
    res["vs_predictions"] = [dict(zip(("id", "what", "low", "high", "measured", "verdict"), r)) for r in verdicts(pred, m)]
    Path(args.out).write_text(json.dumps(res, indent=1, default=float))

    print("\nRESULTS (server = the function's own timer; laptop = includes the India-Frankfurt network, NOT a property of the design)")
    for mode in ("precomputed", "live"):
        w = res["warm"][mode]
        s, c = w["server_ms"], w["client_ms"]
        print(f"  {mode:<11s} server p50 {s['p50']:.1f} ms [{s['p50_ci'][0]:.1f}-{s['p50_ci'][1]:.1f}], p95 {s['p95']:.1f} [{s['p95_ci'][0]:.1f}-{s['p95_ci'][1]:.1f}], "
              f"p99 {s['p99']:.1f} | laptop p50 {c['p50']:.0f} ms, p95 {c['p95']:.0f} | n {s['n']}, errors {w['errors']}, fallbacks {w['fallbacks']}")
        print("              stage medians (ms): " + ", ".join(f"{k} {v:.1f}" for k, v in w["stage_median_ms"].items()))
    if cold:
        print(f"  cold starts: {len(cold)}; Init Duration median {pct(inits, 50):.0f} ms (max {max(inits):.0f}); first-request Duration median "
              f"{pct([c['duration_ms'] for c in cold if c['duration_ms']], 50):.0f} ms; max memory {max(mems):.0f} MB of 1,024")
    print(f"  laptop TCP+TLS connect: median {pct(conn_ms, 50):.0f} ms")
    print("\nMEASURED vs PREDICTED (committed before the run)")
    for r in res["vs_predictions"]:
        val = "n/a" if r["measured"] is None else f"{r['measured']:.1f}"
        print(f"  {r['id']} {r['what']:<46s} predicted {r['low']}-{r['high']}: measured {val} -> {r['verdict']}")
    print(f"\nsaved: {Path(args.out).relative_to(ROOT) if Path(args.out).is_relative_to(ROOT) else Path(args.out).name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
