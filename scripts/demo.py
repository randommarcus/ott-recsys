"""Checkpoint 5: demo of the deployed recommender. YOU run it, from the laptop, with the ott-dev profile. Writes reports/module9_demo.html.

    python scripts/demo.py                     # about 2-4 minutes
    python scripts/demo.py --bandit-requests 0 # skip the bandit simulation (it is the only part that changes stored counts)

Every list comes from a real request to the deployed function through its IAM-signed Function URL. Sections:
  A  M2 (the real ranker, trained on MovieLens): the same user mid-session (10 minutes after their last activity) vs after a 3-day break.
  B  M3b with the SIMULATED device, tv vs mobile, under the banner "Synthetic device effect, pipeline demonstration, not a finding"; plus the count of
     Drama-without-Comedy titles in the top 18 over many users (the planted effect: mobile requests had 60% of those likes removed in training).
  C  live (Zilliz vector search, level 5) vs precomputed (exact) candidates for the same user and clock.
  D  the UCB new-release slots under SIMULATED feedback: likes are drawn from hidden rates estimated from MovieLens ratings. SIMULATION, NOT AN A/B TEST.
The page contains no URL, key or account id. Request clock and device are SIMULATED and labelled as such."""
import argparse
import html
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import numpy as np

from ott_recsys.config import Config
from ott_recsys.serving import LABEL_BANDIT, LABEL_CLOCK, LABEL_DEVICE, LABEL_M3B
from ott_recsys.serving.bundle import Bundle
from ott_recsys.serving_client import ClientError, FunctionUrlClient

OUT = Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42).out_dir


def drama_no_comedy_mask(b) -> np.ndarray:
    g = b.genre_names
    return (b.genre_m[:, g.index("Drama")] == 1) & (b.genre_m[:, g.index("Comedy")] == 0)


def hidden_like_rates(b, pseudo: float = 5.0) -> np.ndarray:
    """SIMULATION ONLY: per-movie like rate from MovieLens train + validation ratings, shrunk toward the global rate with `pseudo` pseudo-ratings."""
    from ott_recsys.data import load_user_split
    import pandas as pd
    split = load_user_split(Config(dataset_name="ml-25m", sample_users=20000, sample_seed=42))  # the test portion is never read
    hist = pd.concat([split.train[["item_id", "label"]], split.val[["item_id", "label"]]])
    idx = b.cat_index(hist["item_id"].to_numpy())
    n = np.bincount(idx, minlength=len(b.catalog)).astype(float)
    likes = np.bincount(idx, weights=hist["label"].to_numpy().astype(float), minlength=len(b.catalog))
    g = likes.sum() / n.sum()
    return (likes + pseudo * g) / (n + pseudo)


def ask(client, body: dict):
    status, payload, _ = client.call("POST", "/recommend", body)
    if status != 200:
        raise ClientError(f"http-{status}")
    return payload


def ranked(resp) -> list:
    return [x for x in resp["items"] if x["source"] == "ranker"][:18]


def table(items, mark=None, compare=None) -> str:
    rows = []
    for x in items:
        cls = " class='mark'" if mark is not None and mark(x) else ""
        new = " <span class='new'>not in the other list</span>" if compare is not None and x["movie_id"] not in compare else ""
        src = "UCB new-release slot" if x["source"].startswith("bandit") else "ranker"
        rows.append(f"<tr{cls}><td>{x['position']}</td><td>{html.escape(x['title'])}{new}</td><td>{src}</td></tr>")
    return "<table><tr><th>#</th><th>title</th><th>source</th></tr>" + "".join(rows) + "</table>"


def banner(text: str, kind: str = "info") -> str:
    return f"<div class='banner {kind}'>{html.escape(text)}</div>"


CSS = """body{font-family:system-ui,sans-serif;margin:24px;max-width:1100px;color:#222}h2{margin-top:36px}.cols{display:flex;gap:24px;flex-wrap:wrap}
.cols>div{flex:1;min-width:300px}table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #ddd;padding:3px 6px;text-align:left}
.banner{padding:10px 14px;margin:10px 0;border-radius:6px;font-weight:600}.info{background:#eef4ff;border:1px solid #9bb7ef}
.synthetic{background:#ffe9e9;border:2px solid #d33;color:#a00}.sim{background:#fff6dd;border:2px solid #d9a400}.mark{background:#fff0f0}.new{font-size:11px;color:#a00}
footer{margin-top:40px;font-size:13px;color:#555}"""


def main(argv=None, client=None, bundle=None, rates=None, progress=print) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--region", default="eu-central-1", choices=["eu-central-1", "ap-south-1"])
    ap.add_argument("--profile", default="ott-dev")
    ap.add_argument("--users", type=int, default=40, help="users for the M3b aggregate")
    ap.add_argument("--bandit-requests", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=str(ROOT / "reports" / "module9_demo.html"))
    args = ap.parse_args(argv)
    if client is None:
        import boto3
        client = FunctionUrlClient(boto3.Session(profile_name=args.profile, region_name=args.region), args.region)
    b = bundle or Bundle(OUT / "serving" / "bundle")
    dnc = drama_no_comedy_mask(b)
    is_dnc = lambda x: bool(dnc[b.cat_index([x["movie_id"]])[0]])
    a = np.load(OUT / "serving" / "users.npz")
    rng = np.random.default_rng(args.seed)
    users = [int(a["user_id"][i]) for i in rng.choice(np.flatnonzero(a["has_ttvec"]), max(args.users, 5), replace=False)]
    u = users[0]
    parts, summary = [], {"user": "a sampled MovieLens user (id shown on the page)", "region": args.region}

    # A  M2: context changes the list
    progress("A: M2, mid-session vs after a 3-day break", flush=True)
    near = ask(client, {"user_id": u, "gap_minutes": 10, "seed": 1, "record_impressions": False})
    far = ask(client, {"user_id": u, "gap_minutes": 3 * 24 * 60, "seed": 1, "record_impressions": False})
    rn, rf = ranked(near), ranked(far)
    ids_n, ids_f = {x["movie_id"] for x in rn}, {x["movie_id"] for x in rf}
    summary["A_changed_titles"] = len(ids_n - ids_f)
    parts.append(f"<h2>A. The real model (M2): same user, different moment</h2>"
                 + banner("M2 is the ranker trained on MovieLens history. Context used: time of day, day of week and the current rating session. Device is NOT used.")
                 + banner(LABEL_CLOCK, "sim")
                 + f"<p>User {u}. Left: 10 minutes after their last activity (in session). Right: after a 3-day break (new session). {len(ids_n - ids_f)} of 18 titles differ.</p>"
                 + f"<div class='cols'><div><h3>mid-session (hour {near['clock']['hour_utc']:.0f} UTC)</h3>{table(rn, compare=ids_f)}</div>"
                 + f"<div><h3>after a 3-day break (hour {far['clock']['hour_utc']:.0f} UTC)</h3>{table(rf, compare=ids_n)}</div></div>")

    # B  M3b synthetic device effect
    progress("B: M3b, tv vs mobile (SIMULATED device, synthetic effect)", flush=True)
    tv = ask(client, {"user_id": u, "model": "M3b", "device": "tv", "seed": 1, "record_impressions": False})
    mob = ask(client, {"user_id": u, "model": "M3b", "device": "mobile", "seed": 1, "record_impressions": False})
    counts = {"M3b tv": [], "M3b mobile": [], "M2": []}
    for k, uid in enumerate(users, 1):
        for name, body in (("M3b tv", {"model": "M3b", "device": "tv"}), ("M3b mobile", {"model": "M3b", "device": "mobile"}), ("M2", {})):
            counts[name].append(sum(is_dnc(x) for x in ranked(ask(client, {"user_id": uid, "seed": 1, "record_impressions": False, **body}))))
        if k % 10 == 0:
            progress(f"  B aggregate: {k}/{len(users)} users", flush=True)
    means = {k: float(np.mean(v)) for k, v in counts.items()}
    summary["B_mean_drama_without_comedy_in_top18"] = means
    parts.append("<h2>B. Synthetic device scenario (M3b)</h2>" + banner(LABEL_M3B, "synthetic") + banner(LABEL_DEVICE, "sim")
                 + "<p>In training, on <b>mobile</b> requests 60% of the likes of Drama-without-Comedy titles were removed on purpose. M3b learned that planted rule; "
                   "nothing here says anything about real viewers. Highlighted rows are Drama-without-Comedy titles.</p>"
                 + f"<div class='cols'><div><h3>device = tv</h3>{table(ranked(tv), mark=is_dnc)}</div><div><h3>device = mobile</h3>{table(ranked(mob), mark=is_dnc)}</div></div>"
                 + f"<p>Average number of Drama-without-Comedy titles in the top 18 over {len(users)} users: M3b tv <b>{means['M3b tv']:.2f}</b>, "
                   f"M3b mobile <b>{means['M3b mobile']:.2f}</b>, M2 (ignores device) <b>{means['M2']:.2f}</b>.</p>")

    # C  live vs precomputed
    progress("C: live vs precomputed", flush=True)
    pre = ask(client, {"user_id": u, "seed": 1, "record_impressions": False})
    live = ask(client, {"user_id": u, "mode": "live", "seed": 1, "record_impressions": False})
    rp, rl = ranked(pre), ranked(live)
    ip, il = {x["movie_id"] for x in rp}, {x["movie_id"] for x in rl}
    summary["C_top18_overlap"] = len(ip & il)
    summary["C_live_fallback"] = live["diagnostics"].get("fallback")
    parts.append("<h2>C. Live vector search vs precomputed candidates</h2>"
                 + banner("Live: two searches in the Zilliz vector database at level 5 (approximate; about 92% of the exact candidates in our measurement). "
                          "Precomputed: the exact candidate list stored in DynamoDB. The same M2 ranker orders both.")
                 + f"<p>{len(ip & il)} of 18 titles are shared. Server time: precomputed {pre['timings_ms']['server_total']:.0f} ms, "
                   f"live {live['timings_ms']['server_total']:.0f} ms (of which search {live['timings_ms']['search']:.0f} ms).</p>"
                 + f"<div class='cols'><div><h3>precomputed</h3>{table(rp, compare=il)}</div><div><h3>live</h3>{table(rl, compare=ip)}</div></div>")

    # D  bandit simulation
    if args.bandit_requests:
        progress(f"D: bandit simulation, {args.bandit_requests} requests", flush=True)
        hidden = rates if rates is not None else hidden_like_rates(b)
        sim = np.random.default_rng(args.seed + 7)
        tally = {}
        for k in range(args.bandit_requests):
            resp = ask(client, {"user_id": users[k % len(users)], "gap_minutes": 10, "record_impressions": True})
            for x in resp["items"]:
                if x["source"].startswith("bandit"):
                    arm = int(b.cat_index([x["movie_id"]])[0])
                    reward = int(sim.random() < hidden[arm])
                    client.call("POST", "/feedback", {"arm_id": arm, "reward": reward})
                    t = tally.setdefault(arm, {"title": x["title"], "shown": 0, "liked": 0})
                    t["shown"] += 1
                    t["liked"] += reward
            if (k + 1) % 50 == 0:
                progress(f"  D: {k + 1}/{args.bandit_requests} requests", flush=True)
        top = sorted(tally.items(), key=lambda kv: -kv[1]["shown"])[:12]
        shown = sum(t["shown"] for t in tally.values())
        liked = sum(t["liked"] for t in tally.values())
        summary["D_impressions"], summary["D_simulated_like_rate"] = shown, (liked / shown if shown else None)
        rows = "".join(f"<tr><td>{html.escape(t['title'])}</td><td>{t['shown']}</td><td>{t['liked']}</td><td>{hidden[a_]:.2f}</td></tr>" for a_, t in top)
        parts.append("<h2>D. New-release slots: UCB bandit under simulated feedback</h2>" + banner(LABEL_BANDIT + ": every like below is SIMULATED", "sim")
                     + f"<p>{args.bandit_requests} requests; the 2 new-release slots of each list were chosen by UCB (c = 0.2) and a like was drawn from each title's hidden "
                       f"rate (estimated from MovieLens ratings). {shown} impressions, {liked} simulated likes ({(liked / shown if shown else 0):.0%}). Most-shown titles:</p>"
                     + f"<table><tr><th>title</th><th>shown</th><th>simulated likes</th><th>hidden rate</th></tr>{rows}</table>")

    page = (f"<!doctype html><html><head><meta charset='utf-8'><title>OTT recommender demo</title><style>{CSS}</style></head><body>"
            f"<h1>OTT recommender: live demo of the deployed service</h1><p>Generated {time.strftime('%Y-%m-%d %H:%M')} from requests to the AWS Lambda function in "
            f"{html.escape(args.region)} (IAM-signed Function URL). Data: MovieLens 25M sample of 20,000 users with TMDB descriptions.</p>"
            + "".join(parts)
            + f"<footer>{html.escape(LABEL_CLOCK)} {html.escape(LABEL_DEVICE)} MovieLens ratings are ratings, not viewing. Titles are MovieLens titles.</footer></body></html>")
    Path(args.out).write_text(page)
    Path(args.out).with_suffix(".json").write_text(json.dumps(summary, indent=1, default=float))
    print(f"demo written: {Path(args.out).name} (+ {Path(args.out).with_suffix('.json').name})")
    print(json.dumps(summary, indent=1, default=float))
    return 0


if __name__ == "__main__":
    sys.exit(main())
