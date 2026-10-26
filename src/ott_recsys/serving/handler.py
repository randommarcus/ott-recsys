"""AWS Lambda entry point and the request logic (Module 9).

Routes (Function URL, AuthType AWS_IAM):
  POST /recommend  {user_id, mode: precomputed|live, model: M2|M3b, device (M3b only, SIMULATED), gap_minutes}
  POST /feedback   {arm_id, reward: 0|1}   SIMULATED like/dislike for a new-release arm. SIMULATION, NOT AN A/B TEST.
  GET  /health

Rules: no secret is ever logged or returned (errors carry exception TYPE names only); every response carries its labels (see __init__).
Output list = 20 slots: positions 1-18 the ranker's best main-lane items, positions 19-20 chosen by the UCB bandit (c = 0.2) from the user's
eligible new-release candidates (released before the clock, within the 365-day shelf)."""
import base64
import json
import logging
import os
import time
from pathlib import Path

import numpy as np

from . import LABEL_BANDIT, LABEL_CLOCK, LABEL_DEVICE, LABEL_M3B
from .bundle import DEVICES, Bundle
from .features import build_request_features, hour_dow, matrix
from .search import SearchError, live_candidates
from .secrets import ConfigError
from .ucb import C_PREREGISTERED, pick

log = logging.getLogger("ott")
SHELF_S = 365 * 86400
N_SLOTS, N_BANDIT = 20, 2


class BadRequest(Exception):
    pass


class App:
    def __init__(self, bundle, users, bandit, searcher_factory=None, c: float = C_PREREGISTERED):
        self.b, self.users, self.bandit, self._factory, self._searcher, self.c = bundle, users, bandit, searcher_factory, None, c
        self.cold = True  # the first request in a container reports it

    # ------------------------------------------------------------------ parsing
    @staticmethod
    def _parse(body: dict) -> dict:
        try:
            uid = int(body["user_id"])
        except (KeyError, TypeError, ValueError):
            raise BadRequest("user_id (integer) is required") from None
        mode, model = body.get("mode", "precomputed"), body.get("model", "M2")
        if mode not in ("precomputed", "live"):
            raise BadRequest("mode must be precomputed or live")
        if model not in ("M2", "M3b"):
            raise BadRequest("model must be M2 or M3b")
        device = body.get("device")
        if model == "M3b" and device not in DEVICES:
            raise BadRequest("M3b needs a SIMULATED device: tv, mobile, web or tablet")
        try:
            gap = float(body.get("gap_minutes", 10))
        except (TypeError, ValueError):
            raise BadRequest("gap_minutes must be a number") from None
        if not (0.5 <= gap <= 60 * 24 * 365):
            raise BadRequest("gap_minutes must be between 0.5 and 525600")
        return {"user_id": uid, "mode": mode, "model": model, "device": device if model == "M3b" else None, "gap": gap,
                "record": bool(body.get("record_impressions", True)), "seed": body.get("seed")}

    def _live_searcher(self):
        if self._searcher is None:
            if self._factory is None:
                raise ConfigError("live mode is not configured")
            self._searcher = self._factory()
        return self._searcher

    # ------------------------------------------------------------------ /recommend
    def recommend(self, body: dict):
        t0 = time.perf_counter()
        q = self._parse(body)
        T = {}
        rec = self.users.get_user(q["user_id"])
        T["user_read"] = time.perf_counter() - t0
        if rec is None:
            return 404, {"error": "unknown user_id"}
        last_ts = int(rec["last_ts"])
        t_req = last_ts + max(1, int(round(q["gap"] * 60)))

        # ---- candidates
        info, fallback = {"searches": 0}, None
        t1 = time.perf_counter()
        cand = rec["cand"]
        if q["mode"] == "live":
            try:
                cand, info = live_candidates(self.b, rec, self._live_searcher())
            except (SearchError, ConfigError) as exc:
                fallback = f"live retrieval failed ({type(exc).__name__}); used the precomputed list"
                log.warning("live fallback: %s", type(exc).__name__)
                cand = rec["cand"]
        T["search"] = time.perf_counter() - t1
        if cand is None:
            return 500, {"error": "no candidate list for this user"}

        # ---- features and ranker
        t2 = time.perf_counter()
        model = self.b.rankers[q["model"]]
        f = build_request_features(self.b, rec, cand, t_req, device=q["device"])
        X = matrix(f, model.feature_names)
        T["features"] = time.perf_counter() - t2
        t3 = time.perf_counter()
        score = model.predict(X)
        T["ranker"] = time.perf_counter() - t3

        # ---- list assembly: 18 ranker slots + 2 bandit slots
        t4 = time.perf_counter()
        lane, rel, c = f["_lane"], f["_rel"], f["_cand"]
        main_rows = np.flatnonzero(~lane)
        main_rows = main_rows[np.argsort(-score[main_rows], kind="stable")][:N_SLOTS - N_BANDIT]
        eligible = np.flatnonzero(lane & ~np.isnan(rel) & (rel > t_req - SHELF_S))  # on the 365-day shelf (released <= t_req already)
        picks, arm_n, arm_s = np.empty(0, int), np.empty(0), np.empty(0)
        if len(eligible):
            arm_ids = c[eligible]
            arm_n, arm_s = self.bandit.get_arms(arm_ids)
            rng = np.random.default_rng(q["seed"]) if q["seed"] is not None else None
            picks = pick(arm_n, arm_s, N_BANDIT, self.c, rng)
            if q["record"]:
                self.bandit.add_impressions(arm_ids[picks])
        T["bandit"] = time.perf_counter() - t4

        titles = self.b.titles
        items = []
        for r in main_rows:
            items.append({"position": len(items) + 1, "movie_id": int(self.b.catalog[c[r]]), "title": titles[c[r]], "source": "ranker",
                          "new_release_lane": False, "score": round(float(score[r]), 5)})
        # when there are fewer than 18 main items the bandit slots simply follow
        for j in picks:
            r = eligible[j]
            items.append({"position": len(items) + 1, "movie_id": int(self.b.catalog[c[r]]), "title": titles[c[r]], "source": "bandit UCB c=%.1f" % self.c,
                          "new_release_lane": True, "score": round(float(score[r]), 5), "arm_impressions_before": int(arm_n[j]),
                          "arm_simulated_likes_before": int(arm_s[j])})
        labels = [LABEL_CLOCK]
        if q["model"] == "M3b":
            labels += [LABEL_DEVICE, LABEL_M3B]
        if len(picks):
            labels.append(LABEL_BANDIT + ": arm counts include SIMULATED feedback")
        hour, dow = hour_dow(t_req)
        total = time.perf_counter() - t0
        resp = {"user_id": q["user_id"], "mode": q["mode"], "model": q["model"], "device_simulated": q["device"], "items": items, "labels": labels,
                "clock": {"last_activity_ts": last_ts, "gap_minutes": q["gap"], "request_ts": t_req, "hour_utc": hour, "day_of_week": dow,
                          "in_session": bool(f["in_session"][0]) if len(c) else None},
                "timings_ms": {**{k: round(v * 1000, 3) for k, v in T.items()}, "server_total": round(total * 1000, 3)},
                "diagnostics": {"cold_start": self.cold, "candidates_in": int((np.asarray(cand) >= 0).sum()), "candidates_released": int(len(c)),
                                "main_eligible": int((~lane).sum()), "bandit_arms_eligible": int(len(eligible)), "searches": info.get("searches", 0),
                                "truncated_main": bool(info.get("truncated_main", False)), "fallback": fallback}}
        self.cold = False
        log.info("recommend ok mode=%s model=%s ms=%.1f", q["mode"], q["model"], total * 1000)
        return 200, resp

    # ------------------------------------------------------------------ /feedback
    def feedback(self, body: dict):
        try:
            arm, reward = int(body["arm_id"]), int(body["reward"])
        except (KeyError, TypeError, ValueError):
            raise BadRequest("arm_id (catalog index) and reward (0 or 1) are required") from None
        if reward not in (0, 1) or not (0 <= arm < len(self.b.catalog)) or self.b.known[arm]:
            raise BadRequest("arm_id must be a new-release (cold) catalog index and reward 0 or 1")
        self.bandit.add_feedback(arm, reward)
        return 200, {"ok": True, "labels": [LABEL_BANDIT + ": this like/dislike is SIMULATED"]}

    def health(self):
        return 200, {"ok": True, "models": sorted(self.b.rankers), "items": int(len(self.b.catalog)), "live_configured": self._factory is not None}


# ---------------------------------------------------------------------- Lambda glue
_APP = None


def _default_app() -> App:
    from .search import ZillizREST
    from .secrets import load_from_ssm
    from .store import DynamoStore
    bundle = Bundle(Path(os.environ.get("OTT_BUNDLE_DIR", str(Path(__file__).resolve().parents[2] / "bundle"))))
    store = DynamoStore(os.environ.get("OTT_USERS_TABLE", "ott-users"), os.environ.get("OTT_BANDIT_TABLE", "ott-bandit"))
    return App(bundle, store, store, searcher_factory=lambda: ZillizREST(load_from_ssm()))


def handle(app: App, method: str, path: str, body: dict):
    try:
        if method == "GET" and path == "/health":
            return app.health()
        if method == "POST" and path == "/recommend":
            return app.recommend(body)
        if method == "POST" and path == "/feedback":
            return app.feedback(body)
        return 404, {"error": "unknown route"}
    except BadRequest as bad:
        return 400, {"error": bad.args[0]}  # BadRequest is only ever raised with a literal message written in this file
    except Exception as exc:  # never echo the message: it could contain connection details
        log.error("request failed: %s", type(exc).__name__)
        return 500, {"error": "internal error", "type": type(exc).__name__}


def handler(event, context=None):
    global _APP
    if _APP is None:
        _APP = _default_app()
    http = (event.get("requestContext") or {}).get("http") or {}
    method, path = http.get("method", event.get("method", "POST")), event.get("rawPath", event.get("path", "/recommend"))
    raw = event.get("body") or "{}"
    if event.get("isBase64Encoded"):
        raw = base64.b64decode(raw).decode()
    try:
        body = json.loads(raw)
    except ValueError:
        return {"statusCode": 400, "headers": {"Content-Type": "application/json"}, "body": json.dumps({"error": "body must be JSON"})}
    status, payload = handle(_APP, method, path, body if isinstance(body, dict) else {})
    return {"statusCode": status, "headers": {"Content-Type": "application/json"}, "body": json.dumps(payload)}
