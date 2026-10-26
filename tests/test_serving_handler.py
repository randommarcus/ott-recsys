"""End-to-end local runs of the Lambda request logic on real users, with the in-memory twin of the DynamoDB tables and an exact numpy stand-in
for Zilliz. No AWS and no Zilliz involved."""
import base64
import json
from decimal import Decimal

import numpy as np
import pytest

from ott_recsys import context
from ott_recsys.serving import LABEL_BANDIT, LABEL_CLOCK, LABEL_DEVICE, LABEL_M3B
from ott_recsys.serving import handler as H
from ott_recsys.serving.bundle import DEVICES, Bundle
from ott_recsys.serving.records import pack, unpack
from ott_recsys.serving.search import NumpySearch, SearchError
from ott_recsys.serving.store import DynamoStore, MemoryStore
from ott_recsys.serving_offline import record_at

from .serving_real import HAVE_REAL, OUT

SV = OUT / "serving"
real = pytest.mark.skipif(not (HAVE_REAL and (SV / "users.npz").exists()), reason="serving artifacts not built")


@pytest.fixture(scope="module")
def world():
    b = Bundle(SV / "bundle")
    a = dict(np.load(SV / "users.npz"))
    return b, a


def make_app(world, users=range(0, 400)):
    b, a = world
    store = MemoryStore()
    for i in users:
        store.put_user(record_at(a, i))
    return H.App(b, store, store, searcher_factory=lambda: NumpySearch(b)), store


def post(app, route, body):
    return H.handle(app, "POST", route, body)


def uid(world, i):
    return int(world[1]["user_id"][i])


def find_user_with_arms(world, app, minimum=4):
    for i in range(400):
        s, r = post(app, "/recommend", {"user_id": uid(world, i), "record_impressions": False})
        if s == 200 and r["diagnostics"]["bandit_arms_eligible"] >= minimum and r["diagnostics"]["main_eligible"] >= 18:
            return i, r
    pytest.fail("no sampled user has enough eligible new-release arms")


@real
def test_list_layout_and_rules(world):
    b, a = world
    app, _ = make_app(world)
    i, r = find_user_with_arms(world, app)
    items = r["items"]
    assert [x["position"] for x in items] == list(range(1, 21))
    assert [x["source"] for x in items[:18]] == ["ranker"] * 18 and all(x["source"].startswith("bandit UCB c=0.2") for x in items[18:])
    assert [x["new_release_lane"] for x in items] == [False] * 18 + [True] * 2
    scores = [x["score"] for x in items[:18]]
    assert scores == sorted(scores, reverse=True)
    ids = [x["movie_id"] for x in items]
    assert len(set(ids)) == 20
    seen = set(a["seen_flat"][a["seen_off"][i]:a["seen_off"][i + 1]].tolist())
    cat = b.cat_index(ids)
    assert not seen & set(cat.tolist()), "an already-rated movie was recommended"
    t_req = r["clock"]["request_ts"]
    rel = b.rel_ts[cat]
    assert (rel[18:] <= t_req).all() and (rel[18:] > t_req - 365 * 86400).all(), "bandit slots must be on the 365-day shelf"
    ok = ~np.isnan(rel)
    assert (rel[ok] <= t_req).all(), "an unreleased movie was recommended"
    assert LABEL_CLOCK in r["labels"] and any(LABEL_BANDIT in x for x in r["labels"])
    assert set(r["timings_ms"]) >= {"user_read", "search", "features", "ranker", "bandit", "server_total"}
    assert json.dumps(r)  # serialisable


@real
def test_user_without_eligible_new_releases_gets_twenty_ranker_slots(world):
    app, _ = make_app(world)
    for i in range(400):
        s, r = post(app, "/recommend", {"user_id": uid(world, i), "record_impressions": False})
        if r["diagnostics"]["bandit_arms_eligible"] == 0 and r["diagnostics"]["main_eligible"] >= 20:
            assert len(r["items"]) == 20 and all(x["source"] == "ranker" for x in r["items"])
            return
    pytest.fail("no sampled user without eligible new releases")


@real
def test_bandit_counts_and_exploration(world):
    app, store = make_app(world)
    i, _ = find_user_with_arms(world, app)
    u = uid(world, i)
    s1, r1 = post(app, "/recommend", {"user_id": u, "seed": 1})
    arms1 = [world[0].cat_index([x["movie_id"]])[0] for x in r1["items"][18:]]
    n, s = store.get_arms(arms1)
    assert n.tolist() == [1, 1] and s.tolist() == [0, 0]
    s2, r2 = post(app, "/recommend", {"user_id": u, "seed": 2})
    arms2 = [world[0].cat_index([x["movie_id"]])[0] for x in r2["items"][18:]]
    assert not set(arms1) & set(arms2), "UCB must try never-shown arms before repeating one"
    before = store.get_arms(arms2)[0].tolist()
    post(app, "/recommend", {"user_id": u, "seed": 3, "record_impressions": False})
    assert store.get_arms(arms2)[0].tolist() == before, "record_impressions=false must not change the counts"
    status, fb = post(app, "/feedback", {"arm_id": int(arms1[0]), "reward": 1})
    assert status == 200 and "SIMULATED" in fb["labels"][0] and store.get_arms([arms1[0]])[1].tolist() == [1]


@real
def test_seeded_requests_are_reproducible(world):
    app1, _ = make_app(world, range(0, 60))
    app2, _ = make_app(world, range(0, 60))
    u = uid(world, 3)
    r1 = post(app1, "/recommend", {"user_id": u, "seed": 7})[1]
    r2 = post(app2, "/recommend", {"user_id": u, "seed": 7})[1]
    assert [x["movie_id"] for x in r1["items"]] == [x["movie_id"] for x in r2["items"]]


@real
def test_m3b_is_labelled_and_reacts_to_the_simulated_device(world):
    app, _ = make_app(world, range(0, 150))
    assert "M3b" in world[0].rankers, "run scripts/refit_m3b.py and rebuild the bundle"
    s, r = post(app, "/recommend", {"user_id": uid(world, 0), "model": "M3b"})
    assert s == 400 and "SIMULATED" in r["error"]
    changed = 0
    for i in range(100):
        out = {d: post(app, "/recommend", {"user_id": uid(world, i), "model": "M3b", "device": d, "record_impressions": False})[1] for d in ("tv", "mobile")}
        assert all(LABEL_M3B in o["labels"] and LABEL_DEVICE in o["labels"] for o in out.values())
        assert out["tv"]["device_simulated"] == "tv"
        changed += [x["movie_id"] for x in out["tv"]["items"][:18]] != [x["movie_id"] for x in out["mobile"]["items"][:18]]
    assert changed > 0, "the synthetic device column should change at least some lists"
    assert LABEL_M3B == "Synthetic device effect, pipeline demonstration, not a finding"
    s, r = post(app, "/recommend", {"user_id": uid(world, 0), "model": "M2", "device": "mobile"})
    assert r["device_simulated"] is None and LABEL_M3B not in r["labels"]  # M2 never uses a device


@real
def test_gap_controls_the_session(world):
    app, _ = make_app(world, range(0, 100))
    flips = 0
    for i in range(60):
        a_ = post(app, "/recommend", {"user_id": uid(world, i), "gap_minutes": 10, "record_impressions": False})[1]
        b_ = post(app, "/recommend", {"user_id": uid(world, i), "gap_minutes": 3 * 24 * 60, "record_impressions": False})[1]
        assert a_["clock"]["in_session"] is True and b_["clock"]["in_session"] is False
        assert b_["clock"]["request_ts"] - a_["clock"]["request_ts"] == (3 * 24 * 60 - 10) * 60
        flips += [x["movie_id"] for x in a_["items"][:18]] != [x["movie_id"] for x in b_["items"][:18]]
    assert flips > 0, "session context should change at least some lists"
    h, d = a_["clock"]["hour_utc"], a_["clock"]["day_of_week"]
    import pandas as pd
    ts = pd.to_datetime(a_["clock"]["request_ts"], unit="s")
    assert (h, d) == (ts.hour, ts.day_of_week)


@real
def test_live_mode_gives_the_same_lists_as_precomputed(world):
    app, _ = make_app(world, range(0, 120))
    same = 0
    for i in range(60):
        p = post(app, "/recommend", {"user_id": uid(world, i), "record_impressions": False, "seed": 1})[1]
        l = post(app, "/recommend", {"user_id": uid(world, i), "mode": "live", "record_impressions": False, "seed": 1})[1]
        assert l["diagnostics"]["fallback"] is None and l["diagnostics"]["searches"] in (1, 2)
        same += [x["movie_id"] for x in p["items"]] == [x["movie_id"] for x in l["items"]]
    assert same >= 57, f"live and precomputed lists agree for only {same} of 60 users"


@real
def test_user_without_a_vector_gets_a_popularity_based_list(world):
    b, a = world
    i = int(np.flatnonzero(~a["has_ttvec"])[0])
    app, _ = make_app(world, [i])
    for mode in ("precomputed", "live"):
        s, r = post(app, "/recommend", {"user_id": uid(world, i), "mode": mode})
        assert s == 200 and len(r["items"]) >= 18 and r["diagnostics"]["fallback"] is None


@real
def test_live_search_failure_falls_back_to_the_precomputed_list(world):
    b, a = world
    store = MemoryStore()
    store.put_user(record_at(a, 0))
    class Down:
        def search(self, *args, **kw): raise SearchError("URLError")
    app = H.App(b, store, store, searcher_factory=lambda: Down())
    s, r = post(app, "/recommend", {"user_id": uid(world, 0), "mode": "live"})
    s0, r0 = post(app, "/recommend", {"user_id": uid(world, 0), "record_impressions": False, "seed": 1})
    assert s == 200 and "SearchError" in r["diagnostics"]["fallback"]
    assert [x["movie_id"] for x in r["items"][:18]] == [x["movie_id"] for x in r0["items"][:18]]


@real
def test_bad_requests(world):
    app, _ = make_app(world, range(0, 5))
    u = uid(world, 0)
    assert post(app, "/recommend", {"user_id": 999999999})[0] == 404
    for body in ({}, {"user_id": "x"}, {"user_id": u, "mode": "magic"}, {"user_id": u, "model": "M9"}, {"user_id": u, "gap_minutes": 0},
                 {"user_id": u, "gap_minutes": "soon"}, {"user_id": u, "gap_minutes": 10 ** 9}):
        assert post(app, "/recommend", body)[0] == 400, body
    known = int(np.flatnonzero(world[0].known)[0])
    for body in ({}, {"arm_id": known, "reward": 1}, {"arm_id": -1, "reward": 1}, {"arm_id": int(np.flatnonzero(~world[0].known)[0]), "reward": 2}):
        assert post(app, "/feedback", body)[0] == 400, body
    assert H.handle(app, "GET", "/health", {})[0] == 200 and H.handle(app, "GET", "/nope", {})[0] == 404


@real
def test_lambda_event_shapes(world):
    app, _ = make_app(world, range(0, 5))
    H._APP = app
    try:
        body = json.dumps({"user_id": uid(world, 0), "seed": 1})
        r = H.handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/recommend", "body": body})
        r64 = H.handler({"requestContext": {"http": {"method": "POST"}}, "rawPath": "/recommend", "isBase64Encoded": True,
                         "body": base64.b64encode(body.encode()).decode()})
        assert r["statusCode"] == r64["statusCode"] == 200 and len(json.loads(r["body"])["items"]) == 20
        assert H.handler({"requestContext": {"http": {"method": "GET"}}, "rawPath": "/health"})["statusCode"] == 200
    finally:
        H._APP = None


# ---------------------------------------------------------------------------------------------- DynamoDB contract (fake client, no AWS)
class _Binary:
    def __init__(self, v): self.value = v


class _FakeTable:
    def __init__(self, store, key): self.store, self.key = store, key

    def get_item(self, Key):
        item = self.store.get(Key[self.key])
        return {"Item": item} if item is not None else {}

    def update_item(self, Key, UpdateExpression, ExpressionAttributeValues):
        assert all(isinstance(v, int) and not isinstance(v, float) for v in ExpressionAttributeValues.values()), "DynamoDB rejects floats"
        assert UpdateExpression.startswith("ADD ")
        field = UpdateExpression.split()[1]
        cur = self.store.setdefault(Key[self.key], {self.key: Decimal(Key[self.key])})
        cur[field] = cur.get(field, Decimal(0)) + Decimal(list(ExpressionAttributeValues.values())[0])


class _FakeResource:
    def __init__(self):
        self.users, self.arms = {}, {}
        self.batch_calls = 0

    def Table(self, name): return _FakeTable(self.users, "user_id") if name == "ott-users" else _FakeTable(self.arms, "arm_id")

    def batch_get_item(self, RequestItems):
        self.batch_calls += 1
        (name, spec), = RequestItems.items()
        assert name == "ott-bandit" and len(spec["Keys"]) <= 100
        return {"Responses": {name: [self.arms[k["arm_id"]] for k in spec["Keys"] if k["arm_id"] in self.arms]}}


@real
def test_dynamo_store_contract_and_item_sizes(world):
    b, a = world
    res = _FakeResource()
    ds = DynamoStore(resource=res)
    rec = record_at(a, 0)
    packed = pack(rec)
    assert all(isinstance(v, (int, bytes)) and not isinstance(v, float) for v in packed.values()), "only ints and bytes may be stored"
    res.users[rec["user_id"]] = {k: (Decimal(v) if isinstance(v, int) else _Binary(v)) for k, v in packed.items()}
    back = ds.get_user(rec["user_id"])
    for k in ("cand", "ttvec", "prof", "seen", "ncf_ug"):
        assert np.array_equal(back[k], rec[k], equal_nan=True) if rec[k] is not None else back[k] is None
    assert ds.get_user(-5) is None
    ds.add_impressions([7, 7, 9])
    ds.add_feedback(7, 1)
    n, s = ds.get_arms([7, 9, 11])
    assert n.tolist() == [2, 1, 0] and s.tolist() == [1, 0, 0]
    ds.get_arms(list(range(250)))
    assert res.batch_calls == 1 + 3, "BatchGetItem takes 100 keys per call"
    sizes = [sum(len(v) if isinstance(v, bytes) else 8 for v in pack(record_at(a, i)).values()) for i in range(0, len(a["user_id"]), 7)]
    assert max(sizes) < 400_000, "a DynamoDB item may not exceed 400 KB"
    print(f"record bytes: mean {np.mean(sizes):.0f}, max {max(sizes)}")


def test_record_roundtrip_and_device_order():
    rec = {"user_id": 5, "cand": np.array([3, -1, 7], np.int32), "last_ts": 1_500_000_000, "sess_len": 4, "sess_likes": 2, "ncf_known": 1,
           "ttvec": np.arange(4, dtype=np.float32), "prof": None, "seen": np.array([1, 2, 3], np.int32), "sess_items": np.array([], np.int32)}
    back = unpack(pack(rec))
    assert back["user_id"] == 5 and np.array_equal(back["cand"], rec["cand"]) and back["prof"] is None and back["sess_items"] is None
    assert np.array_equal(back["ttvec"], rec["ttvec"]) and back["ncf_known"] == 1
    assert DEVICES == context.DEVICES, "device one-hot order must match the order the M3b model was trained with"
